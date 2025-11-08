#!/usr/bin/env python3
"""
harvest_oai.py — Cosecha OAI-PMH (ListRecords) y guarda:
  - un XML por registro en raw/oai/<RI>/
  - un headers.jsonl con metadatos mínimos (oai_identifier, datestamp, setSpecs, status, etc.)

Uso típico:
  python src/harvest_oai.py --ri UADY --prefix oai_dc --from-days 30
  # o con fecha ISO explícita:
  python src/harvest_oai.py --ri UADY --prefix oai_dc --from-iso 2025-01-01

Requisitos:
  - configs/origins.yaml con el bloque del RI (name, oai_baseurl, harvest_window_days opcional).
  - Paquetes: sickle, pyyaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from sickle import Sickle
from sickle.models import Record


# --- Rutas del proyecto (relativas a la raíz del repo) ---
RAW_OAI = Path("raw/oai")
STATE = Path("state")
LOGS = Path("logs")


def iso_today_date() -> str:
    """Fecha en formato OAI (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def iso_days_ago(days: int) -> str:
    """Devuelve YYYY-MM-DD para hoy - days (UTC)."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


def ensure_dirs(ri_name: str) -> Path:
    """Crea carpetas destino y retorna el directorio para el RI."""
    out_dir = RAW_OAI / ri_name
    out_dir.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_id(text: str) -> str:
    """
    Genera un nombre de archivo seguro a partir del OAI identifier.
    Conserva algo legible y añade un hash corto para evitar colisiones.
    """
    base = re.sub(r"[^a-zA-Z0-9_.-]", "_", text).strip("_") or "rec"
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{base}__{h}"


def load_origin(ri: str) -> Dict[str, Any]:
    """Carga el bloque del RI desde configs/origins.yaml (búsqueda case-insensitive)."""
    cfg_path = Path("configs/origins.yaml")
    if not cfg_path.exists():
        raise SystemExit("ERROR: no existe configs/origins.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for repo in cfg.get("repos", []):
        if repo.get("name", "").lower() == ri.lower():
            return repo
    raise SystemExit(f"ERROR: RI '{ri}' no está en configs/origins.yaml")


def build_oai_params(
    prefix: str,
    from_iso: str | None,
    from_days: int | None,
    default_days: int,
    set_spec: str | None,
    until_iso: str | None,
) -> Dict[str, Any]:
    """
    Construye parámetros para ListRecords:
      - Prioridad: --from-iso > --from-days > harvest_window_days (config) > None
      - Por defecto no pasamos 'until' (será 'hasta ahora' según el servidor).
    """
    params: Dict[str, Any] = {"metadataPrefix": prefix}
    # FROM
    if from_iso:
        params["from"] = from_iso
    elif from_days is not None:
        params["from"] = iso_days_ago(from_days)
    elif default_days > 0:
        params["from"] = iso_days_ago(default_days)
    # SET (opcional)
    if set_spec:
        params["set"] = set_spec
    # UNTIL (opcional)
    if until_iso:
        params["until"] = until_iso
    return params


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cosecha OAI-PMH (ListRecords) y guarda XML+headers.jsonl"
    )
    ap.add_argument(
        "--ri",
        required=True,
        help="Nombre del RI como aparece en configs/origins.yaml (e.g., UADY)",
    )
    ap.add_argument(
        "--prefix", default="oai_dc", help="metadataPrefix (por defecto: oai_dc)"
    )
    ap.add_argument(
        "--from-iso", help="Fecha inicial ISO OAI (YYYY-MM-DD o YYYY-MM-DDThh:mm:ssZ)"
    )
    ap.add_argument("--from-days", type=int, help="Ventana relativa en días (hoy - N)")
    ap.add_argument("--until-iso", help="Fecha final ISO OAI (opcional)")
    ap.add_argument("--set", dest="set_spec", help="setSpec OAI (opcional)")
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Límite duro para pruebas (0 = sin límite)",
    )
    args = ap.parse_args()

    # 1) Config del RI
    origin = load_origin(args.ri)
    baseurl = origin.get("oai_baseurl")
    if not baseurl:
        raise SystemExit("ERROR: 'oai_baseurl' faltante en configs/origins.yaml")
    default_days = int(origin.get("harvest_window_days", 30))

    # 2) Directorios y archivos de salida
    out_dir = ensure_dirs(origin["name"])
    headers_path = out_dir / "headers.jsonl"

    # 3) Conectar con el servidor OAI
    # Nota: mantenemos configuración simple y robusta; Sickle maneja resumptionToken internamente.
    sickle = Sickle(baseurl)

    # 4) Parámetros de ListRecords
    params = build_oai_params(
        prefix=args.prefix,
        from_iso=args.from_iso,
        from_days=args.from_days,
        default_days=default_days,
        set_spec=args.set_spec,
        until_iso=args.until_iso,
    )

    print(f"[harvest] RI={origin['name']} baseurl={baseurl}")
    print(f"[harvest] Params={params}")

    # 5) Iterar registros y guardar
    count = 0
    with headers_path.open("a", encoding="utf-8") as hlog:
        try:
            it = sickle.ListRecords(**params)
            for item in it:
                # item es sickle.models.Record (o compatible)
                rec: Record = item  # ayuda de tipos
                head = rec.header
                oai_identifier = head.identifier
                status = "deleted" if head.deleted else "active"

                # Serializamos el <record> tal como llega.
                xml_text = rec.raw
                fname = safe_id(oai_identifier)
                xml_path = out_dir / f"{fname}.xml"
                xml_path.write_text(xml_text, encoding="utf-8")

                # Header mínimo (JSON Lines)
                hdr = {
                    "ri_name": origin["name"],
                    "prefix": args.prefix,
                    "oai_identifier": oai_identifier,
                    "datestamp": head.datestamp,
                    "setSpecs": list(head.setSpecs) if head.setSpecs else [],
                    "status": status,
                    "source_xml_path": str(xml_path),
                    "harvested_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
                hlog.write(json.dumps(hdr, ensure_ascii=False) + "\n")

                count += 1
                if count % 50 == 0:
                    print(f"[harvest] {count} registros...")
                if args.max_records and count >= args.max_records:
                    print(
                        f"[harvest] Límite de --max-records alcanzado: {args.max_records}"
                    )
                    break

        except Exception as e:
            # Log de error sencillo (sin detener el proceso si ya hubo descargas)
            err_path = LOGS / "harvest_errors.log"
            with err_path.open("a", encoding="utf-8") as ferr:
                ferr.write(
                    json.dumps(
                        {
                            "ri": origin["name"],
                            "params": params,
                            "error": repr(e),
                            "when": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[harvest] ERROR: {e!r} (ver {err_path})")

    print(
        f"[harvest] Terminado. Registros guardados: {count}. XML en {out_dir}, headers en {headers_path}"
    )


if __name__ == "__main__":
    main()
