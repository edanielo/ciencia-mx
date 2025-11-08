#!/usr/bin/env python3
"""
extract_text.py — Extrae texto de PDFs del spool con Tika y aplica OCR selectivo.

Flujo:
  1) Lee "spool/bin/*.pdf" (u otros binarios permitidos que tengas).
  2) Llama a Tika (sin OCR) y calcula métricas de calidad.
  3) Si la calidad es baja, reintenta con OCR ("ocr_and_text", idioma configurable).
  4) Normaliza texto (UTF-8 NFC, controla saltos y espacios).
  5) Guarda en corpus/raw/<RI>/<sha256>.txt y registra en corpus/raw/<RI>/manifest.jsonl.
  6) Borra el binario del spool si no usas --keep-spool.

Requisitos:
  - Tika Server escuchando en http://localhost:9998 (ver README / docker run).
  - fetch.log con entradas 'ok' para mapear path→sha256/url.
  - environment.yml con requests, lxml, pyyaml, etc.

Uso:
  conda activate cienciamx
  python src/extract_text.py --ri UADY --tika http://localhost:9998 --lang "spa+eng" --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
import yaml

# --- Rutas del proyecto ---
SPOOL = Path("spool/bin")
LOGS = Path("logs")
CORPUS = Path("corpus/raw")
RAW_OAI = Path("raw/oai")


@dataclass
class FetchMeta:
    path: str
    sha256: str
    url: Optional[str]
    content_type: Optional[str]
    bytes: Optional[int]
    ri: Optional[str]


def load_origin(ri: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path("configs/origins.yaml").read_text(encoding="utf-8"))
    for repo in cfg.get("repos", []):
        if repo.get("name", "").lower() == ri.lower():
            return repo
    raise SystemExit(f"RI no encontrado en configs/origins.yaml: {ri}")


def load_fetch_log() -> Dict[str, FetchMeta]:
    """
    Devuelve un índice por path (relativo como queda en el log: 'spool/bin/archivo.pdf')
    a metadatos de descarga (sha256, url, content_type, bytes, ri).
    """
    idx: Dict[str, FetchMeta] = {}
    flog = LOGS / "fetch.log"
    if not flog.exists():
        return idx
    for line in flog.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "ok":
            continue
        path = rec.get("path") or rec.get("spool_path") or ""
        if not path:
            continue
        # Normalizamos a ruta relativa tipo "spool/bin/archivo.pdf"
        path_rel = str(Path(path))
        idx[path_rel] = FetchMeta(
            path=path_rel,
            sha256=rec.get("sha256") or "",
            url=rec.get("url"),
            content_type=rec.get("content_type"),
            bytes=rec.get("bytes"),
            ri=rec.get("ri"),
        )
    return idx


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(s: str) -> str:
    # Unicode NFC
    s = unicodedata.normalize("NFC", s)
    # Saltos de línea consistentes
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Quita control chars no imprimibles salvo \n y \t
    s = re.sub(r"[^\x09\x0A\x20-\x7E\u0080-\uFFFF]", "", s)
    # Colapsa más de 2 saltos en 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Quita espacios en blanco muy repetidos
    s = re.sub(r"[ \t]{3,}", "  ", s)
    return s.strip()


def quality_metrics(text: str) -> Dict[str, Any]:
    n_chars = len(text)
    letters = sum(ch.isalpha() for ch in text)
    alpha_ratio = (letters / n_chars) if n_chars else 0.0
    # tokens: cuenta "palabras" aprox. con \w (incluye letras con acento)
    tokens = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    # score simple 0..1 (mezcla de riqueza y densidad alfabética)
    score = min(1.0, round(0.6 * alpha_ratio + 0.4 * min(1.0, tokens / 500.0), 3))
    return {
        "n_chars": n_chars,
        "tokens": tokens,
        "alpha_ratio": round(alpha_ratio, 4),
        "text_quality_score": score,
    }


def tika_extract_text(
    pdf_path: Path,
    tika_url: str,
    ocr_strategy: str,  # "no_ocr" | "ocr_and_text" | "ocr_only"
    ocr_lang: str,
    timeout: int = 180,
) -> Tuple[str, Dict[str, Any]]:
    """
    Llama a Tika Server /tika (PUT) con cabeceras para controlar OCR en PDFs.
    Devuelve (texto, {'status_code': int, 'tika_ocr_strategy': str}).
    """
    endpoint = f"{tika_url.rstrip('/')}/tika"
    headers = {
        "Accept": "text/plain",
        "Content-Type": "application/pdf",
        # Estrategia de OCR para PDFs
        "X-Tika-PDFOcrStrategy": ocr_strategy,
        # Idioma para OCR (Tesseract): "spa+eng" por ejemplo
        "X-Tika-OCRLanguage": ocr_lang,
    }
    with pdf_path.open("rb") as f:
        resp = requests.put(endpoint, headers=headers, data=f, timeout=timeout)
    if resp.status_code == 200:
        return resp.text or "", {
            "status_code": resp.status_code,
            "tika_ocr_strategy": ocr_strategy,
        }
    # Errores comunes: 415, 422, 500
    return "", {
        "status_code": resp.status_code,
        "tika_ocr_strategy": ocr_strategy,
        "error": resp.text[:4000],
    }


def should_apply_ocr(
    metrics: Dict[str, Any], min_tokens: int, min_alpha_ratio: float
) -> bool:
    return metrics["tokens"] < min_tokens or metrics["alpha_ratio"] < min_alpha_ratio


def iter_spool_files(limit: int | None = None) -> Iterable[Path]:
    files = sorted(SPOOL.glob("*"))
    if limit is None or limit <= 0:
        yield from files
    else:
        for p in files[:limit]:
            yield p


def ensure_dirs(ri: str) -> Path:
    out_dir = CORPUS / ri
    out_dir.mkdir(parents=True, exist_ok=True)
    (LOGS).mkdir(parents=True, exist_ok=True)
    return out_dir


def append_manifest(ri: str, rec: Dict[str, Any]) -> None:
    mpath = CORPUS / ri / "manifest.jsonl"
    with mpath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extrae texto del spool con Tika (OCR selectivo, normalización y métricas)."
    )
    ap.add_argument(
        "--ri", required=True, help="Nombre del RI (como en configs/origins.yaml)."
    )
    ap.add_argument(
        "--tika",
        default="http://localhost:9998",
        help="Base URL de Tika Server (default: http://localhost:9998)",
    )
    ap.add_argument(
        "--lang", default="spa+eng", help="Idiomas OCR (Tesseract), ej. 'spa+eng'"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Máximo de archivos a procesar (0 = todos)"
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=120,
        help="Umbral mínimo de tokens para evitar OCR",
    )
    ap.add_argument(
        "--min-alpha",
        type=float,
        default=0.20,
        help="Umbral mínimo alpha_ratio para evitar OCR",
    )
    ap.add_argument(
        "--keep-spool",
        action="store_true",
        help="No borrar el binario del spool tras extraer",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnóstico de arranque y progreso",
    )
    args = ap.parse_args()

    # valida que el RI existe (error temprano si no)
    load_origin(args.ri)
    out_dir = ensure_dirs(args.ri)
    fetch_idx = load_fetch_log()

    # Sanity check Tika
    try:
        r = requests.get(args.tika.rstrip("/") + "/version", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(
            f"[extract] ERROR: Tika no responde en {args.tika} → {e}", file=sys.stderr
        )
        print(
            "[extract] Sugerencia: docker run -d --name tika -p 9998:9998 apache/tika:latest-full"
        )
        sys.exit(1)

    # Listado de archivos del spool
    spool_files = [p for p in sorted(SPOOL.glob("*")) if p.is_file()]
    if args.limit and args.limit > 0:
        spool_files = spool_files[: args.limit]
    if args.verbose:
        print(
            f"[extract] RI={args.ri}  Tika={args.tika}  Spool_files={len(spool_files)}  Out={out_dir}"
        )
    if not spool_files:
        print(
            f"[extract] No hay archivos en {SPOOL}. Ejecuta resolve_fetch.py primero.",
            file=sys.stderr,
        )
        print(
            f"[extract] Ejemplo: python src/resolve_fetch.py --ri {args.ri} --limit 10"
        )
        sys.exit(0)

    processed = 0
    ok = 0

    for pdf in spool_files:
        processed += 1
        if args.verbose:
            print(f"[extract] Procesando {processed}/{len(spool_files)} → {pdf.name}")

        # Metadatos desde fetch.log (si existen)
        key = str(pdf)
        meta = fetch_idx.get(key)
        sha = meta.sha256 if (meta and meta.sha256) else compute_sha256(pdf)
        url = meta.url if meta else None
        ctype = meta.content_type if meta else None
        size = meta.bytes if meta else pdf.stat().st_size

        # Primer intento: sin OCR
        text, info = tika_extract_text(
            pdf, args.tika, ocr_strategy="no_ocr", ocr_lang=args.lang
        )
        text_norm = normalize_text(text)
        qm = quality_metrics(text_norm)

        ocr_applied = False
        ocr_strategy = "no_ocr"

        if should_apply_ocr(qm, args.min_tokens, args.min_alpha):
            # Reintento con OCR
            text2, info2 = tika_extract_text(
                pdf, args.tika, ocr_strategy="ocr_and_text", ocr_lang=args.lang
            )
            text2_norm = normalize_text(text2)
            qm2 = quality_metrics(text2_norm)

            # Si mejora (más tokens o mejor score), nos quedamos con el OCR
            if (qm2["tokens"] > qm["tokens"]) or (
                qm2["text_quality_score"] > qm["text_quality_score"]
            ):
                text_norm, qm = text2_norm, qm2
                ocr_applied = True
                ocr_strategy = "ocr_and_text"

        # Guardado
        out_txt = out_dir / f"{sha}.txt"
        out_txt.write_text(text_norm, encoding="utf-8")

        # Manifiesto
        rec = {
            "ri": args.ri,
            "sha256": sha,
            "source_spool_path": str(pdf),
            "source_url": url,
            "content_type": ctype,
            "bytes": size,
            "ocr_applied": ocr_applied,
            "ocr_strategy": ocr_strategy,
            "metrics": qm,
            "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tika_url": args.tika,
        }
        append_manifest(args.ri, rec)

        # Borrado del spool (si no se pide conservar)
        if not args.keep_spool:
            try:
                pdf.unlink(missing_ok=True)
            except Exception:
                pass

        ok += 1
        if ok % 10 == 0:
            print(f"[extract] {ok} archivos extraídos → {out_dir}")

    print(
        f"[extract] Terminado. Procesados: {processed}, extraídos: {ok}. Salida: {out_dir}"
    )


if __name__ == "__main__":
    main()
