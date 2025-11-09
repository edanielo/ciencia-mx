#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import hashlib
import sys
import zipfile
from pathlib import Path
from datetime import datetime

OUT = Path("out/solr")
CORPUS_RAW = Path("corpus/raw")

REQUIRED = {
    "id": str,
    "ri_s": str,
    "sha256_s": str,
    "ocr_b": bool,
    "tokens_i": int,
    "alpha_ratio_f": (int, float),
    "score_f": (int, float),
    "text_path_s": str,
}
OPTIONAL = {
    "url_s": str,
    "year_i": int,
    "doi_s": str,
    "persons_ss": list,
    "orgs_ss": list,
    "places_ss": list,
    "keyphrases_ss": list,
    "text_t": str,  # solo si empacaste con --include-text
}

SCHEMA_CONTRACT = {
    "id": {"type": "string", "desc": "sha256 del documento"},
    "ri_s": {"type": "string", "desc": "nombre del RI de origen"},
    "sha256_s": {"type": "string", "desc": "sha256 del binario/texto"},
    "url_s": {"type": "string?", "desc": "URL fuente si disponible"},
    "ocr_b": {"type": "boolean", "desc": "si se aplicó OCR"},
    "tokens_i": {"type": "int", "desc": "conteo de tokens"},
    "alpha_ratio_f": {"type": "float", "desc": "razón alfabética"},
    "score_f": {"type": "float", "desc": "score de calidad de texto"},
    "year_i": {"type": "int?", "desc": "año heurístico"},
    "doi_s": {"type": "string?", "desc": "DOI heurístico"},
    "persons_ss": {"type": "strings[]", "desc": "personas (NER)"},
    "orgs_ss": {"type": "strings[]", "desc": "organizaciones (NER)"},
    "places_ss": {"type": "strings[]", "desc": "lugares (NER)"},
    "keyphrases_ss": {"type": "strings[]", "desc": "keyphrases (YAKE)"},
    "text_path_s": {"type": "string", "desc": "ruta al .txt en corpus/raw"},
    "text_t": {"type": "text?", "desc": "texto truncado (opcional)"},
}

README_SOLR = """# Bundle Solr-ready

Este paquete contiene:
- `docs*.jsonl`: documentos 1-por-línea para indexación
- `schema_contract.json`: contrato de campos y tipos
- `CHECKSUMS.sha256`: sumas de verificación
- Este README

## Indexación rápida (ejemplo con docker solr:9)
docker run -d --name solr -p 8983:8983 solr:9
docker exec -it solr solr create -c cienciamx_<RI>
# Copia el JSONL al contenedor (o monta volumen) y postea:
# docker cp docs.jsonl solr:/var/solr/data/
# docker exec -it solr bin/post -c cienciamx_<RI> /var/solr/data/docs.jsonl
"""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_doc(d: dict, check_text_path: bool) -> list[str]:
    errs = []
    # Required types
    for k, tp in REQUIRED.items():
        if k not in d:
            errs.append(f"missing {k}")
            continue
        v = d[k]
        if tp is int:
            if not isinstance(v, int):
                errs.append(f"type {k} expected int got {type(v).__name__}")
        elif tp is bool:
            if not isinstance(v, bool):
                errs.append(f"type {k} expected bool got {type(v).__name__}")
        elif tp is str:
            if not isinstance(v, str):
                errs.append(f"type {k} expected str got {type(v).__name__}")
        else:  # tuple
            if not isinstance(v, tp):
                errs.append(f"type {k} expected {tp} got {type(v).__name__}")
    # Optional basic checks
    for k, tp in OPTIONAL.items():
        if k in d and not isinstance(d[k], tp):
            errs.append(f"type {k} expected {tp} got {type(d[k]).__name__}")
    # text_path exists?
    if check_text_path and "text_path_s" in d:
        if not Path(d["text_path_s"]).exists():
            errs.append(f"text_path_s not found: {d['text_path_s']}")
    return errs


def split_jsonl(src: Path, dst_dir: Path, max_per_file: int) -> list[Path]:
    outs = []
    if max_per_file <= 0:
        outs.append(src)
        return outs
    base = src.stem
    part, lines = 1, 0
    out = None
    with src.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if lines == 0:
                out = (dst_dir / f"{base}-part{part:04d}.jsonl").open(
                    "w", encoding="utf-8"
                )
                outs.append(out.name)  # store path str temporarily
            out.write(line)
            lines += 1
            if lines >= max_per_file:
                out.close()
                part += 1
                lines = 0
        if out and not out.closed:
            out.close()
    return [Path(p) for p in outs]


def main():
    ap = argparse.ArgumentParser(
        description="Valida, parte y empaqueta docs.jsonl para entrega a Solr"
    )
    ap.add_argument("--ri", required=True)
    ap.add_argument(
        "--check-text-path",
        action="store_true",
        help="Verifica existencia de text_path_s",
    )
    ap.add_argument(
        "--max-per-file",
        type=int,
        default=0,
        help="Partir en archivos de N docs (0 = no partir)",
    )
    ap.add_argument(
        "--bundle-name", default="", help="Nombre base del zip (auto por defecto)"
    )
    args = ap.parse_args()

    ri = args.ri
    solr_dir = OUT / ri
    src = solr_dir / "docs.jsonl"
    if not src.exists():
        sys.exit(f"ERROR: no existe {src}. Corre pack_solr_jsonl.py primero.")

    # Validación
    ids = set()
    errors = 0
    total = 0
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[validate] L{i}: JSON inválido -> {e}", file=sys.stderr)
                errors += 1
                continue
            # id único
            _id = d.get("id")
            if not isinstance(_id, str):
                print(f"[validate] L{i}: id ausente o no str", file=sys.stderr)
                errors += 1
                continue
            if _id in ids:
                print(f"[validate] L{i}: id duplicado {_id}", file=sys.stderr)
                errors += 1
            ids.add(_id)
            # campos/tipos
            errs = validate_doc(d, args.check_text_path)
            if errs:
                print(f"[validate] L{i}: " + "; ".join(errs), file=sys.stderr)
                errors += len(errs)

    if errors:
        print(
            f"[validate] ERRORES: {errors} en {total} docs. Corrige antes de bundle.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[validate] OK: {total} docs validados, ids únicos={len(ids)}")

    # Partir si procede
    work_dir = solr_dir / "bundle_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    parts = split_jsonl(src, work_dir, args.max_per_file)
    if len(parts) == 1 and parts[0] == src:
        # copiar docs.jsonl a work_dir para empaquetar
        dst = work_dir / "docs.jsonl"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        parts = [dst]

    # schema contract & readme
    (work_dir / "schema_contract.json").write_text(
        json.dumps(SCHEMA_CONTRACT, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (work_dir / "README_SOLR.md").write_text(
        README_SOLR.replace("<RI>", ri), encoding="utf-8"
    )

    # checksums
    with (work_dir / "CHECKSUMS.sha256").open("w", encoding="utf-8") as chk:
        for p in sorted(work_dir.glob("*.jsonl")):
            chk.write(f"{sha256_file(p)}  {p.name}\n")
        chk.write(
            f"{sha256_file(work_dir/'schema_contract.json')}  schema_contract.json\n"
        )
        chk.write(f"{sha256_file(work_dir/'README_SOLR.md')}  README_SOLR.md\n")

    # zip bundle
    date = datetime.utcnow().strftime("%Y%m%d")
    bundle_name = args.bundle_name or f"solr_bundle_{ri}_{date}.zip"
    bundle_path = solr_dir / bundle_name
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work_dir.iterdir()):
            z.write(p, arcname=p.name)

    print(
        f"[bundle] Listo: {bundle_path}  (archivos: {[p.name for p in sorted(work_dir.iterdir())]})"
    )


if __name__ == "__main__":
    main()
