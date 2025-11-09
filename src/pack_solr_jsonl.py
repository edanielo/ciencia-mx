#!/usr/bin/env python3
"""
pack_solr_jsonl.py — Empaqueta enriched.jsonl a JSONL “Solr-ready”.
Salida: out/solr/<RI>/docs.jsonl
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

CLEAN = Path("corpus/clean")
OUT = Path("out/solr")
RAW = Path("corpus/raw")


def load_enriched(ri: str):
    p = CLEAN / ri / "enriched.jsonl"
    if not p.exists():
        raise SystemExit(f"ERROR: no existe {p}. Ejecuta enrich_min.py primero.")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def clamp(s: str, n: int = 20000) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "\n...[TRUNCATED]..."


def main():
    ap = argparse.ArgumentParser(description="Empaquetado JSONL para Solr")
    ap.add_argument("--ri", required=True)
    ap.add_argument(
        "--include-text",
        action="store_true",
        help="Incluye texto (truncado) como field text_t",
    )
    args = ap.parse_args()

    outdir = OUT / args.ri
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "docs.jsonl"

    count = 0
    with outpath.open("w", encoding="utf-8") as out:
        for r in load_enriched(args.ri):
            sha = r.get("sha256")
            text = ""
            if args.include_text:
                tpath = r.get("text_path")
                if tpath and Path(tpath).exists():
                    text = clamp(
                        Path(tpath).read_text(encoding="utf-8", errors="ignore")
                    )

            # Campos requeridos
            doc = {
                "id": sha,
                "ri_s": r.get("ri"),
                "sha256_s": sha,
                "ocr_b": bool(r.get("ocr_applied")),
                "tokens_i": int(r.get("metrics", {}).get("tokens", 0)),
                "alpha_ratio_f": float(r.get("metrics", {}).get("alpha_ratio", 0.0)),
                "score_f": float(r.get("metrics", {}).get("text_quality_score", 0.0)),
                "persons_ss": r.get("persons") or [],
                "orgs_ss": r.get("orgs") or [],
                "places_ss": r.get("places") or [],
                "keyphrases_ss": r.get("keyphrases") or [],
                "text_path_s": r.get("text_path"),
            }

            # Opcionales: añade solo si traen valor válido
            url = r.get("source_url")
            if isinstance(url, str) and url.strip():
                doc["url_s"] = url
            year = r.get("year_guess")
            if isinstance(year, int):
                doc["year_i"] = year
            doi = r.get("doi_guess")
            if isinstance(doi, str) and doi.strip():
                doc["doi_s"] = doi
            if args.include_text and text:
                doc["text_t"] = text

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"[pack] Listo. Docs empaquetados: {count}. Salida: {outpath}")


if __name__ == "__main__":
    main()
