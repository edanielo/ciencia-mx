#!/usr/bin/env python3
"""
build_nlp_corpus.py — Construye un corpus PLN masivo desde corpus/raw/<RI>.

Salidas:
  - nlp/<RI>/all.txt      (texto plano; docs concatenados con separadores)
  - nlp/<RI>/all.jsonl    (JSONL: 1 doc/línea con metadatos + texto)

Características:
  - Lee corpus/raw/<RI>/manifest.jsonl y cada <sha>.txt
  - Normaliza (UTF-8 NFC), limpia controles, colapsa saltos
  - Dedup por hash del texto normalizado
  - Filtro por tokens mínimos (--min-tokens) y opcional --lower
"""

from __future__ import annotations
import argparse
import json
import hashlib
import re
import unicodedata
from pathlib import Path

CORPUS_RAW = Path("corpus/raw")
NLP = Path("nlp")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(
        r"[^\x09\x0A\x20-\x7E\u0080-\uFFFF]", "", s
    )  # quita controles salvo \t y \n
    s = re.sub(r"\n{3,}", "\n\n", s)  # no más de 2 saltos seguidos
    s = re.sub(r"[ \t]{3,}", "  ", s)
    return s.strip()


def token_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))


def load_manifest(ri: str):
    mpath = CORPUS_RAW / ri / "manifest.jsonl"
    if not mpath.exists():
        raise SystemExit(f"ERROR: no existe {mpath}")
    with mpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser(
        description="Construye corpus PLN masivo (txt + jsonl)"
    )
    ap.add_argument("--ri", required=True)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--lower", action="store_true", help="Convierte a minúsculas")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = todos")
    args = ap.parse_args()

    outdir = NLP / args.ri
    outdir.mkdir(parents=True, exist_ok=True)
    txt_path = outdir / "all.txt"
    jl_path = outdir / "all.jsonl"

    seen_hashes = set()
    added = 0

    with (
        txt_path.open("w", encoding="utf-8") as ft,
        jl_path.open("w", encoding="utf-8") as fj,
    ):
        for rec in load_manifest(args.ri):
            sha = rec.get("sha256")
            tpath = CORPUS_RAW / args.ri / f"{sha}.txt"
            if not tpath.exists():
                continue
            raw = tpath.read_text(encoding="utf-8", errors="ignore")
            text = normalize_text(raw)
            if args.lower:
                text = text.lower()
            if token_count(text) < args.min_tokens:
                continue

            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            meta = {
                "sha256": sha,
                "ri": args.ri,
                "source_url": rec.get("source_url"),
                "year_guess": rec.get("metrics", {}).get("year_i")
                or rec.get("year_guess"),
                "tokens": rec.get("metrics", {}).get("tokens"),
                "alpha_ratio": rec.get("metrics", {}).get("alpha_ratio"),
                "ocr_applied": rec.get("ocr_applied"),
                "text_hash": h,
            }
            fj.write(
                json.dumps({"meta": meta, "text": text}, ensure_ascii=False) + "\n"
            )

            ft.write(f"\n\n<<<DOC {sha}>>>\n\n")
            ft.write(text)

            added += 1
            if args.max_docs and added >= args.max_docs:
                break

    print(
        f"[nlp] Listo. Documentos añadidos: {added}. Salidas:\n  - {txt_path}\n  - {jl_path}"
    )


if __name__ == "__main__":
    main()
