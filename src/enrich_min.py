#!/usr/bin/env python3
"""
enrich_min.py — Enriquecimiento mínimo: YAKE + spaCy (es) + heurísticas.
Entrada: corpus/raw/<RI>/manifest.jsonl + <sha>.txt
Salida: corpus/clean/<RI>/enriched.jsonl
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import spacy
from yake import KeywordExtractor

CORPUS_RAW = Path("corpus/raw")
CORPUS_CLEAN = Path("corpus/clean")

DOI_RX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")
EMAIL_RX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RX = re.compile(r"https?://\S+")


def load_manifest(ri: str) -> List[Dict[str, Any]]:
    m = CORPUS_RAW / ri / "manifest.jsonl"
    if not m.exists():
        raise SystemExit(f"ERROR: no existe {m}")
    rows = []
    for line in m.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def extract_entities(nlp, text: str) -> Dict[str, List[str]]:
    doc = nlp(text[:200_000])  # límite de seguridad
    ents = {"persons": set(), "orgs": set(), "places": set()}
    for ent in doc.ents:
        if ent.label_ in ("PER", "PERSON"):
            ents["persons"].add(ent.text)
        elif ent.label_ in ("ORG",):
            ents["orgs"].add(ent.text)
        elif ent.label_ in ("LOC", "GPE"):
            ents["places"].add(ent.text)
    return {k: sorted(v) for k, v in ents.items()}


def extract_yake(text: str, topk: int = 12) -> List[str]:
    kw = KeywordExtractor(lan="es", n=1, top=topk)  # n=1..3 si quieres multi-gramas
    pairs = kw.extract_keywords(text[:400_000])
    # pairs: [(keyphrase, score)] -> ordenar por score asc
    phrases = [p[0] for p in sorted(pairs, key=lambda x: x[1])]
    # limpieza rápida
    uniq: List[str] = []
    seen: Set[str] = set()
    for p in phrases:
        t = p.strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        uniq.append(t)
    return uniq


def heuristics(text: str) -> Dict[str, Any]:
    dois = DOI_RX.findall(text)
    years = [
        int(y) for y in YEAR_RX.findall(text.replace(" ", ""))
    ]  # cuidado con falsos positivos
    years = [int(m.group(0)) for m in re.finditer(YEAR_RX, text)]
    emails = EMAIL_RX.findall(text)
    urls = URL_RX.findall(text)
    year_guess = None
    if years:
        # elige año “razonable” (moda o mínimo > 1950)
        cand = [y for y in years if y >= 1950]
        year_guess = min(cand) if cand else min(years)
    doi_guess = dois[0] if dois else None
    return {
        "year_guess": year_guess,
        "doi_guess": doi_guess,
        "emails": sorted(set(emails))[:5],
        "urls": sorted(set(urls))[:10],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Enriquecimiento mínimo del corpus (YAKE + spaCy + heurísticas)"
    )
    ap.add_argument("--ri", required=True)
    ap.add_argument("--spacy-model", default="es_core_news_md")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--yake-top", type=int, default=12)
    args = ap.parse_args()

    rows = load_manifest(args.ri)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    outdir = CORPUS_CLEAN / args.ri
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "enriched.jsonl"

    try:
        nlp = spacy.load(args.spacy_model)
    except Exception as e:
        raise SystemExit(
            f"ERROR: carga spaCy '{args.spacy_model}'. Ejecuta: python -m spacy download {args.spacy_model}\n{e}"
        )

    processed = 0
    with outpath.open("w", encoding="utf-8") as out:
        for r in rows:
            sha = r.get("sha256")
            txt_path = CORPUS_RAW / args.ri / f"{sha}.txt"
            if not txt_path.exists():
                continue
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            ents = extract_entities(nlp, text)
            keys = extract_yake(text, topk=args.yake_top)
            heur = heuristics(text)
            doc = {
                "ri": args.ri,
                "sha256": sha,
                "source_url": r.get("source_url"),
                "ocr_applied": r.get("ocr_applied"),
                "metrics": r.get("metrics"),
                "persons": ents["persons"],
                "orgs": ents["orgs"],
                "places": ents["places"],
                "keyphrases": keys,
                **heur,
                "text_path": str(txt_path),
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed += 1
            if processed % 20 == 0:
                print(f"[enrich] {processed} docs → {outpath}")

    print(f"[enrich] Terminado. Docs procesados: {processed}. Salida: {outpath}")


if __name__ == "__main__":
    main()
