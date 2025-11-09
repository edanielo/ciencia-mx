#!/usr/bin/env python3
"""
enrich_min.py — Enriquecimiento mínimo: YAKE + spaCy (es/en) + heurísticas.
Entrada: corpus/raw/<RI>/manifest.jsonl + <sha>.txt
Salida: corpus/clean/<RI>/enriched.jsonl
Mejoras:
  - Detección heurística de idioma (es/en) → spaCy/YAKE por idioma.
  - YAKE n=1..3 + filtro de stopwords → menos ruido.
  - Post-filtros NER (longitud, casing) y top-N por tipo.
"""

from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

import spacy
from yake import KeywordExtractor

CORPUS_RAW = Path("corpus/raw")
CORPUS_CLEAN = Path("corpus/clean")

DOI_RX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")
EMAIL_RX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RX = re.compile(r"https?://\S+")

# --- Stopwords cortas para heurística/filtrado rápido (evitamos dependencia extra) ---
ES_STOP = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "más",
    "o",
    "pero",
    "sus",
    "le",
    "ya",
    "o",
    "u",
    "e",
    "es",
    "son",
    "ser",
    "fue",
}
EN_STOP = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "a",
    "for",
    "is",
    "on",
    "that",
    "with",
    "as",
    "by",
    "an",
    "from",
    "this",
    "at",
    "or",
    "be",
    "are",
    "it",
    "was",
    "were",
    "has",
    "have",
    "had",
}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[^\x09\x0A\x20-\x7E\u0080-\uFFFF]", "", s)
    return s


def detect_lang(text: str) -> str:
    """Heurística simple es/en: cuenta stopwords y presencia de acentos."""
    t = text.lower()
    # ventanas pequeñas para eficiencia
    head = t[:20000]
    es_hits = sum(1 for w in ES_STOP if f" {w} " in f" {head} ")
    en_hits = sum(1 for w in EN_STOP if f" {w} " in f" {head} ")
    if any(ch in head for ch in "áéíóúñü"):
        es_hits += 2
    return "es" if es_hits >= en_hits else "en"


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


def _clean_ent(txt: str) -> Optional[str]:
    t = txt.strip()
    # descarta demasiado largo o muy corto
    if len(t) < 2 or len(t) > 80:
        return None
    # si tiene más de 1 palabra y todas minúsculas, suele ser ruido
    if " " in t and t == t.lower():
        return None
    # evita que inicie con símbolos raros
    if not re.match(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", t):
        return None
    return t


def extract_entities(
    nlp, lang: str, text: str, top_per_type: int = 20
) -> Dict[str, List[str]]:
    doc = nlp(text[:200_000])
    persons: List[str] = []
    orgs: List[str] = []
    places: List[str] = []
    for ent in doc.ents:
        lbl = ent.label_
        keep = _clean_ent(ent.text)
        if not keep:
            continue
        if lbl in ("PER", "PERSON"):
            persons.append(keep)
        elif lbl in ("ORG",):
            orgs.append(keep)
        elif lbl in ("LOC", "GPE"):
            places.append(keep)

    # únicos y recorta a top N (preserva orden)
    def uniq_top(xs: List[str], n: int) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
            if len(out) >= n:
                break
        return out

    return {
        "persons": uniq_top(persons, top_per_type),
        "orgs": uniq_top(orgs, top_per_type),
        "places": uniq_top(places, top_per_type),
    }


def extract_yake(text: str, lang: str, topk: int = 12) -> List[str]:
    """Extrae n-gramas 1..3 con YAKE, filtra stopwords y dupes, prioriza mejor score."""
    text = text[:400_000]
    lan = "es" if lang == "es" else "en"
    candidates: List[str] = []
    for n in (1, 2, 3):
        pairs = KeywordExtractor(lan=lan, n=n, top=topk).extract_keywords(text)
        for phrase, score in pairs:
            candidates.append((phrase.strip(), score))
    # ordena por score (menor es mejor) y filtra
    candidates.sort(key=lambda x: x[1])
    stop = ES_STOP if lang == "es" else EN_STOP
    out: List[str] = []
    seen: Set[str] = set()
    for ph, _ in candidates:
        if not ph or len(ph) > 80:
            continue
        # tokens, quita ruido extremo
        toks = ph.split()
        if all(t.lower() in stop for t in toks):
            continue
        k = ph.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(ph)
        if len(out) >= topk:  # top final
            break
    return out


def heuristics(text: str) -> Dict[str, Any]:
    dois = DOI_RX.findall(text)
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
    ap.add_argument(
        "--spacy-model-es", default="es_core_news_md", help="Modelo spaCy para español"
    )
    ap.add_argument(
        "--spacy-model-en", default="en_core_web_sm", help="Modelo spaCy para inglés"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--yake-top", type=int, default=12, help="Top final (tras fusionar n=1..3)"
    )
    ap.add_argument("--ents-top", type=int, default=20, help="Top por tipo de entidad")
    args = ap.parse_args()

    rows = load_manifest(args.ri)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    outdir = CORPUS_CLEAN / args.ri
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "enriched.jsonl"

    # Carga perezosa por idioma
    nlp_cache: Dict[str, Any] = {}

    def get_nlp(lang: str):
        if lang in nlp_cache:
            return nlp_cache[lang]
        model = args.spacy_model_es if lang == "es" else args.spacy_model_en
        try:
            nlp_cache[lang] = spacy.load(model)
        except Exception:
            # Fallback: pipeline vacía que no rompe el flujo
            nlp_cache[lang] = spacy.blank("es" if lang == "es" else "en")
        return nlp_cache[lang]

    processed = 0
    with outpath.open("w", encoding="utf-8") as out:
        for r in rows:
            sha = r.get("sha256")
            txt_path = CORPUS_RAW / args.ri / f"{sha}.txt"
            if not txt_path.exists():
                continue
            raw = txt_path.read_text(encoding="utf-8", errors="ignore")
            text = normalize_text(raw)
            lang = detect_lang(text)
            nlp = get_nlp(lang)
            ents = extract_entities(nlp, lang, text, top_per_type=args.ents_top)
            keys = extract_yake(text, lang=lang, topk=args.yake_top)
            heur = heuristics(text)
            doc = {
                "ri": args.ri,
                "sha256": sha,
                "lang": lang,
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
