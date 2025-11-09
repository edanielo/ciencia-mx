#!/usr/bin/env python3
"""
extract_dc_index.py — Extrae Dublin Core desde raw/oai/<RI>/*.xml y genera corpus/meta/<RI>/dc_index.jsonl

Campos extraídos (por registro OAI):
  - title[s], creators, subjects, description, types, languages, publisher, identifiers
  - url (primer http/https en identifiers), doi (regex), year_guess (regex en dc:date)
Salida:
  corpus/meta/<RI>/dc_index.jsonl  (líneas: {"xml_path": "...", "dc": {...}})
"""
from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
from lxml import etree

RAW_OAI = Path("raw/oai")
META = Path("corpus/meta")

DOI_RX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")

LANG_MAP = {
    "spa": "es",
    "es": "es",
    "esp": "es",
    "eng": "en",
    "en": "en",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return " ".join(s.split())


def texts(root: etree._Element, tag: str, ns: Dict[str, str]) -> List[str]:
    vals: List[str] = []
    for el in root.xpath(f".//dc:{tag}", namespaces=ns):
        t = norm("".join(el.itertext()))
        if t:
            vals.append(t)
    return vals


def first_http(identifiers: List[str]) -> Optional[str]:
    for x in identifiers:
        t = x.strip()
        if t.startswith(("http://", "https://")):
            return t
    return None


def first_doi(identifiers: List[str]) -> Optional[str]:
    for x in identifiers:
        m = DOI_RX.search(x)
        if m:
            return m.group(0)
    return None


def guess_year(dates: List[str]) -> Optional[int]:
    for d in dates:
        m = YEAR_RX.search(d)
        if m:
            y = int(m.group(0))
            if 1800 <= y <= 2100:
                return y
    return None


def main():
    ap = argparse.ArgumentParser(description="Extrae Dublin Core a dc_index.jsonl")
    ap.add_argument("--ri", required=True)
    args = ap.parse_args()

    ri = args.ri
    in_dir = RAW_OAI / ri
    if not in_dir.exists():
        raise SystemExit(f"ERROR: no existe {in_dir}")

    out_dir = META / ri
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / "dc_index.jsonl"

    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    total = 0
    ok = 0
    with outpath.open("w", encoding="utf-8") as out:
        for xml_path in sorted(in_dir.glob("*.xml")):
            total += 1
            try:
                root = etree.parse(str(xml_path)).getroot()
            except Exception:
                continue
            titles = texts(root, "title", ns)
            creators = texts(root, "creator", ns)
            subjects = texts(root, "subject", ns)
            descriptions = texts(root, "description", ns)
            types = texts(root, "type", ns)
            languages = [
                LANG_MAP.get(x.lower(), x.lower()) for x in texts(root, "language", ns)
            ]
            publishers = texts(root, "publisher", ns)
            identifiers = texts(root, "identifier", ns)
            dates = texts(root, "date", ns)

            url = first_http(identifiers)
            doi = first_doi(identifiers)
            year = guess_year(dates)
            description = descriptions[0] if descriptions else ""

            rec = {
                "xml_path": str(xml_path),
                "dc": {
                    "titles": titles,
                    "creators": creators,
                    "subjects": subjects,
                    "description": description,
                    "types": types,
                    "languages": languages,
                    "publishers": publishers,
                    "identifiers": identifiers,
                    "url": url,
                    "doi": doi,
                    "year_guess": year,
                },
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
    print(f"[dc] {ri}: procesados={total}, indexados={ok}, salida={outpath}")


if __name__ == "__main__":
    main()
