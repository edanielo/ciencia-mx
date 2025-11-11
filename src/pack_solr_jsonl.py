#!/usr/bin/env python3
"""
pack_solr_jsonl.py — Empaqueta enriched.jsonl a JSONL “Solr-ready”.
Añade (opcionalmente) Dublin Core desde corpus/meta/<RI>/dc_index.jsonl,
usando el mapeo sha256 → xml_path presente en logs/fetch.log.
Salida: out/solr/<RI>/docs.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CLEAN = Path("corpus/clean")
OUT = Path("out/solr")
LOGS = Path("logs")
META = Path("corpus/meta")
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


def load_fetch_map(ri: str) -> dict[str, str]:
    """
    Devuelve sha256 -> xml_path (fuente OAI) a partir de logs/fetch.log (status=ok).
    """
    fmap: dict[str, str] = {}
    p = LOGS / "fetch.log"
    if not p.exists():
        return fmap
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ri") != ri or rec.get("status") != "ok":
            continue
        sha = rec.get("sha256")
        src = rec.get("source")  # ruta al XML OAI guardada por resolve_fetch.py
        if isinstance(sha, str) and isinstance(src, str) and sha not in fmap:
            fmap[sha] = src
    return fmap


def load_dc_index(ri: str) -> dict[str, dict]:
    """
    Devuelve xml_path -> dc_dict desde corpus/meta/<RI>/dc_index.jsonl
    """
    idx: dict[str, dict] = {}
    p = META / ri / "dc_index.jsonl"
    if not p.exists():
        return idx
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        xmlp = rec.get("xml_path")
        dc = rec.get("dc")
        if isinstance(xmlp, str) and isinstance(dc, dict):
            idx[xmlp] = dc
    return idx


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
    # Carga mapeos para DC
    fetch_map = load_fetch_map(args.ri)  # sha256 -> xml_path
    dc_index = load_dc_index(args.ri)  # xml_path -> dc dict

    # --- Índices auxiliares ---
    # 1) enriched.jsonl por sha
    enriched_by_sha: dict[str, dict] = {}
    for _r in load_enriched(args.ri):
        _sha = _r.get("sha256")
        if isinstance(_sha, str) and _sha:
            enriched_by_sha[_sha] = _r
    # 2) invertir fetch_map → xml_path -> sha
    xml_to_sha: dict[str, str] = {}
    for _sha, _xml in fetch_map.items():
        if isinstance(_xml, str) and _xml and _xml not in xml_to_sha:
            xml_to_sha[_xml] = _sha

    count = 0
    with outpath.open("w", encoding="utf-8") as out:
        for xml_path, dc in dc_index.items():
            sha = xml_to_sha.get(xml_path)
            # --- Construcción base del doc ---
            if sha and sha in enriched_by_sha:
                # Doc CON bitstream/texto (mantiene tus campos actuales)
                r = enriched_by_sha[sha]
                text = ""
                if args.include_text:
                    tpath = r.get("text_path")
                    if tpath and Path(tpath).exists():
                        text = clamp(
                            Path(tpath).read_text(encoding="utf-8", errors="ignore")
                        )
                doc = {
                    "id": sha,
                    "ri_s": r.get("ri") or args.ri,
                    "sha256_s": sha,
                    "ocr_b": bool(r.get("ocr_applied")),
                    "tokens_i": int(r.get("metrics", {}).get("tokens", 0)),
                    "alpha_ratio_f": float(
                        r.get("metrics", {}).get("alpha_ratio", 0.0)
                    ),
                    "score_f": float(
                        r.get("metrics", {}).get("text_quality_score", 0.0)
                    ),
                    "persons_ss": r.get("persons") or [],
                    "orgs_ss": r.get("orgs") or [],
                    "places_ss": r.get("places") or [],
                    "keyphrases_ss": r.get("keyphrases") or [],
                    "text_path_s": r.get("text_path"),
                    "has_bitstream_b": True,
                }
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
            else:
                # Doc SOLO metadatos (sin bitstream/texto)
                doc = {
                    "id": xml_path,  # identificador estable para este registro DC
                    "ri_s": args.ri,
                    "has_bitstream_b": False,
                }

            # --- Inyección Dublin Core (usando 'dc' ya cargado para este xml_path) ---
            titles = [
                t for t in (dc.get("titles") or []) if isinstance(t, str) and t.strip()
            ]
            if titles:
                doc.setdefault("title_s", titles[0])
                doc.setdefault("title_t", titles[0])
            creators = [
                c
                for c in (dc.get("creators") or [])
                if isinstance(c, str) and c.strip()
            ]
            if creators:
                doc["authors_ss"] = creators
                if "persons_ss" not in doc:
                    doc["persons_ss"] = creators
            subjects = [
                s
                for s in (dc.get("subjects") or [])
                if isinstance(s, str) and s.strip()
            ]
            if subjects:
                doc["subjects_ss"] = subjects
            desc = dc.get("description")
            if isinstance(desc, str) and desc.strip():
                doc.setdefault("abstract_t", desc)
            types = [
                t for t in (dc.get("types") or []) if isinstance(t, str) and t.strip()
            ]
            if types:
                doc["type_ss"] = types
            langs = [
                lang
                for lang in (dc.get("languages") or [])
                if isinstance(lang, str) and lang.strip()
            ]
            if langs:
                doc.setdefault("lang_s", langs[0])
            pubs = [
                p
                for p in (dc.get("publishers") or [])
                if isinstance(p, str) and p.strip()
            ]
            if pubs:
                doc["publisher_ss"] = pubs
            if "url_s" not in doc:
                url_dc = dc.get("url")
                if isinstance(url_dc, str) and url_dc.strip():
                    doc["url_s"] = url_dc
            if "doi_s" not in doc:
                doi_dc = dc.get("doi")
                if isinstance(doi_dc, str) and doi_dc.strip():
                    doc["doi_s"] = doi_dc
            if "year_i" not in doc:
                y = dc.get("year_guess")
                if isinstance(y, int):
                    doc["year_i"] = y

            # --- Saneamiento y normalización (igual que ya hacías) ---
            if "year_i" in doc:
                try:
                    doc["year_i"] = int(doc["year_i"])
                except (TypeError, ValueError):
                    doc.pop("year_i", None)
            optional_keys = {
                "url_s",
                "year_i",
                "doi_s",
                "persons_ss",
                "orgs_ss",
                "places_ss",
                "keyphrases_ss",
                "text_t",
                "title_t",
            }
            for _k in list(doc.keys()):
                if _k not in optional_keys:
                    continue
                _v = doc[_k]
                if (
                    _v is None
                    or (isinstance(_v, str) and not _v.strip())
                    or (isinstance(_v, list) and len(_v) == 0)
                ):
                    doc.pop(_k, None)
            if "keyphrases_ss" in doc and isinstance(doc["keyphrases_ss"], list):
                doc["keyphrases_ss"] = [
                    s for s in {str(x).strip() for x in doc["keyphrases_ss"]} if s
                ]

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"[pack] Listo. Docs empaquetados: {count}. Salida: {outpath}")


if __name__ == "__main__":
    main()
