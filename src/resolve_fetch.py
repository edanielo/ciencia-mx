#!/usr/bin/env python3
"""
resolve_fetch.py — Resuelve dc:identifier → descarga binarios al spool (efímero)
Características:
  - Extrae URLs (http/https, DOI, handle) desde los XML OAI (oai_dc).
  - HEAD → valida MIME permitido; si no, GET con "sniff" de PDF.
  - Si es página HTML (landing), raspa primer enlace PDF/bitstream (DSpace/OJS).
  - Nombra archivos con extensión correcta al final (maneja querystring y Content-Disposition).
  - Deduplicación intra-lote por sha256 (no repite descargas idénticas).
  - Log JSONL en logs/fetch.log con status: ok | duplicate | skipped | no_url | no_download | error.

Uso:
  conda activate cienciamx
  python src/resolve_fetch.py --ri UADY --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from lxml import etree, html
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Rutas del proyecto ---
RAW_OAI = Path("raw/oai")
SPOOL = Path("spool/bin")
LOGS = Path("logs")

# --- Mapeo simple de Content-Type a extension ---
EXT_BY_CTYPE = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/plain": ".txt",
}


# ---------------------- Utilidades de configuración y logging ----------------------
def load_origin(ri: str) -> Dict[str, Any]:
    cfg_path = Path("configs/origins.yaml")
    if not cfg_path.exists():
        raise SystemExit("ERROR: no existe configs/origins.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for repo in cfg.get("repos", []):
        if repo.get("name", "").lower() == ri.lower():
            return repo
    raise SystemExit(f"RI no encontrado en configs/origins.yaml: {ri}")


def open_log() -> io.TextIOWrapper:
    LOGS.mkdir(parents=True, exist_ok=True)
    return open(LOGS / "fetch.log", "a", encoding="utf-8")


def log(flog: io.TextIOWrapper, **rec) -> None:
    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------- Extracción de URLs desde XML oai_dc ----------------------
def normalize_identifier(t: str) -> str:
    t = t.strip()
    if t.lower().startswith("doi:"):
        # convierte "doi:10.xxxx/yyy" a https://doi.org/10.xxxx/yyy
        return "https://doi.org/" + t.split(":", 1)[1].strip()
    return t


def find_urls_from_xml(xml_bytes: bytes) -> List[str]:
    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    }
    root = etree.fromstring(xml_bytes)
    urls: List[str] = []
    for el in root.xpath(".//dc:identifier", namespaces=ns):
        t = (el.text or "").strip()
        if not t:
            continue
        if (
            t.startswith(("http://", "https://"))
            or "doi.org" in t.lower()
            or t.lower().startswith("doi:")
        ):
            urls.append(normalize_identifier(t))

    # Prioridad: DOI > handle > .pdf > resto
    def priority(u: str) -> Tuple[bool, bool, bool]:
        low = u.lower()
        return (
            "doi.org" not in low,
            ("handle" not in low and "hdl.handle.net" not in low),
            not low.endswith(".pdf"),
        )

    # Quitar duplicados preservando orden
    seen = set()
    ordered: List[str] = []
    for u in sorted(urls, key=priority):
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


# ---------------------- Heurísticas de nombre de archivo ----------------------
def filename_from_cd(cd_header: Optional[str]) -> Optional[str]:
    # Content-Disposition: attachment; filename="algo.pdf"
    if not cd_header:
        return None
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd_header, flags=re.I)
    return m.group(1) if m else None


def sanitize_name(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._")
    if len(name) > max_len:
        root, ext = os.path.splitext(name)
        name = root[: max_len - len(ext) - 1] + "_" + ext
    return name or "download"


def build_spool_name(url: str, content_type: str, headers: Dict[str, str]) -> str:
    """
    Devuelve un nombre con la extensión correcta al final.
    Si hay querystring, se pasa a sufijo antes de la extensión:
      .../document.pdf?sequence=1&isAllowed=y  ->  document_sequence_1_isAllowed_y.pdf
    Considera Content-Disposition y Content-Type.
    """
    p = urllib.parse.urlparse(url)
    cd_name = filename_from_cd(
        headers.get("Content-Disposition") or headers.get("content-disposition")
    )
    path_name = Path(p.path).name or "download"
    base = cd_name or path_name

    stem, ext_from_path = os.path.splitext(base)
    q_suffix = ""
    if p.query:
        q = re.sub(r"[^A-Za-z0-9]+", "_", p.query).strip("_")[:60]
        if q:
            q_suffix = f"_{q}"

    ctype_norm = (content_type or "").split(";")[0].strip().lower()
    ext = EXT_BY_CTYPE.get(ctype_norm, ext_from_path or ".bin")

    final = sanitize_name(f"{stem}{q_suffix}{ext}")
    return final


def unique_path(path: Path) -> Path:
    """Evita sobrescribir si ya existe; agrega _1, _2..."""
    if not path.exists():
        return path
    stem = path.stem
    ext = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1


# ---------------------- Red de requests y detección de tipos ----------------------
def is_allowed_mime(ctype: str, url: str, allowed: set[str]) -> bool:
    c = (ctype or "").split(";")[0].strip().lower()
    if any(c.startswith(m) for m in allowed):
        return True
    # Fallback por extensión de URL
    low = url.lower().rsplit("?", 1)[0].rsplit("#", 1)[0]
    if low.endswith(".pdf") and any(a.startswith("application/pdf") for a in allowed):
        return True
    return False


def head(session: requests.Session, url: str) -> requests.Response:
    return session.head(url, allow_redirects=True, timeout=30)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_stream(session: requests.Session, url: str) -> requests.Response:
    r = session.get(url, stream=True, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r


def sniff_is_pdf(resp: requests.Response, max_peek: int = 8192) -> Tuple[bool, bytes]:
    """Lee algunos KB y detecta firma PDF (%PDF-)."""
    blob = b""
    try:
        for chunk in resp.iter_content(8192):
            if not chunk:
                break
            blob += chunk
            if len(blob) >= max_peek:
                break
    except Exception:
        pass
    return (blob.startswith(b"%PDF-"), blob)


def scrape_pdf_link(session: requests.Session, url: str) -> Optional[str]:
    """
    Si llegamos a una landing HTML, intenta encontrar el primer enlace PDF/bitstream.
    Útil para DSpace/OJS (links con .pdf o /bitstream/ en href).
    """
    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        doc = html.fromstring(r.content)
        hrefs = [h.strip() for h in doc.xpath("//a/@href") if h and h.strip()]
        cands: List[str] = []
        for h in hrefs:
            low = h.lower()
            if ".pdf" in low or "/bitstream/" in low:
                cands.append(h)
        if not cands:
            return None
        from urllib.parse import urljoin

        for c in cands:
            cand = urljoin(r.url, c)
            low = cand.lower()
            if low.endswith(".pdf") or "/bitstream/" in low:
                return cand
    except Exception:
        return None
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def download_stream(
    session: requests.Session, url: str, max_bytes: int
) -> Dict[str, Any]:
    """
    Descarga con GET y guarda en spool/bin con nombre consistente.
    Devuelve dict con path, bytes, sha256 y content_type.
    """
    resp = session.get(url, stream=True, allow_redirects=True, timeout=60)
    resp.raise_for_status()

    SPOOL.mkdir(parents=True, exist_ok=True)
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    name = build_spool_name(resp.url, ctype, resp.headers)
    path = unique_path(SPOOL / name)

    h = hashlib.sha256()
    total = 0
    with path.open("wb") as f:
        for chunk in resp.iter_content(8192):
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                resp.close()
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Excede max_bytes: {total} > {max_bytes}")
            h.update(chunk)
            f.write(chunk)

    return {
        "path": str(path),
        "bytes": total,
        "sha256": h.hexdigest(),
        "content_type": ctype,
    }


# ----------------------------------- Main -----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resuelve dc:identifier y descarga binarios al spool (con sniff/fallback)"
    )
    ap.add_argument(
        "--ri", required=True, help="Nombre del RI en configs/origins.yaml (e.g., UADY)"
    )
    ap.add_argument(
        "--limit", type=int, default=20, help="Máximo de descargas válidas a intentar"
    )
    args = ap.parse_args()

    origin = load_origin(args.ri)
    allowed = set(origin.get("allowed_mime", ["application/pdf"]))
    max_bytes = int(origin.get("max_bytes_pdf", 30_000_000))

    session = requests.Session()
    session.headers["User-Agent"] = "CienciaMX Fetch/1.1"
    flog = open_log()

    xml_dir = RAW_OAI / origin["name"]
    files = sorted(xml_dir.glob("*.xml"))

    if not files:
        print(
            f"[fetch] No hay XML en {xml_dir}. Ejecuta primero harvest_oai.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    picked = 0
    seen_hashes: set[str] = set()

    for x in files:
        if picked >= args.limit:
            break

        xml = x.read_bytes()
        urls = find_urls_from_xml(xml)

        if not urls:
            log(flog, ri=origin["name"], source=str(x), status="no_url")
            continue

        success = False

        for url in urls:
            try:
                # 1) HEAD: ¿Content-Type permitido?
                try:
                    h = head(session, url)
                    ctype = (h.headers.get("Content-Type") or "").split(";")[0].strip()
                except Exception:
                    ctype = ""

                if is_allowed_mime(ctype, url, allowed):
                    res = download_stream(session, url, max_bytes)
                    if res["sha256"] in seen_hashes:
                        Path(res["path"]).unlink(missing_ok=True)
                        log(
                            flog,
                            ri=origin["name"],
                            source=str(x),
                            url=url,
                            sha256=res["sha256"],
                            status="duplicate",
                        )
                    else:
                        seen_hashes.add(res["sha256"])
                        log(
                            flog,
                            ri=origin["name"],
                            source=str(x),
                            url=url,
                            **res,
                            status="ok",
                        )
                        picked += 1
                        success = True
                        print(
                            f"[fetch] {picked:04d} {res['content_type'] or '?'} {res['bytes']}B → {res['path']}"
                        )
                    break

                # 2) GET + sniff PDF
                resp = get_stream(session, url)
                sniff_pdf, _ = sniff_is_pdf(resp)
                if sniff_pdf and any(a.startswith("application/pdf") for a in allowed):
                    resp.close()
                    res = download_stream(session, url, max_bytes)
                    if res["sha256"] in seen_hashes:
                        Path(res["path"]).unlink(missing_ok=True)
                        log(
                            flog,
                            ri=origin["name"],
                            source=str(x),
                            url=url,
                            sha256=res["sha256"],
                            status="duplicate",
                        )
                    else:
                        seen_hashes.add(res["sha256"])
                        log(
                            flog,
                            ri=origin["name"],
                            source=str(x),
                            url=url,
                            **res,
                            status="ok",
                        )
                        picked += 1
                        success = True
                        print(
                            f"[fetch] {picked:04d} {res['content_type'] or 'application/pdf'} {res['bytes']}B → {res['path']}"
                        )
                    break

                # 3) Si es HTML (landing), intenta raspar un PDF/bitstream
                is_html = "text/html" in (ctype or "") or (
                    resp.headers.get("Content-Type", "").startswith("text/html")
                )
                resp.close()
                if is_html:
                    pdf_link = scrape_pdf_link(session, url)
                    if pdf_link:
                        # valida por HEAD
                        try:
                            h2 = head(session, pdf_link)
                            ctype2 = (
                                (h2.headers.get("Content-Type") or "")
                                .split(";")[0]
                                .strip()
                            )
                        except Exception:
                            ctype2 = ""
                        if is_allowed_mime(ctype2, pdf_link, allowed):
                            res = download_stream(session, pdf_link, max_bytes)
                            if res["sha256"] in seen_hashes:
                                Path(res["path"]).unlink(missing_ok=True)
                                log(
                                    flog,
                                    ri=origin["name"],
                                    source=str(x),
                                    url=pdf_link,
                                    sha256=res["sha256"],
                                    status="duplicate",
                                )
                            else:
                                seen_hashes.add(res["sha256"])
                                log(
                                    flog,
                                    ri=origin["name"],
                                    source=str(x),
                                    url=pdf_link,
                                    **res,
                                    status="ok",
                                )
                                picked += 1
                                success = True
                                print(
                                    f"[fetch] {picked:04d} {res['content_type'] or '?'} {res['bytes']}B → {res['path']}"
                                )
                            break

                # Nada funcionó para este URL
                log(
                    flog,
                    ri=origin["name"],
                    source=str(x),
                    url=url,
                    status="skipped",
                    reason=f"ctype='{ctype}' no permitido",
                )

            except Exception as e:
                log(
                    flog,
                    ri=origin["name"],
                    source=str(x),
                    url=url,
                    status="error",
                    error=str(e),
                )
                continue

        if not success:
            log(flog, ri=origin["name"], source=str(x), status="no_download")

    flog.close()
    print(f"[fetch] Listo. Descargados: {picked}/{args.limit}")


if __name__ == "__main__":
    main()
