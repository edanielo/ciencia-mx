#!/usr/bin/env python3
"""
metrics_report.py — Lee corpus/raw/<RI>/manifest.jsonl y genera métricas y muestras.
Salida: reports/<RI>/metrics.json + (opcional) sample_low.jsonl
"""

from __future__ import annotations
import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List

CORPUS = Path("corpus/raw")
REPORTS = Path("reports")


def load_manifest(ri: str) -> List[Dict[str, Any]]:
    mpath = CORPUS / ri / "manifest.jsonl"
    if not mpath.exists():
        raise SystemExit(f"ERROR: no existe {mpath}")
    rows: List[Dict[str, Any]] = []
    with mpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return float(xs_sorted[f])
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return float(d0 + d1)


def main():
    ap = argparse.ArgumentParser(description="Métricas del manifest del corpus")
    ap.add_argument("--ri", required=True)
    ap.add_argument("--outdir", help="Carpeta de reporte (por defecto reports/<RI>)")
    ap.add_argument(
        "--sample-low", type=int, default=15, help="Muestra de N docs con menor score"
    )
    args = ap.parse_args()

    rows = load_manifest(args.ri)
    if not rows:
        raise SystemExit("Manifest vacío.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir) if args.outdir else (REPORTS / args.ri)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    ocr = sum(1 for r in rows if r.get("ocr_applied"))
    tokens = [r.get("metrics", {}).get("tokens", 0) for r in rows]
    alpha = [r.get("metrics", {}).get("alpha_ratio", 0.0) for r in rows]
    score = [r.get("metrics", {}).get("text_quality_score", 0.0) for r in rows]

    data = {
        "ri": args.ri,
        "docs": n,
        "ocr_docs": ocr,
        "ocr_pct": round(100.0 * ocr / max(1, n), 2),
        "tokens_median": int(stats.median(tokens)) if tokens else 0,
        "tokens_p25": int(percentile(tokens, 0.25)),
        "tokens_p75": int(percentile(tokens, 0.75)),
        "alpha_median": round(stats.median(alpha), 4) if alpha else 0.0,
        "score_median": round(stats.median(score), 3) if score else 0.0,
    }

    # muestra de casos con menor score (para inspección)
    sample = sorted(
        rows, key=lambda r: r.get("metrics", {}).get("text_quality_score", 0.0)
    )[: args.sample_low]

    (outdir / "metrics.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (outdir / "sample_low.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[metrics] {args.ri} → docs={n}, ocr%={data['ocr_pct']}, "
        f"tokens~med={data['tokens_median']} (p25={data['tokens_p25']}, p75={data['tokens_p75']})"
    )


if __name__ == "__main__":
    main()
