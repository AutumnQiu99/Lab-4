#!/usr/bin/env python3
"""Run a Hugging Face sentiment pipeline over tweets.csv and write prediction columns.

Latency is measured on CPU, batch size 1, **one thread** (the lab's serving budget).
Set OMP/MKL/torch thread counts before importing torch.

Example (run from the repo root, not from this script's directory):

    python scripts/add_model_predictions.py \\
        --hf_id Elron/deberta-v3-large-sentiment \\
        --col deberta
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP *before* numpy/torch import, otherwise they keep a thread pool.
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
):
    os.environ[_k] = "1" if _k != "TOKENIZERS_PARALLELISM" else "false"

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "tweets.csv"

# LABEL_0/1/2 (RoBERTa-style), NEG/NEU/POS (BERTweet-style), and bare 0/1/2.
# All follow the tweet_eval order: 0=negative, 1=neutral, 2=positive.
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    "NEG": "negative",
    "NEU": "neutral",
    "POS": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
}


def normalize_label(raw: str) -> str:
    key = str(raw).strip()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    upper = key.upper()
    if upper in LABEL_MAP:
        return LABEL_MAP[upper]
    return key.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_id", required=True, help="Hugging Face model id")
    parser.add_argument("--col", required=True, help="Column prefix to write (e.g. deberta)")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to tweets.csv (default: repo-root tweets.csv)",
    )
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Only (re)measure {col}_latency_ms; leave existing predictions untouched.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()

    clf = pipeline(
        "text-classification",
        model=args.hf_id,
        truncation=True,
        max_length=128,
        framework="pt",
        device=-1,  # CPU
    )

    preds: list[str] = []
    scores: list[float] = []
    latencies: list[float] = []
    for t in texts:
        t0 = time.perf_counter()
        out = clf(t)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(normalize_label(out["label"]))
        scores.append(float(out["score"]))

    col = args.col
    if args.latency_only:
        if col not in df.columns:
            raise SystemExit(f"--latency-only needs an existing {col!r} column in {csv_path}")
        changed = int((df[col].astype(str).str.lower() != pd.Series(preds)).sum())
        print(f"--latency-only: keeping saved predictions ({changed} would have changed)")
    else:
        df[col] = preds
        df[f"{col}_score"] = scores
    df[f"{col}_latency_ms"] = latencies
    df.to_csv(csv_path, index=False)

    s = pd.Series(latencies, dtype=float)
    p50 = float(s.quantile(0.50))
    p95 = float(s.quantile(0.95))
    print(f"wrote {col}, {col}_score, {col}_latency_ms to {csv_path}")
    print(
        f"n={len(latencies)}  p50_latency_ms={p50:.3f}  p95_latency_ms={p95:.3f}  "
        f"cpu_threads={torch.get_num_threads()}"
    )


if __name__ == "__main__":
    main()
