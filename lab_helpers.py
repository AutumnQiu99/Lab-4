"""Helpers for the Step 8 regression gate (load predictions, slice accuracy, latency)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slices import add_metadata, get_slices

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tweets.csv"

# Single source of truth for the model registry: lab4.ipynb imports this too, so the
# notebook and the gate can never disagree. `csv_col` is the column prefix in tweets.csv.
MODELS: dict[str, dict[str, str]] = {
    "baseline": {
        "hf_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "csv_col": "roberta",
    },
    "candidate_v1": {
        "hf_id": "LYTinn/finetuning-sentiment-model-tweet-gpt2",
        "csv_col": "gpt2",
    },
    "candidate_v2": {
        "hf_id": "Elron/deberta-v3-large-sentiment",
        "csv_col": "deberta",
    },
}

BASELINE = "baseline"


def load_predictions(model_name: str) -> pd.DataFrame:
    """Read tweets.csv and return text, label, pred, conf plus recomputed metadata."""
    if model_name not in MODELS:
        raise KeyError(f"Unknown model {model_name!r}. Expected one of {list(MODELS)}")
    col = MODELS[model_name]["csv_col"]
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if col not in df.columns or f"{col}_score" not in df.columns:
        raise KeyError(
            f"tweets.csv has no columns {col!r} / {col}_score. "
            f"Expected columns {col!r} and {col}_score in tweets.csv."
        )
    out = pd.DataFrame(
        {
            "text": df["text"],
            "label": df["label"].astype(str).str.lower(),
            "pred": df[col].astype(str).str.lower(),
            "conf": pd.to_numeric(df[f"{col}_score"], errors="coerce"),
        }
    )
    lat_col = f"{col}_latency_ms"
    if lat_col in df.columns:
        out["latency_ms"] = pd.to_numeric(df[lat_col], errors="coerce")
    return add_metadata(out)


def slice_accuracy(df: pd.DataFrame, slice_name: str) -> tuple[float, int]:
    """Return (accuracy, n) for a named slice from get_slices(df)."""
    slices = get_slices(df)
    if slice_name not in slices:
        raise KeyError(f"Unknown slice {slice_name!r}. Known: {list(slices)}")
    g = df.loc[slices[slice_name]]
    n = int(len(g))
    if n == 0:
        return float("nan"), 0
    acc = float((g["pred"] == g["label"]).mean())
    return acc, n


def latency_stats(model_name: str) -> dict:
    """p50/p95 from `{csv_col}_latency_ms` if present; empty dict otherwise."""
    if model_name not in MODELS:
        raise KeyError(f"Unknown model {model_name!r}. Expected one of {list(MODELS)}")
    col = MODELS[model_name]["csv_col"]
    lat_col = f"{col}_latency_ms"
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if lat_col not in df.columns:
        return {}
    s = pd.to_numeric(df[lat_col], errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
    }
