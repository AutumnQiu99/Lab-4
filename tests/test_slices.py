"""Regression gate: encode slices as tests with thresholds chosen before seeing the candidate."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from lab_helpers import BASELINE, MODELS, latency_stats, load_predictions, slice_accuracy

if "MODEL" not in os.environ or not os.environ["MODEL"].strip():
    raise RuntimeError(
        "MODEL environment variable is not set. "
        "Run e.g. MODEL=baseline pytest tests/ -v  or  MODEL=candidate_v1 pytest tests/ -v"
    )

MODEL = os.environ["MODEL"]
if MODEL not in MODELS:
    raise RuntimeError(f"Unknown MODEL={MODEL!r}. Expected one of {list(MODELS)}")

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.yaml"


def _load_manifest() -> dict:
    with MANIFEST_PATH.open() as f:
        return yaml.safe_load(f)


MANIFEST = _load_manifest()
SLICE_CASES = [
    (entry["slice"], float(entry["threshold"])) for entry in MANIFEST.get("slices", [])
]
P50_MS_MAX = float(MANIFEST["latency"]["p50_ms_max"])


@pytest.fixture(scope="module")
def pred_df():
    return load_predictions(MODEL)


@pytest.fixture(scope="module")
def baseline_df():
    return load_predictions(BASELINE)


@pytest.mark.parametrize("slice_name,threshold", SLICE_CASES, ids=[s for s, _ in SLICE_CASES])
def test_slice_meets_threshold(slice_name, threshold, pred_df, gate_record):
    acc, n = slice_accuracy(pred_df, slice_name)
    test_id = f"slice::{slice_name}"
    if n < 30:
        gate_record(test_id, acc, threshold, "skipped")
        pytest.skip(f"slice={slice_name} n={n} < 30")
    passed = acc >= threshold
    gate_record(test_id, acc, threshold, "passed" if passed else "failed")
    assert passed, (
        f"slice={slice_name} observed accuracy={acc:.4f} "
        f"threshold={threshold:.4f} n={n}"
    )


@pytest.mark.parametrize("slice_name", [s for s, _ in SLICE_CASES])
def test_slice_no_regression_vs_baseline(slice_name, pred_df, baseline_df, gate_record):
    if MODEL == BASELINE:
        gate_record(f"vs_baseline::{slice_name}", None, None, "skipped")
        pytest.skip("vs-baseline check is skipped when MODEL == baseline")
    cand_acc, cand_n = slice_accuracy(pred_df, slice_name)
    base_acc, base_n = slice_accuracy(baseline_df, slice_name)
    n = min(cand_n, base_n)
    floor = base_acc - 0.02
    test_id = f"vs_baseline::{slice_name}"
    if n < 30:
        gate_record(test_id, cand_acc, floor, "skipped")
        pytest.skip(f"slice={slice_name} n={n} < 30")
    passed = cand_acc >= floor
    gate_record(test_id, cand_acc, floor, "passed" if passed else "failed")
    assert passed, (
        f"slice={slice_name} observed accuracy={cand_acc:.4f} "
        f"threshold={floor:.4f} (baseline={base_acc:.4f} - 0.02) n={n}"
    )


def test_p50_latency(gate_record):
    stats = latency_stats(MODEL)
    test_id = "latency::p50"
    if not stats:
        gate_record(test_id, None, P50_MS_MAX, "skipped")
        pytest.skip(f"no latency column for MODEL={MODEL}")
    p50 = stats["p50"]
    passed = p50 <= P50_MS_MAX
    gate_record(test_id, p50, P50_MS_MAX, "passed" if passed else "failed")
    assert passed, (
        f"check=latency_p50 observed p50={p50:.2f} ms "
        f"threshold={P50_MS_MAX:.2f} ms n={len(load_predictions(MODEL))} tweets "
        f"(p95={stats['p95']:.2f} ms)"
    )
