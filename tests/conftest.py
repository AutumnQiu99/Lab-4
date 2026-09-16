"""Collect gate observations into tests/gate_results_{MODEL}.csv at session end."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config._gate_results = []  # type: ignore[attr-defined]


@pytest.fixture
def gate_record(request: pytest.FixtureRequest):
    def _record(test_id: str, observed, threshold, outcome: str) -> None:
        request.config._gate_results.append(  # type: ignore[attr-defined]
            {
                "test_id": test_id,
                "observed": observed,
                "threshold": threshold,
                "outcome": outcome,
            }
        )

    return _record


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    model = os.environ.get("MODEL", "unset")
    out = Path(__file__).resolve().parent / f"gate_results_{model}.csv"
    rows = getattr(session.config, "_gate_results", [])
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_id", "observed", "threshold", "outcome"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
