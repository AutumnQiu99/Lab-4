"""Hypothesis-driven tweet metadata and slice masks.

Students: add your own metadata columns and slices here. Do not reuse only
the examples below — extend them with at least five hypothesis-driven slices.
"""

from __future__ import annotations

import pandas as pd
import emoji

# Five example metadata columns used throughout the notebook pivots.
META_COLS: list[str] = [
    "emoji_count",
    "has_hashtag",
    "has_mention",
    "has_negation",
    "length_bucket",
]

# Contractions (don't, can't, isn't, ...) count as negation. Non-capturing group
# avoids the pandas UserWarning from a capturing group in str.contains.
NEGATION_RE = r"\b(?:not|never|no|\w+n't)\b"


def count_emojis(text: str) -> int:
    return sum(ch in emoji.EMOJI_DATA for ch in str(text))


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute slicing metadata on a copy of `df` (requires a `text` column)."""
    out = df.copy()
    text = out["text"].astype(str)
    out["emoji_count"] = text.apply(count_emojis).astype(int)
    out["has_hashtag"] = text.str.contains(r"#\w+", regex=True)
    out["has_mention"] = text.str.contains(r"@\w+", regex=True)
    out["has_negation"] = text.str.contains(NEGATION_RE, regex=True, case=False)
    out["length_bucket"] = pd.cut(
        text.str.len(),
        bins=[0, 50, 100, 200, 1000, 10_000],
        labels=["0-50", "51-100", "101-200", "201-1000", "1001+"],
        include_lowest=True,
    ).astype(str)
    return out


def get_slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return boolean masks for named slices. Edit this dict as you add slices."""
    return {
        "emoji_gt3": df["emoji_count"] > 3,
        "has_negation": df["has_negation"] == True,  # noqa: E712
        "has_hashtag": df["has_hashtag"] == True,  # noqa: E712
    }
