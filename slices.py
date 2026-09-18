"""Hypothesis-driven tweet metadata and slice masks.

Students: add your own metadata columns and slices here. Do not reuse only
the examples below — extend them with at least five hypothesis-driven slices.
"""

from __future__ import annotations

import pandas as pd
import emoji

# Metadata columns carried through the notebook pivots (df_wide, df_eval).
META_COLS: list[str] = [
    "emoji_count",
    "has_hashtag",
    "has_mention",
    "has_negation",
    "length_bucket",
    "is_all_caps",
    "has_question_mark",
    "has_multiple_sentences",
    "has_exclamation_mark",
    "has_quote",
]

# Contractions (don't, can't, isn't, ...) count as negation. Non-capturing group
# avoids the pandas UserWarning from a capturing group in str.contains.
NEGATION_RE = r"\b(?:not|never|no|\w+n't)\b"

# An ALL-CAPS *word* of 3+ letters ("STOP", "WTF"), not a fully uppercase tweet.
ALL_CAPS_RE = r"\b[A-Z]{3,}\b"

# Sentence terminators; 2 or more means the tweet has multiple sentences.
SENTENCE_END_RE = r"[.!?]+"


def count_emojis(text: str) -> int:
    return sum(ch in emoji.EMOJI_DATA for ch in str(text))


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute slicing metadata on a copy of `df` (requires a `text` column)."""
    out = df.copy()
    text = out["text"].astype(str)
    out["emoji_count"] = text.apply(count_emojis).astype(int)
    out["has_hashtag"] = text.str.contains(r"#\w+", regex=True)
    out["has_mention"] = text.str.contains(r"@\w+", regex=True)
    out["is_all_caps"] = text.str.contains(ALL_CAPS_RE, regex=True)
    out["has_question_mark"] = text.str.contains(r"\?", regex=True)
    out["has_multiple_sentences"] = text.str.count(SENTENCE_END_RE) >= 2
    out["has_exclamation_mark"] = text.str.contains(r"!", regex=True)
    out["has_quote"] = text.str.contains(r"['\"]", regex=True)
    out["has_negation"] = text.str.contains(NEGATION_RE, regex=True, case=False)
    out["length_bucket"] = pd.cut(
        text.str.len(),
        bins=[0, 50, 100, 200, 1000, 10_000],
        labels=["0-50", "51-100", "101-200", "201-1000", "1001+"],
        include_lowest=True,
    ).astype(str)
    return out


def get_slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return boolean masks for named slices. Edit this dict as you add slices.

    Each slice carries the hypothesis it tests: what about these tweets might
    confuse a model? Slices under 30 tweets are skipped by the Step 8 gate.
    """
    return {
        # Emoji carry sentiment the text may not state, and sit outside most
        # tokenizers' vocabularies, so the model has to guess.
        "emoji_gt0": df["emoji_count"] > 0,

        # Negation flips polarity late in the sentence; the model may latch onto
        # the sentiment word and miss the "not" in front of it.
        "has_negation": df["has_negation"] == True,  # noqa: E712

        # Hashtags are often topic tags, not sentiment, but look like emphasis.
        "has_hashtag": df["has_hashtag"] == True,  # noqa: E712

        # A mention makes the tweet part of a conversation the model cannot see.
        "has_mention": df["has_mention"] == True,  # noqa: E712

        # Model might see an exclamation mark and assume the tweet is positive.
        "has_exclamation_mark": df["has_exclamation_mark"] == True,  # noqa: E712

        # Model can be confused by the sentiment of a question: a rhetorical
        # question carries sentiment the literal interrogative form does not.
        "has_question_mark": df["has_question_mark"] == True,  # noqa: E712

        # Model can read ALL-CAPS as emphasis and so under-predict neutral,
        # assuming a shouty tweet must be strongly positive or negative.
        "is_all_caps": df["is_all_caps"] == True,  # noqa: E712

        # Each sentence can carry a different sentiment, so the model may report
        # only the first clause instead of the tweet as a whole.
        "has_multiple_sentences": df["has_multiple_sentences"] == True,  # noqa: E712

        # Model can assume a quote expresses the tweeter's own sentiment rather
        # than someone else's reported speech.
        "has_quote": df["has_quote"] == True,  # noqa: E712    
    }
