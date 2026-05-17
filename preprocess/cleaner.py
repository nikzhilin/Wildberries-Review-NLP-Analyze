import re

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

_MIN_TOKENS = 3
_MAX_TOKENS = 512


def _clean_single(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text).strip()
    tokens = text.split()
    if len(tokens) > _MAX_TOKENS:
        tokens = tokens[:_MAX_TOKENS]
    return " ".join(tokens)


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare text_clean for BERT semantic analysis.

    Steps applied to every review:
    - strip NUL bytes from all string columns (PostgreSQL rejects them)
    - remove HTML tags and URLs
    - lowercase
    - collapse whitespace

    Reviews shorter than _MIN_TOKENS words after cleaning are dropped —
    they carry no semantic signal for BERT.
    Reviews longer than _MAX_TOKENS words are truncated — BERT's hard
    limit is 512 sub-word tokens; word-level truncation is a safe upper bound.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.replace("\x00", "", regex=False)

    df["text_clean"] = df["text"].fillna("").apply(_clean_single)

    token_counts = df["text_clean"].str.split().str.len().fillna(0)
    df = df[token_counts >= _MIN_TOKENS].copy()  # type: ignore[assignment]

    return df
