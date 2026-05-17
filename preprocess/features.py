import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["text"].fillna("")
    answer = df["answer"].fillna("")

    df["review_length"] = text.str.len().astype(int)
    df["word_count"] = text.str.split().str.len().fillna(0).astype(int)
    df["has_answer"] = answer.str.strip().str.len() > 0
    df["answer_length"] = answer.str.len().astype(int)

    return df
