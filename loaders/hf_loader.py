import pandas as pd
from datasets import load_dataset

from config import settings


def load_reviews(max_rows: int | None = None) -> pd.DataFrame:
    """Load wb-feedbacks from HuggingFace and return a raw DataFrame.

    Uses streaming=True so only the requested rows are downloaded — the full
    dataset (~10 GB) is never materialised on disk.ц
    """
    if max_rows is None:
        max_rows = settings.max_rows

    print(f"Fetching dataset '{settings.hf_dataset}' from HuggingFace (streaming, limit={max_rows:,})...")
    ds = load_dataset(settings.hf_dataset, split="train", streaming=True)
    ds = ds.take(max_rows)

    df = pd.DataFrame(ds)
    print(f"Loaded DataFrame: {len(df):,} rows, columns: {list(df.columns)}")
    return df
