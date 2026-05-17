from config import settings
from loaders.hf_loader import load_reviews
from preprocess.pipeline import run


def main() -> None:
    print(f"MAX_ROWS={settings.max_rows:,}  BATCH_SIZE={settings.batch_size:,}")
    df = load_reviews()
    run(df, batch_size=settings.batch_size)


if __name__ == "__main__":
    main()
