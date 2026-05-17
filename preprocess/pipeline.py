import hashlib

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from db import SessionLocal
from db.models import Product, Review
from preprocess.cleaner import clean_text
from preprocess.features import engineer_features
from config import settings

MARKETPLACE = settings.marketplace


def _upsert_products(session, df: pd.DataFrame) -> dict[int, int]:
    """Upsert products, return {nmId -> product.id}."""
    nm_ids = df["nmId"].unique().tolist()
    str_ids = [str(n) for n in nm_ids]

    values = [{"marketplace": MARKETPLACE, "external_id": str(nm_id)} for nm_id in nm_ids]
    stmt = pg_insert(Product).values(values)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_products_marketplace_external_id")
    session.execute(stmt)
    session.flush()

    all_products = session.query(Product).filter(
        Product.marketplace == MARKETPLACE,
        Product.external_id.in_(str_ids),
    ).all()

    return {int(p.external_id): p.id for p in all_products}


def _upsert_batch(session, batch: pd.DataFrame, product_map: dict[int, int]) -> int:
    """Upsert reviews (insert or update on conflict), return count upserted."""
    # deduplicate within the batch itself to avoid duplicate-key on multi-row insert
    batch = batch.drop_duplicates(subset="external_id")

    values = []
    for row in batch.to_dict("records"):
        product_id = product_map.get(int(row["nmId"]))
        if product_id is None:
            continue
        values.append({
            "product_id": product_id,
            "external_id": row["external_id"],
            "text": "" if pd.isna(row["text"]) else row["text"],
            "text_clean": "" if pd.isna(row["text_clean"]) else row["text_clean"],
            "rating": int(row["productValuation"]) if pd.notna(row["productValuation"]) else None,
            "color": row["color"] if bool(pd.notna(row.get("color"))) else None,
            "answer": row["answer"] if bool(pd.notna(row.get("answer"))) else None,
            "has_answer": bool(row["has_answer"]),
            "review_length": int(row["review_length"]),
            "word_count": int(row["word_count"]),
            "answer_length": int(row["answer_length"]),
        })

    if not values:
        return 0

    stmt = pg_insert(Review).values(values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_reviews_external_id",
        set_={
            "text_clean": stmt.excluded.text_clean,
            "rating": stmt.excluded.rating,
            "color": stmt.excluded.color,
            "answer": stmt.excluded.answer,
            "has_answer": stmt.excluded.has_answer,
            "review_length": stmt.excluded.review_length,
            "word_count": stmt.excluded.word_count,
            "answer_length": stmt.excluded.answer_length,
        },
    )
    session.execute(stmt)
    return len(values)


def run(df: pd.DataFrame, batch_size: int = 5_000) -> None:
    """Stage 1 pipeline: clean → features → persist to DB."""
    print("Cleaning text...")
    df = clean_text(df)

    # Drop reviews with no text — they're unanalysable and would break the hash
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()  # type: ignore[assignment]

    # Normalise nmId to Int64 so astype(str) gives "12345" not "12345.0"
    df["nmId"] = df["nmId"].astype("Int64")

    print("Engineering features...")
    df = engineer_features(df)

    print("Building review external IDs...")
    df["external_id"] = (df["nmId"].astype(str) + ":" + df["text"]).apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:32]
    )

    with SessionLocal() as session:
        print("Upserting products...")
        product_map = _upsert_products(session, df)
        session.commit()
        print(f"  {len(product_map):,} products in DB")

        total_inserted = 0
        n_batches = (len(df) + batch_size - 1) // batch_size

        for start in tqdm(range(0, len(df), batch_size), total=n_batches, desc="Saving reviews"):
            batch = df.iloc[start : start + batch_size]
            total_inserted += _upsert_batch(session, batch, product_map)
            session.commit()

    print(f"Done. Upserted {total_inserted:,} reviews.")
