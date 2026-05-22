"""Stage 2 entrypoint: read reviews from DB → sentiment → embeddings → clustering → rating."""
from __future__ import annotations

import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from config import settings
from db import SessionLocal
from db.models import ReviewML, Topic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

N_WORKERS = 2


def _load_unprocessed_reviews(session) -> tuple[list[int], list[str]]:
    """Return IDs and texts for reviews that have no sentiment label yet.

    LEFT JOIN skips rows already present in review_ml with a non-null label,
    so re-running after a partial run or crash picks up exactly where it left off.
    """
    rows = session.execute(
        text(
            """
            SELECT r.id, r.text_clean
            FROM reviews r
            LEFT JOIN review_ml m ON m.review_id = r.id
            WHERE r.text_clean IS NOT NULL
              AND r.text_clean != ''
              AND m.sentiment_label IS NULL
            ORDER BY r.id
            """
        )
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def _sentiment_worker(
    args: tuple[str, int, int, list[int], list[str]],
) -> list[tuple[int, str, float]]:
    """Multiprocessing worker: owns its own ONNX session.

    Deduplicates texts within the chunk before inference so identical reviews
    (copy-paste on WB) are classified only once.

    Args:
        args: (model_name, batch_size, num_threads, ids, texts)

    Returns:
        List of (review_id, label, score) in the same order as ids/texts.
    """
    model_name, batch_size, num_threads, ids, texts = args

    # Import inside the worker so the class is not pickled across process boundary.
    from ml.sentiment import SentimentAnalyzer

    # Exact-dedup: dict.fromkeys preserves insertion order (Python 3.7+).
    unique_texts = list(dict.fromkeys(texts))
    analyzer = SentimentAnalyzer(model_name, batch_size=batch_size, num_threads=num_threads)
    preds = analyzer.predict_batch(unique_texts)
    text_to_result: dict[str, tuple[str, float]] = dict(zip(unique_texts, preds))

    return [
        (rid, text_to_result[t][0], text_to_result[t][1])
        for rid, t in zip(ids, texts)
    ]


def _upsert_sentiment_batch(
    session,
    ids: list[int],
    labels: list[str],
    scores: list[float],
) -> None:
    """Upsert sentiment results into review_ml.

    On conflict updates only sentiment_label and sentiment_score —
    never touches topic_id, fake_score, predicted_rating, or embedding.

    Args:
        session: SQLAlchemy session (caller commits).
        ids: Review primary keys.
        labels: Sentiment labels ("positive" | "negative" | "neutral").
        scores: Softmax probabilities of the winning class.
    """
    if not ids:
        return

    values = [
        {"review_id": rid, "sentiment_label": lbl, "sentiment_score": sc}
        for rid, lbl, sc in zip(ids, labels, scores)
    ]
    stmt = pg_insert(ReviewML).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["review_id"],
        set_={
            "sentiment_label": stmt.excluded.sentiment_label,
            "sentiment_score": stmt.excluded.sentiment_score,
        },
    )
    session.execute(stmt)


def run_sentiment(
    model_name: str,
    batch_size: int,
    upsert_chunk_size: int = 5_000,
) -> None:
    """Read unprocessed reviews, classify sentiment in parallel, upsert results.

    Incremental: skips reviews that already have a sentiment_label in review_ml.
    Parallel: splits work across N_WORKERS processes, each with its own ONNX session.
    Dedup: each worker deduplicates its chunk before inference.
    Upsert happens in the main process after all workers finish.

    Args:
        model_name: HuggingFace model identifier or local path.
        batch_size: Inference sub-batch size passed to each worker.
        upsert_chunk_size: Rows per upsert statement.
    """
    with SessionLocal() as session:
        ids, texts = _load_unprocessed_reviews(session)

    total = len(ids)
    if not total:
        logger.info("No unprocessed reviews — nothing to do.")
        return

    logger.info("Unprocessed reviews: %s  workers: %d", f"{total:,}", N_WORKERS)

    # Each worker gets cpu_count // N_WORKERS ORT threads so they don't fight.
    num_threads = max(1, (os.cpu_count() or 4) // N_WORKERS)

    mid = (total + 1) // 2  # ceiling so worker-0 gets the bigger half when odd
    worker_args: list[tuple] = [
        (model_name, batch_size, num_threads, ids[:mid], texts[:mid]),
        (model_name, batch_size, num_threads, ids[mid:], texts[mid:]),
    ]

    # Build the ORT artifact in the main process before forking workers.
    # Without this, N workers all see a missing artifact and race to export it.
    from ml.sentiment import ensure_ort_artifact
    ensure_ort_artifact(model_name)

    with Pool(N_WORKERS) as pool:
        parts = pool.map(_sentiment_worker, worker_args)

    all_results = parts[0] + parts[1]
    all_ids    = [r[0] for r in all_results]
    all_labels = [r[1] for r in all_results]
    all_scores = [r[2] for r in all_results]

    with SessionLocal() as session:
        with tqdm(total=total, desc="Upsert", unit="rev") as pbar:
            for start in range(0, total, upsert_chunk_size):
                end = start + upsert_chunk_size
                _upsert_sentiment_batch(
                    session,
                    all_ids[start:end],
                    all_labels[start:end],
                    all_scores[start:end],
                )
                session.commit()
                pbar.update(min(upsert_chunk_size, total - start))

    logger.info("Done. Sentiment written for %s reviews.", f"{total:,}")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _load_reviews_without_embeddings(session) -> tuple[list[int], list[str]]:
    """Return IDs and texts for reviews that have no embedding yet."""
    rows = session.execute(
        text(
            """
            SELECT r.id, r.text_clean
            FROM reviews r
            LEFT JOIN review_ml m ON m.review_id = r.id
            WHERE r.text_clean IS NOT NULL
              AND r.text_clean != ''
              AND m.embedding IS NULL
            ORDER BY r.id
            """
        )
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def _upsert_embeddings_batch(session, ids: list[int], vecs: np.ndarray) -> None:
    """Upsert embeddings into review_ml; never touches other columns."""
    if not ids:
        return
    values = [{"review_id": rid, "embedding": vecs[i].tolist()} for i, rid in enumerate(ids)]
    stmt = pg_insert(ReviewML).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["review_id"],
        set_={"embedding": stmt.excluded.embedding},
    )
    session.execute(stmt)


def run_embeddings(model_name: str, batch_size: int, chunk_size: int = 5_000) -> None:
    """Encode reviews without embeddings and save incrementally.

    Single-process with all CPU threads: no pickle overhead, progressive saves,
    fully resumable on restart.
    """
    import torch
    from ml.embeddings import EmbeddingEncoder

    torch.set_num_threads(os.cpu_count() or 4)
    encoder = EmbeddingEncoder(model_name, batch_size=batch_size)

    with SessionLocal() as session:
        ids, texts = _load_reviews_without_embeddings(session)

    total = len(ids)
    if not total:
        logger.info("No reviews without embeddings — skipping.")
        return

    logger.info("Reviews to embed: %s  chunk_size: %d  threads: %d",
                f"{total:,}", chunk_size, torch.get_num_threads())

    with tqdm(total=total, desc="Embeddings", unit="rev") as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_ids = ids[start:end]
            chunk_texts = texts[start:end]

            vecs = encoder.encode_batch(chunk_texts)

            with SessionLocal() as session:
                _upsert_embeddings_batch(session, chunk_ids, vecs)
                session.commit()

            pbar.update(end - start)

    logger.info("Done. Embeddings written for %s reviews.", f"{total:,}")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _load_product_ids_for_clustering(session) -> list[int]:
    """Return product IDs that have at least one review with embedding + sentiment."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT r.product_id
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            WHERE m.embedding IS NOT NULL
              AND m.sentiment_label IS NOT NULL
            ORDER BY r.product_id
            """
        )
    ).fetchall()
    return [r[0] for r in rows]


def _load_product_reviews(
    session, product_id: int
) -> tuple[list[int], list[list[float]], list[str], list[str]]:
    """Return (review_ids, embeddings, sentiment_labels, texts) for one product.

    Embeddings arrive from pgvector as raw strings over a text() query and are
    parsed here with json.loads — pgvector uses JSON-array wire format.
    """
    import json

    rows = session.execute(
        text(
            """
            SELECT m.review_id, m.embedding::text, m.sentiment_label,
                   COALESCE(r.text_clean, '') AS text_clean
            FROM review_ml m
            JOIN reviews r ON r.id = m.review_id
            WHERE r.product_id = :pid
              AND m.embedding IS NOT NULL
              AND m.sentiment_label IS NOT NULL
            ORDER BY m.review_id
            """
        ),
        {"pid": product_id},
    ).fetchall()
    ids    = [r[0] for r in rows]
    embs   = [json.loads(r[1]) for r in rows]
    labels = [r[2] for r in rows]
    texts  = [r[3] for r in rows]
    return ids, embs, labels, texts


def run_clustering(min_reviews: int = 50) -> None:
    """Run per-product BERTopic clustering for pos and neg reviews.

    For each product:
      - Loads reviews with embeddings + sentiment from DB
      - Runs BERTopic separately for positive and negative polarity
      - Deletes old topics for (product_id, polarity) before inserting new ones
      - Outliers (topic=-1) are stored as a topic with keywords="other"
      - Updates review_ml.topic_id for every assigned review
    """
    from ml.clustering import cluster_product

    with SessionLocal() as session:
        product_ids = _load_product_ids_for_clustering(session)

    if not product_ids:
        logger.info("No products ready for clustering — skipping.")
        return

    logger.info("Products to cluster: %d", len(product_ids))

    for product_id in tqdm(product_ids, desc="Clustering products", unit="prod"):
        with SessionLocal() as session:
            ids, embs_raw, labels, texts = _load_product_reviews(session, product_id)

        if not ids:
            continue

        embeddings = np.array(embs_raw, dtype=np.float32)
        topic_results = cluster_product(
            product_id=product_id,
            review_ids=ids,
            embeddings=embeddings,
            sentiment_labels=labels,
            texts=texts,
            min_reviews=min_reviews,
        )

        if not topic_results:
            continue

        with SessionLocal() as session:
            # Delete old topics for this product; cascades are not set, so update
            # review_ml.topic_id to NULL first to avoid FK violations.
            polarities_present = list({tr.polarity for tr in topic_results})
            for polarity in polarities_present:
                old_topic_ids = session.execute(
                    text(
                        "SELECT id FROM topics WHERE product_id=:pid AND polarity=:pol"
                    ),
                    {"pid": product_id, "pol": polarity},
                ).scalars().all()
                if old_topic_ids:
                    session.execute(
                        text(
                            "UPDATE review_ml SET topic_id = NULL "
                            "WHERE topic_id = ANY(:ids)"
                        ),
                        {"ids": old_topic_ids},
                    )
                    session.execute(
                        delete(Topic).where(
                            Topic.product_id == product_id,
                            Topic.polarity == polarity,
                        )
                    )

            # Insert new topics and collect topic_id → review_ids mapping.
            review_to_topic: dict[int, int] = {}
            for tr in topic_results:
                topic = Topic(
                    product_id=product_id,
                    polarity=tr.polarity,
                    keywords=tr.keywords,
                    review_count=tr.review_count,
                )
                session.add(topic)
                session.flush()  # populate topic.id
                for rid in tr.review_ids:
                    review_to_topic[rid] = int(topic.id)  # type: ignore[arg-type]

            # Bulk-update review_ml.topic_id.
            if review_to_topic:
                session.execute(
                    text(
                        "UPDATE review_ml SET topic_id = v.tid "
                        "FROM (VALUES " +
                        ", ".join(f"({rid}, {tid})" for rid, tid in review_to_topic.items()) +
                        ") AS v(rid, tid) "
                        "WHERE review_ml.review_id = v.rid"
                    )
                )

            session.commit()

    logger.info("Clustering done.")


# ---------------------------------------------------------------------------
# Rating predictor
# ---------------------------------------------------------------------------

_RATING_MODEL_PATH = Path("models/rating_predictor_v1.cbm")


def _load_reviews_for_rating(session) -> pd.DataFrame:
    """Return all reviews that have sentiment scores, with their features."""
    rows = session.execute(
        text(
            """
            SELECT r.id, r.review_length, r.has_answer::int AS has_answer,
                   r.color, m.sentiment_label, m.sentiment_score, r.rating
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            WHERE r.rating IS NOT NULL
              AND m.sentiment_label IS NOT NULL
              AND m.sentiment_score IS NOT NULL
            ORDER BY r.id
            """
        )
    ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["id", "review_length", "has_answer", "color",
                 "sentiment_label", "sentiment_score", "rating"],
    )


def _upsert_predicted_ratings(
    session, ids: list[int], preds: np.ndarray, chunk_size: int = 5_000
) -> None:
    """Upsert predicted_rating into review_ml; never touches other columns."""
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        values = [
            {"review_id": ids[i], "predicted_rating": float(preds[i])}
            for i in range(start, end)
        ]
        stmt = pg_insert(ReviewML).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["review_id"],
            set_={"predicted_rating": stmt.excluded.predicted_rating},
        )
        session.execute(stmt)
        session.commit()


def run_rating_predictor(model_path: Path = _RATING_MODEL_PATH) -> None:
    """Train CatBoost (if no artifact exists) and predict ratings for all reviews.

    Idempotent: re-running overwrites predicted_rating with fresh predictions.
    Model artifact is cached in models/ and reused on subsequent runs.
    To force retraining, delete the artifact file.
    """
    from ml.rating_predictor import RatingPredictor

    with SessionLocal() as session:
        df = _load_reviews_for_rating(session)

    if df.empty:
        logger.info("No reviews with sentiment features — skipping rating predictor.")
        return

    logger.info("Reviews available for rating prediction: %s", f"{len(df):,}")

    if model_path.exists():
        logger.info("Loading existing model from %s", model_path)
        predictor = RatingPredictor.load(model_path)
    else:
        logger.info("No model artifact found — training from scratch.")
        predictor = RatingPredictor()
        metrics = predictor.train(df)
        logger.info("Training done: %s", metrics)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save(model_path)

    logger.info("Running predictions…")
    preds = predictor.predict(df)

    ids = df["id"].tolist()
    with SessionLocal() as session:
        _upsert_predicted_ratings(session, ids, preds)

    logger.info(
        "Done. predicted_rating written for %s reviews  "
        "(mean=%.3f  std=%.3f)",
        f"{len(ids):,}",
        float(preds.mean()),
        float(preds.std()),
    )


def run_alerts() -> None:
    """Evaluate all threshold rules and persist a fresh snapshot to the alerts table.

    Deletes all existing alerts before inserting so each run produces a
    consistent, non-duplicated snapshot.  Safe to re-run at any time.
    """
    from collections import Counter
    from datetime import datetime, timezone

    from alerts.rules import evaluate_all  # type: ignore[import-untyped]
    from db.models import Alert

    with SessionLocal() as session:
        result = session.execute(text("DELETE FROM alerts"))
        deleted: int = getattr(result, "rowcount", 0)
        if deleted:
            logger.info("Cleared %d stale alerts.", deleted)

        alert_rows = evaluate_all(session)

        if not alert_rows:
            logger.info("No alerts triggered.")
            session.commit()
            return

        now = datetime.now(timezone.utc)
        session.add_all(
            [
                Alert(
                    product_id=a.product_id,
                    rule_name=a.rule_name,
                    severity=a.severity,
                    details=a.details,
                    created_at=now,
                )
                for a in alert_rows
            ]
        )
        session.commit()

    by_severity = Counter(a.severity for a in alert_rows)
    logger.info(
        "Alerts written: %d total  (high=%d  medium=%d  low=%d)",
        len(alert_rows),
        by_severity.get("high", 0),
        by_severity.get("medium", 0),
        by_severity.get("low", 0),
    )


def main() -> None:
    logger.info(
        "sentiment_model=%s  embedding_model=%s  workers=%d",
        settings.sentiment_model,
        settings.embedding_model,
        N_WORKERS,
    )
    run_sentiment(
        model_name=settings.sentiment_model,
        batch_size=settings.sentiment_batch_size,
        upsert_chunk_size=settings.batch_size,
    )
    run_embeddings(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        chunk_size=settings.batch_size,
    )
    run_clustering(min_reviews=settings.min_cluster_reviews)
    run_rating_predictor()
    run_alerts()


if __name__ == "__main__":
    main()
