"""Per-product BERTopic clustering, separate for positive and negative reviews."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_POLARITY_MAP = {
    "positive": "pos",
    "negative": "neg",
}
_TOP_WORDS = 10


@dataclass
class TopicResult:
    """Clustering output for one polarity of one product."""

    polarity: str           # "pos" | "neg"
    keywords: str           # comma-separated top words, or "other"
    review_count: int
    review_ids: list[int] = field(default_factory=list)


def _build_bertopic():
    """Construct a BERTopic instance with fixed UMAP + HDBSCAN config."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_epochs=200,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[а-яёa-zA-Z][а-яёa-zA-Z\-]{2,}\b",
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )


def _keywords_for_topic(topic_model, topic_id: int) -> str:
    """Return top-N words for a BERTopic topic as a comma-separated string.

    BERTopic returns ('', 1e-05) placeholder entries when a topic has no
    distinctive vocabulary — those are filtered out before joining.
    """
    words = topic_model.get_topic(topic_id)
    if not words:
        return "other"
    meaningful = [w for w, _ in words[:_TOP_WORDS] if w.strip()]
    return ", ".join(meaningful) if meaningful else "other"


def cluster_product(
    product_id: int,
    review_ids: list[int],
    embeddings: np.ndarray,
    sentiment_labels: list[str],
    texts: list[str],
    min_reviews: int = 50,
) -> list[TopicResult]:
    """Run BERTopic per polarity for a single product.

    Args:
        product_id: DB primary key (used only for logging).
        review_ids: Review PKs aligned with the other lists.
        embeddings: Float32 array of shape (N, dim).
        sentiment_labels: "positive" | "negative" | "neutral" per review.
        texts: Cleaned review texts used by BERTopic's TF-IDF for keyword extraction.
        min_reviews: Skip a polarity if it has fewer than this many reviews.

    Returns:
        List of TopicResult — one per discovered topic (including "other" for
        outliers) across both polarities.  Empty if all polarities are skipped.
    """
    results: list[TopicResult] = []

    for sentiment_label, polarity in _POLARITY_MAP.items():
        mask = [i for i, lbl in enumerate(sentiment_labels) if lbl == sentiment_label]
        if len(mask) < min_reviews:
            logger.debug(
                "product_id=%d polarity=%s: %d reviews < %d, skipping",
                product_id,
                polarity,
                len(mask),
                min_reviews,
            )
            continue

        pol_ids   = [review_ids[i] for i in mask]
        pol_embs  = embeddings[mask]
        pol_texts = [texts[i] for i in mask]

        logger.info(
            "product_id=%d polarity=%s: fitting BERTopic on %d reviews",
            product_id,
            polarity,
            len(pol_ids),
        )

        topic_model = _build_bertopic()
        try:
            topics, _ = topic_model.fit_transform(pol_texts, embeddings=pol_embs)
        except Exception:
            logger.exception("BERTopic failed for product_id=%d polarity=%s", product_id, polarity)
            continue

        # Group review_ids by assigned topic.
        topic_to_ids: dict[int, list[int]] = {}
        for rid, tid in zip(pol_ids, topics):
            topic_to_ids.setdefault(tid, []).append(rid)

        # Convert each BERTopic topic id to a TopicResult.
        for tid, rids in topic_to_ids.items():
            if tid == -1:
                keywords = "other"
            else:
                keywords = _keywords_for_topic(topic_model, tid)
            results.append(
                TopicResult(
                    polarity=polarity,
                    keywords=keywords,
                    review_count=len(rids),
                    review_ids=rids,
                )
            )

    return results
