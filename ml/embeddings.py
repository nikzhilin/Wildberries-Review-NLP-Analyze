"""Batch text embedding with sentence-transformers."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Encode texts to float32 vectors using sentence-transformers.

    Args:
        model_name: HuggingFace model id or local path.
        batch_size: Sub-batch size passed to SentenceTransformer.encode().
        num_threads: PyTorch intra-op thread count (0 = torch default).
    """

    def __init__(self, model_name: str, batch_size: int = 128, num_threads: int = 0) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if num_threads > 0:
            torch.set_num_threads(num_threads)

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        logger.info("EmbeddingEncoder ready (model=%s, batch=%d, threads=%d)",
                    model_name, batch_size, torch.get_num_threads())

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to a float32 array of shape (len(texts), dim).

        Args:
            texts: Non-empty strings. Caller must filter empty strings.

        Returns:
            Float32 array in input order.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
