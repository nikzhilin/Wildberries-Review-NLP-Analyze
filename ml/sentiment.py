"""Sentiment classification: optimum + ORT int8 + sort-by-length + dynamic padding."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

_LABEL_MAP: dict[str, str] = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
}

_MAX_LENGTH = 128
_MODELS_DIR = Path("models")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ensure_ort_artifact(model_name: str) -> Path:
    """Export model → ONNX fp32 → int8 via optimum; cache under models/.

    Call from the main process before spawning workers so each worker finds
    the artifact on disk and skips the export step entirely.

    Returns the directory containing the quantized model.onnx.
    """
    slug = model_name.replace("/", "_")
    int8_dir = _MODELS_DIR / f"{slug}_int8"

    if int8_dir.exists() and any(int8_dir.glob("*.onnx")):
        logger.info("ORT int8 artifact found: %s", int8_dir)
        return int8_dir

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    fp32_dir = _MODELS_DIR / f"{slug}_fp32"
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %s → ONNX fp32 via optimum …", model_name)
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    ort_model.save_pretrained(str(fp32_dir))
    logger.info("fp32 saved → %s", fp32_dir)

    logger.info("Quantizing fp32 → int8 …")
    quantizer = ORTQuantizer.from_pretrained(str(fp32_dir))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(int8_dir), quantization_config=qconfig)

    # ORTQuantizer writes model_quantized.onnx; rename so from_pretrained() finds it.
    quantized_file = int8_dir / "model_quantized.onnx"
    if quantized_file.exists():
        quantized_file.rename(int8_dir / "model.onnx")

    logger.info("int8 model saved → %s", int8_dir)
    return int8_dir


class SentimentAnalyzer:
    """Batch sentiment classifier: ORT int8 via optimum, 128 tokens, sort-by-length.

    On first call exports the model to ONNX and quantizes it to int8 via optimum;
    artifacts are cached under models/ so subsequent runs skip the export step.
    Call ensure_ort_artifact() in the main process before spawning workers.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        num_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSequenceClassification

        int8_dir = ensure_ort_artifact(model_name)

        logger.info("Loading ORT int8 session from %s", int8_dir)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = num_threads if num_threads is not None else (os.cpu_count() or 4)

        self._model = ORTModelForSequenceClassification.from_pretrained(
            str(int8_dir), session_options=so
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self._id2label: dict[int, str] = {k: v.upper() for k, v in cfg.id2label.items()}
        self._batch_size = batch_size
        logger.info(
            "SentimentAnalyzer ready (ORT int8, max_len=%d, batch=%d, labels=%s)",
            _MAX_LENGTH,
            batch_size,
            list(self._id2label.values()),
        )

    def predict_batch(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        """Run inference on a sequence of texts.

        Sorts by descending character length so each sub-batch has similar sequence
        lengths — padding=True then pads only to the longest in that sub-batch.
        Original order is restored before returning.

        Args:
            texts: Non-empty strings. Caller must filter out empty strings.

        Returns:
            List of (label, score) in input order.
            Labels: "positive" | "negative" | "neutral".
            Score: softmax probability of the winning class (0–1).

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        texts = list(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        sorted_texts = [texts[i] for i in order]

        results_sorted: list[tuple[str, float]] = []
        for start in range(0, len(sorted_texts), self._batch_size):
            results_sorted.extend(self._infer(sorted_texts[start : start + self._batch_size]))

        out: list[tuple[str, float] | None] = [None] * len(texts)
        for sorted_idx, orig_idx in enumerate(order):
            out[orig_idx] = results_sorted[sorted_idx]
        return out  # type: ignore[return-value]

    def _infer(self, batch: list[str]) -> list[tuple[str, float]]:
        import torch

        enc = self._tokenizer(
            batch,
            max_length=_MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**enc).logits.numpy()

        probs = _softmax(logits)
        label_ids = probs.argmax(axis=-1).tolist()
        top_scores = probs.max(axis=-1).tolist()
        return [
            (_LABEL_MAP.get(self._id2label.get(lid, "NEUTRAL"), "neutral"), float(sc))
            for lid, sc in zip(label_ids, top_scores)
        ]
