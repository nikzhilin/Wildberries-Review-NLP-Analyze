"""CatBoost regression for star-rating prediction from behavioral features."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUM_FEATURES = ["review_length", "word_count", "answer_length", "sentiment_score", "has_answer"]
_CAT_FEATURES = ["sentiment_label", "color"]
_ALL_FEATURES = _NUM_FEATURES + _CAT_FEATURES
_TARGET = "rating"


def _build_pool(df: pd.DataFrame, with_target: bool = True):
    from catboost import Pool

    X = pd.DataFrame(df[_ALL_FEATURES].copy())

    for col in _CAT_FEATURES:
        X[col] = cast(pd.Series, X[col]).fillna("unknown").astype(str)
    for col in _NUM_FEATURES:
        X[col] = pd.Series(
            pd.to_numeric(cast(pd.Series, X[col]), errors="coerce")
        ).fillna(0)

    y = cast(pd.Series, df[_TARGET]).to_numpy().astype(float) if with_target else None
    return Pool(data=X, label=y, cat_features=_CAT_FEATURES)


def _make_model(iterations: int, depth: int, learning_rate: float):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=False,
    )


class RatingPredictor:
    """CatBoost regressor: behavioral features → predicted rating (1–5).

    Features:
        review_length    int    character count of the cleaned review text
        sentiment_score  float  softmax confidence of the winning sentiment class
        has_answer       bool   whether the seller posted a reply
        sentiment_label  cat    "positive" | "negative" | "neutral"
        color            cat    product colour/variant string

    Args:
        iterations: Maximum boosting rounds (early stopping may reduce this).
        depth: Tree depth.
        learning_rate: Gradient step size.
    """

    def __init__(
        self,
        iterations: int = 800,
        depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self._iterations = iterations
        self._depth = depth
        self._learning_rate = learning_rate
        self._model = _make_model(iterations, depth, learning_rate)

    def train(self, df: pd.DataFrame, val_fraction: float = 0.2) -> dict[str, Any]:
        """Train on df, report validation metrics, then retrain on full data.

        The val split is used only for metric reporting and finding the best
        iteration count. The final model is retrained on all rows using that
        iteration count so every review contributes to the artifact.

        Args:
            df: Must contain all feature columns and 'rating'.
            val_fraction: Share held out for metric reporting.

        Returns:
            {'val_rmse': float, 'val_mae': float, 'best_iteration': int}
        """
        from sklearn.model_selection import train_test_split

        stratify = cast(pd.Series, df[_TARGET])
        min_class_count = int(stratify.value_counts().min())
        try:
            splits: list[pd.DataFrame] = train_test_split(  # type: ignore[assignment]
                df,
                test_size=val_fraction,
                random_state=42,
                stratify=stratify if min_class_count >= 2 else None,
            )
        except ValueError:
            splits = train_test_split(  # type: ignore[assignment]
                df, test_size=val_fraction, random_state=42
            )

        train_df, val_df = splits[0], splits[1]

        logger.info(
            "Training on %s rows, validating on %s",
            f"{len(train_df):,}",
            f"{len(val_df):,}",
        )

        eval_model = _make_model(self._iterations, self._depth, self._learning_rate)
        eval_model.set_params(verbose=100)
        eval_model.fit(
            _build_pool(train_df),
            eval_set=_build_pool(val_df),
            use_best_model=True,
        )
        best_iter = int(eval_model.best_iteration_) + 1

        val_preds = eval_model.predict(_build_pool(val_df, with_target=False))
        val_true = cast(pd.Series, val_df[_TARGET]).to_numpy().astype(float)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_true) ** 2)))
        val_mae = float(np.mean(np.abs(val_preds - val_true)))
        logger.info(
            "Val metrics  RMSE=%.4f  MAE=%.4f  best_iteration=%d",
            val_rmse, val_mae, best_iter,
        )

        logger.info(
            "Retraining on full dataset (%s rows, %d iterations)…",
            f"{len(df):,}", best_iter,
        )
        self._model = _make_model(best_iter, self._depth, self._learning_rate)
        self._model.set_params(verbose=100)
        self._model.fit(_build_pool(df))

        return {"val_rmse": val_rmse, "val_mae": val_mae, "best_iteration": best_iter}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings clipped to [1.0, 5.0].

        Args:
            df: DataFrame with the same feature columns used during training.

        Returns:
            float32 array of shape (len(df),).
        """
        preds = self._model.predict(_build_pool(df, with_target=False))
        return np.clip(preds, 1.0, 5.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save the trained model in CatBoost binary format."""
        self._model.save_model(str(path))
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RatingPredictor":
        """Load a previously saved model artifact."""
        from catboost import CatBoostRegressor

        obj = cls.__new__(cls)
        obj._model = CatBoostRegressor()
        obj._model.load_model(str(path))
        logger.info("Model loaded ← %s", path)
        return obj
