"""RUL models.

``make_model`` returns scikit-learn's RandomForest when available, otherwise a
numpy-only ridge regression so the pipeline runs with no third-party ML deps.
Both live here (not in a ``__main__`` script) so pickled models reload cleanly
regardless of how the training script was launched.
"""

import numpy as np


class RidgeFallback:
    """Closed-form ridge regression (numpy only)."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.w: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        X1 = np.hstack([np.ones((len(X), 1)), X])
        A = X1.T @ X1 + self.alpha * np.eye(X1.shape[1])
        self.w = np.linalg.solve(A, X1.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X1 = np.hstack([np.ones((len(X), 1)), X])
        return X1 @ self.w


def make_model(seed: int = 42):
    """Return (model, name). Prefers sklearn RandomForest; falls back to ridge.

    ``seed`` is exposed so the forest can take part in the seed-averaged comparison on the
    same footing as the sequence models. Its only stochastic element is the forest's own
    bootstrap and feature sampling — there is no train/val split to move — so its spread
    across seeds is small, but reporting it measured rather than assumed is the point.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed
        )
        return model, "RandomForestRegressor (sklearn)"
    except ImportError:
        return RidgeFallback(alpha=10.0), (
            "RidgeFallback (numpy) — install scikit-learn for the real baseline"
        )
