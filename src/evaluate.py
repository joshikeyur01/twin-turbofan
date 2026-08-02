"""RUL evaluation metrics.

``rmse`` is symmetric. ``phm_score`` is the asymmetric scoring function from the
PHM'08 / C-MAPSS challenge: predicting failure LATE (d > 0, over-estimating RUL)
is penalised more heavily than predicting it early, because a late prediction
means the part fails in service.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def phm_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred) - np.asarray(y_true)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))
