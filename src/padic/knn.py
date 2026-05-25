
from __future__ import annotations
import numpy as np
from typing import Optional, List
from sklearn.base import BaseEstimator, ClassifierMixin
from .field import Qp, QpContext
from .metrics import padic_dist

class PadicKNNClassifier(BaseEstimator, ClassifierMixin):
    """kNN in (Qp^d, d_p) using max metric across coordinates (ultrametric product).
    Simple, demonstrative; not optimized.
    """
    def __init__(self, ctx: QpContext, k: int = 3):
        self.ctx = ctx
        self.k = k
        self._X: Optional[List[List[Qp]]] = None
        self._y: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _dist_vec(self, a: List[Qp], b: List[Qp]) -> float:
        # product ultrametric: max_j |a_j - b_j|_p
        return max(padic_dist(ai, bi) for ai, bi in zip(a, b))

    def fit(self, X: List[List[Qp]], y: np.ndarray):
        self._X = X
        self._y = np.asarray(y)
        self.classes_ = np.unique(self._y)
        return self

    def predict(self, X: List[List[Qp]]) -> np.ndarray:
        assert self._X is not None and self._y is not None
        out = []
        for x in X:
            dists = np.array([self._dist_vec(x, xi) for xi in self._X])
            idx = np.argsort(dists)[:self.k]
            votes, counts = np.unique(self._y[idx], return_counts=True)
            out.append(votes[np.argmax(counts)])
        return np.array(out)
