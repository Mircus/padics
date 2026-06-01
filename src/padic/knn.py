"""
padic.knn
~~~~~~~~~
k-nearest-neighbours classifier in (Q_p^d, d_p) using the product ultrametric.

The product ultrametric on Q_p^d is:
    d(u, v) = max_j  d_p(u_j, v_j)

This is still an ultrametric (the max of ultrametrics is an ultrametric).

Usage
-----
The classifier follows the scikit-learn estimator protocol: fit / predict /
predict_proba / get_params / set_params.  Input data is a list of Q_p^d
vectors (list of list of Qp).

To embed a NumPy float array into Qp^d see :func:`embed_float_array`.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .field import Qp, QpContext
from .metrics import padic_dist


def embed_float_array(X: np.ndarray, ctx: QpContext, scale: int = 100) -> List[List[Qp]]:
    """Embed a float array X of shape (n, d) into Qp^d.

    Each float value is mapped to the integer  round(value * scale)  and
    then converted to Qp via Qp.from_int.  The quality of the embedding
    depends on scale and ctx.prec; larger scale captures more decimal
    precision but requires higher ctx.prec to avoid PrecisionError.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    ctx : QpContext
    scale : int
        Multiplier applied before rounding to integer.

    Returns
    -------
    list of list of Qp
        Shape (n, d) in Qp coordinates.
    """
    result = []
    for row in X:
        result.append([Qp.from_int(ctx, int(round(float(v) * scale))) for v in row])
    return result


class PadicKNNClassifier(BaseEstimator, ClassifierMixin):
    """k-nearest-neighbours classifier in (Q_p^d, max-ultrametric).

    Parameters
    ----------
    ctx : QpContext
        Shared context for all Qp elements.
    k : int, default 3
        Number of neighbours.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels seen during fit.
    """

    def __init__(self, ctx: QpContext, k: int = 3):
        self.ctx = ctx
        self.k = k

    def _dist_vec(self, a: List[Qp], b: List[Qp]) -> float:
        """Product ultrametric distance between two Qp^d vectors."""
        if len(a) != len(b):
            raise ValueError(
                f"Vector length mismatch: {len(a)} vs {len(b)}"
            )
        return max(padic_dist(ai, bi) for ai, bi in zip(a, b))

    def _validate_X(self, X: List[List[Qp]], name: str = "X") -> None:
        """Check that X is a non-empty list of equal-length Qp vectors."""
        if not X:
            raise ValueError(f"{name} is empty")
        d = len(X[0])
        for i, row in enumerate(X):
            if len(row) != d:
                raise ValueError(
                    f"{name}[{i}] has length {len(row)}; expected {d}"
                )
            for j, elem in enumerate(row):
                if not isinstance(elem, Qp):
                    raise TypeError(
                        f"{name}[{i}][{j}] is {type(elem).__name__}, expected Qp"
                    )
                if elem.ctx != self.ctx:
                    raise ValueError(
                        f"{name}[{i}][{j}] has context {elem.ctx}; "
                        f"classifier context is {self.ctx}"
                    )

    def fit(self, X: List[List[Qp]], y: np.ndarray) -> "PadicKNNClassifier":
        """Store training data.

        Parameters
        ----------
        X : list of list of Qp, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Class labels.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If k < 1, k > len(X), or len(X) != len(y).
        """
        self._validate_X(X, "X")
        y_arr = np.asarray(y)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.k > len(X):
            raise ValueError(
                f"k={self.k} exceeds number of training samples ({len(X)})")
        if len(X) != len(y_arr):
            raise ValueError(
                f"X and y must have the same length: {len(X)} vs {len(y_arr)}")
        self._X_fit = X
        self._y_fit = y_arr
        self.classes_ = np.unique(self._y_fit)
        return self

    def predict(self, X: List[List[Qp]]) -> np.ndarray:
        """Predict class labels for X.

        Ties in vote count are broken by choosing the class with the
        smallest total distance to the query among the tied classes.

        Parameters
        ----------
        X : list of list of Qp

        Returns
        -------
        np.ndarray of predicted labels
        """
        check_is_fitted(self, "_X_fit")
        self._validate_X(X, "X")
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X: List[List[Qp]]) -> np.ndarray:
        """Return class-probability estimates based on neighbour vote fractions.

        Parameters
        ----------
        X : list of list of Qp

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        check_is_fitted(self, "_X_fit")
        self._validate_X(X, "X")
        proba = np.array([self._proba_one(x) for x in X])
        return proba

    def _neighbours(self, x: List[Qp]):
        """Return (indices, distances) of the k nearest training points."""
        dists = np.array([self._dist_vec(x, xi) for xi in self._X_fit])
        idx = np.argsort(dists)[: self.k]
        return idx, dists[idx]

    def _predict_one(self, x: List[Qp]):
        idx, dists = self._neighbours(x)
        labels = self._y_fit[idx]
        classes, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        tied = classes[counts == max_count]
        if len(tied) == 1:
            return tied[0]
        # Break ties by sum of distances for each tied class
        best_class = None
        best_total = float("inf")
        for cls in tied:
            mask = labels == cls
            total = dists[mask].sum()
            if total < best_total:
                best_total = total
                best_class = cls
        return best_class

    def _proba_one(self, x: List[Qp]) -> np.ndarray:
        idx, _ = self._neighbours(x)
        labels = self._y_fit[idx]
        actual_k = len(idx)
        proba = np.zeros(len(self.classes_))
        for i, cls in enumerate(self.classes_):
            proba[i] = np.sum(labels == cls) / actual_k
        return proba
