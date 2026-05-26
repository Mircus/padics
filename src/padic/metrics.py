"""
padic.metrics
~~~~~~~~~~~~~
p-adic absolute value and distance functions.

Provided functions
------------------
- padic_abs(x)               : |x|_p  (float)
- padic_dist(x, y)           : |x − y|_p  (float)
- pairwise_padic_dist(X)     : n×n distance matrix for a list of Qp elements
- pairwise_padic_dist_vec(X) : n×n matrix for a list of Qp^d vectors
                               (product ultrametric: max over coordinates)
"""

from __future__ import annotations
import numpy as np
from typing import List
from .field import Qp, QpContext


def padic_abs(x: Qp) -> float:
    """Return the p-adic absolute value |x|_p = p^{−v_p(x)}.

    |0|_p = 0.
    """
    return x.abs()


def padic_dist(x: Qp, y: Qp) -> float:
    """Return the p-adic distance d_p(x, y) = |x − y|_p.

    This satisfies the ultrametric (strong triangle) inequality:
        d_p(x, z) ≤ max(d_p(x, y), d_p(y, z))
    """
    return x.sub(y).abs()


def pairwise_padic_dist(X: List[Qp]) -> np.ndarray:
    """Return the n×n pairwise distance matrix for a list of Qp elements.

    Parameters
    ----------
    X : list of Qp
        All elements must share the same QpContext.

    Returns
    -------
    D : np.ndarray, shape (n, n), dtype float64
        D[i, j] = d_p(X[i], X[j]).  Symmetric, zero diagonal.
    """
    n = len(X)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = padic_dist(X[i], X[j])
            D[i, j] = D[j, i] = d
    return D


def pairwise_padic_dist_vec(X: List[List[Qp]]) -> np.ndarray:
    """Return the n×n pairwise distance matrix for Qp^d vectors.

    The distance used is the product ultrametric:
        d(u, v) = max_j  d_p(u_j, v_j)

    Parameters
    ----------
    X : list of list of Qp
        Each inner list is a d-dimensional Qp vector.  All elements must
        share the same QpContext and all vectors must have the same length.

    Returns
    -------
    D : np.ndarray, shape (n, n), dtype float64
    """
    n = len(X)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = max(padic_dist(xi, xj) for xi, xj in zip(X[i], X[j]))
            D[i, j] = D[j, i] = d
    return D
