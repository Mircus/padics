"""
padic — p-adic foundations for data science.

Quick start::

    from padic import QpContext, Qp, QpBall, padic_dist
    ctx = QpContext(p=5, prec=8)
    x = Qp.from_rational(ctx, 7, 12)
    y = Qp.from_int(ctx, 10)
    print(padic_dist(x, y))
"""

from .field import QpContext, Qp, PrecisionError
from .ball import QpBall
from .hensel import hensel_lift_simple
from .btree import (
    BTRootedTree,
    bt_distance,
    bt_distance_full,
    lca_depth,
    digits_p_adic,
    digits_with_valuation,
)
from .metrics import padic_abs, padic_dist, pairwise_padic_dist, pairwise_padic_dist_vec
from .knn import PadicKNNClassifier, embed_float_array
from .hclust import ultrametric_dendrogram

__all__ = [
    # Core arithmetic
    "QpContext",
    "Qp",
    "PrecisionError",
    # Balls / ultrametric
    "QpBall",
    # Hensel lifting
    "hensel_lift_simple",
    # BT tree
    "BTRootedTree",
    "bt_distance",
    "bt_distance_full",
    "lca_depth",
    "digits_p_adic",
    "digits_with_valuation",
    # Metrics
    "padic_abs",
    "padic_dist",
    "pairwise_padic_dist",
    "pairwise_padic_dist_vec",
    # ML
    "PadicKNNClassifier",
    "embed_float_array",
    "ultrametric_dendrogram",
]
