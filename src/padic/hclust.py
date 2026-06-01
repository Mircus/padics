
from typing import List

import numpy as np

from .btree import digits_p_adic, lca_depth
from .field import Qp, QpContext


def ultrametric_dendrogram(ctx: QpContext, X: List[Qp]) -> np.ndarray:
    """Return a symmetric matrix of 'heights' = ctx.prec - LCA_depth.
    This is an ultrametric by construction (within truncated depth).
    """
    n = len(X)
    H = np.zeros((n,n), dtype=int)
    D = [digits_p_adic(x, ctx.prec) for x in X]
    for i in range(n):
        for j in range(i+1,n):
            d = ctx.prec - lca_depth(D[i], D[j])
            H[i,j] = H[j,i] = d
    return H
