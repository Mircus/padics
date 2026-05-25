
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .field import QpContext, Qp

def digits_p_adic(x: Qp, depth: Optional[int]=None) -> List[int]:
    """Return unit digits (base p) of x, ignoring valuation shift for tree alignment.
    In BT, balls are based on digits of coordinates; here we expose unit digits.
    """
    return x.digits(depth)

def lca_depth(d1: List[int], d2: List[int]) -> int:
    """Depth of longest common prefix of two digit sequences."""
    m = min(len(d1), len(d2))
    k = 0
    for i in range(m):
        if d1[i] == d2[i]:
            k += 1
        else:
            break
    return k

def bt_distance(ctx: QpContext, x: Qp, y: Qp) -> int:
    """Tree distance in the truncated p-ary tree (based on unit digits only)."""
    if x.is_zero() and y.is_zero(): return 0
    d = lca_depth(x.digits(ctx.prec), y.digits(ctx.prec))
    # Distance as 2*(depth - lca_depth). Here depth = ctx.prec
    return 2*(ctx.prec - d)

@dataclass
class BTRootedTree:
    """Implicit rooted p-ary tree up to depth 'depth' (we don't materialize nodes)."""
    ctx: QpContext

    def dist(self, x: Qp, y: Qp) -> int:
        return bt_distance(self.ctx, x, y)
