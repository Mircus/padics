"""
padic.btree
~~~~~~~~~~~
Bruhat–Tits (BT) tree surrogate and LCA-based tree distances.

Design notes
------------
The *true* Bruhat–Tits tree for PGL₂(ℚ_p) has vertices corresponding to
homothety classes of ℤ_p-lattices in ℚ_p², with a rich PGL₂(ℚ_p) symmetry
group.  This module instead works with a simpler *truncated p-ary tree* whose
leaves are indexed by residues mod p^prec.

**Two digit representations are provided:**

1. ``digits_p_adic(x)`` — unit digits only (ignores valuation).
   Suitable for unit-level pattern matching.
   Used by ``bt_distance``, ``ultrametric_dendrogram``.
   Caveat: x and p·x have identical unit digits, so bt_distance(x, p·x) = 0
   even though d_p(x, p·x) = p^{−v_p(x)−1}.  This is a known limitation of
   the unit-only surrogate; document this to callers.

2. ``digits_with_valuation(x)`` — valuation encoded as leading zeros followed
   by unit digits.  bt_distance on these sequences is faithful to the p-adic
   metric whenever the valuation difference is within the truncation depth.

Use ``bt_distance_full`` for the valuation-aware variant.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from .field import QpContext, Qp


# ---------------------------------------------------------------------------
# Digit representations
# ---------------------------------------------------------------------------

def digits_p_adic(x: Qp, depth: Optional[int] = None) -> List[int]:
    """Return the unit digits of x (base p), ignoring valuation.

    .. note::
        This representation does **not** distinguish x from p·x (or p^k·x
        for any k).  bt_distance computed on these digits is a surrogate
        that captures unit-level structure but not the vertical (valuation)
        dimension of the tree.  Use ``digits_with_valuation`` for a faithful
        encoding.

    Parameters
    ----------
    x : Qp
    depth : int, optional
        Number of digits (default: x.ctx.prec).

    Returns
    -------
    list of int
    """
    return x.digits(depth)


def digits_with_valuation(x: Qp, total_depth: Optional[int] = None) -> List[int]:
    """Return a digit sequence that encodes valuation as leading zeros.

    The encoding is:
    - If x is zero: return ``total_depth`` zeros.
    - Otherwise: ``max(0, −v)`` leading zeros (for negative valuations the
      sequence starts at the most-negative unit), then the unit digits.

    This makes LCA depth in the digit tree correspond (approximately) to the
    p-adic metric, within the working precision.

    Parameters
    ----------
    x : Qp
    total_depth : int, optional
        Total number of digits returned (default: x.ctx.prec).

    Returns
    -------
    list of int
    """
    depth = x.ctx.prec if total_depth is None else total_depth
    if x.is_zero():
        return [0] * depth
    v = x.v
    unit_digits = x.digits(depth)
    if v >= 0:
        # Positive/zero valuation: prepend v implicit zeros then unit digits
        # (elements with large v look more "similar to zero")
        leading = [0] * min(v, depth)
        combined = leading + unit_digits
        return combined[:depth]
    else:
        # Negative valuation: prepend sentinel (p-1) for each unit of
        # negative valuation so that elements with v < 0 map to *different*
        # tree positions than those with v >= 0, keeping bt_distance_full > 0.
        # We use p-1 (the largest valid digit) instead of p so that all
        # digits remain in the valid range [0, p-1].  Note: p >= 2, so p-1 >= 1.
        sentinel = x.ctx.p - 1
        leading = [sentinel] * min(-v, depth)
        combined = leading + unit_digits
        return combined[:depth]


# ---------------------------------------------------------------------------
# LCA and distance
# ---------------------------------------------------------------------------

def lca_depth(d1: List[int], d2: List[int]) -> int:
    """Return the length of the longest common prefix of two digit sequences."""
    m = min(len(d1), len(d2))
    k = 0
    for i in range(m):
        if d1[i] == d2[i]:
            k += 1
        else:
            break
    return k


def bt_distance(ctx: QpContext, x: Qp, y: Qp) -> int:
    """Tree distance in the truncated p-ary tree (unit digits only).

    Computed as  2 · (ctx.prec − lca_depth(unit_digits(x), unit_digits(y))).

    .. warning::
        This is a *surrogate* distance that ignores valuation.
        bt_distance(x, p·x) = 0 for any nonzero x.
        For a valuation-aware variant use ``bt_distance_full``.
    """
    if x.is_zero() and y.is_zero():
        return 0
    d = lca_depth(x.digits(ctx.prec), y.digits(ctx.prec))
    return 2 * (ctx.prec - d)


def bt_distance_full(ctx: QpContext, x: Qp, y: Qp) -> int:
    """Tree distance using the valuation-aware digit encoding.

    Uses ``digits_with_valuation`` so that elements differing by a power of p
    are no longer collapsed to distance 0.
    """
    dx = digits_with_valuation(x, ctx.prec)
    dy = digits_with_valuation(y, ctx.prec)
    d = lca_depth(dx, dy)
    return 2 * (ctx.prec - d)


# ---------------------------------------------------------------------------
# BTRootedTree helper class
# ---------------------------------------------------------------------------

@dataclass
class BTRootedTree:
    """Implicit rooted p-ary tree up to depth ctx.prec.

    Provides both the unit-only surrogate distance and the
    valuation-aware variant.
    """
    ctx: QpContext

    def dist(self, x: Qp, y: Qp) -> int:
        """Unit-only BT surrogate distance (ignores valuation)."""
        return bt_distance(self.ctx, x, y)

    def dist_full(self, x: Qp, y: Qp) -> int:
        """Valuation-aware BT distance."""
        return bt_distance_full(self.ctx, x, y)
