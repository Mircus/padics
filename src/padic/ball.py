
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from .field import Qp, QpContext, vp_int

@dataclass(frozen=True)
class QpBall:
    """Closed ball B(a, p^{-n}) in Qp at finite precision."""
    center: Qp
    n: int  # radius exponent (radius = p^{-n})

    def contains(self, x: Qp) -> bool:
        if x.ctx != self.center.ctx:
            return False
        # d_p(x, center) <= p^{-n}  <=>  v_p(x-center) >= n
        v = x.sub(self.center).val()
        return v >= self.n

    def intersect(self, other: "QpBall") -> "QpBall|None":
        if self.center.ctx != other.center.ctx:
            return None
        # Non-empty iff centers are close enough
        if self.center.sub(other.center).val() >= min(self.n, other.n):
            # intersection is the smaller (larger n) ball around one of the centers at that radius
            return self if self.n >= other.n else other
        return None

    def refine(self) -> "list[QpBall]":
        """Split the ball into p disjoint sub-balls of radius p^{-(n+1)}.

        The p children are centered at  center + d * p^n  for d = 0, …, p-1.
        Since 0 < d < p implies gcd(d, p) = 1, each such shift has valuation n
        (not center.v + n), so we build the shift directly as Qp(v=n, u_mod=d).
        """
        ctx = self.center.ctx
        p = ctx.p
        subs = []
        for d in range(p):
            if d == 0:
                new_center = self.center
            else:
                # d in {1,…,p-1}: gcd(d,p)=1 so v_p(d*p^n) = n exactly
                shift = Qp(ctx, v=self.n, u_mod=d)
                new_center = self.center.add(shift)
            subs.append(QpBall(new_center, self.n + 1))
        return subs

    def __repr__(self) -> str:
        return f"B({self.center}, p^(-{self.n}))"
