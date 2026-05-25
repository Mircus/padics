
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

def vp_int(n: int, p: int) -> int:
    """p-adic valuation v_p(n) for integer n (v_p(0) = +inf -> return a big sentinel)."""
    if n == 0:
        return 10**9
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v

@dataclass(frozen=True)
class QpContext:
    p: int
    prec: int  # number of base-p digits kept for the unit (mod p^prec)

    def modulus(self) -> int:
        return self.p ** self.prec

@dataclass
class Qp:
    """Finite-precision p-adic element: x = u * p^v with u mod p^N and u not divisible by p (unless x=0)."""
    ctx: QpContext
    v: int                    # valuation (can be negative)
    u_mod: int                # residue of unit modulo p^N; 0 encodes x=0

    @staticmethod
    def zero(ctx: QpContext) -> "Qp":
        return Qp(ctx, v=10**9, u_mod=0)

    @staticmethod
    def from_int(ctx: QpContext, n: int) -> "Qp":
        if n == 0:
            return Qp.zero(ctx)
        v = vp_int(abs(n), ctx.p)
        u = (abs(n) // (ctx.p**v)) % ctx.modulus()
        if u % ctx.p == 0:  # normalize: ensure unit not divisible by p
            # this should not happen unless precision is too small
            # fallback: bump valuation, reduce u
            while u % ctx.p == 0 and u != 0:
                v += 1
                u //= ctx.p
        if n < 0:
            # encode sign in u via modulus wrap
            u = (-u) % ctx.modulus()
        return Qp(ctx, v=v, u_mod=u)

    @staticmethod
    def from_rational(ctx: QpContext, num: int, den: int) -> "Qp":
        if den == 0:
            raise ZeroDivisionError("denominator zero")
        if num == 0:
            return Qp.zero(ctx)
        # factor p-adic valuations
        v_num = vp_int(abs(num), ctx.p)
        v_den = vp_int(abs(den), ctx.p)
        v = v_num - v_den
        num_red = abs(num) // (ctx.p**v_num)
        den_red = abs(den) // (ctx.p**v_den)
        # compute u = num_red * den_red^{-1} mod p^N
        N = ctx.modulus()
        # invert den_red modulo p^N using extended Euclid (since gcd(den_red,p)=1)
        inv = pow(den_red, -1, N)
        u = (num_red * inv) % N
        if (num < 0) ^ (den < 0):
            u = (-u) % N
        # normalize: ensure unit not divisible by p
        while u % ctx.p == 0 and u != 0:
            u //= ctx.p
            v += 1
        return Qp(ctx, v=v, u_mod=u)

    def is_zero(self) -> bool:
        return self.u_mod == 0

    def canonical_mod(self, extra_prec: int = 0) -> Tuple[int,int]:
        """Return (v, u_mod) — the normalized (valuation, unit) representation.

        The element is u_mod * p^v.  extra_prec is accepted for backward
        compatibility but has no effect (the representation is already exact
        to ctx.prec significant digits).
        """
        if self.is_zero():
            return (10**9, 0)
        return (self.v, self.u_mod)

    def add(self, other: "Qp") -> "Qp":
        if self.ctx != other.ctx:
            raise ValueError("Context mismatch")
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        p = self.ctx.p
        prec = self.ctx.prec
        # Align both operands to the lower valuation vmin, then add unit parts.
        # x = u1 * p^v1 = (u1 * p^(v1-vmin)) * p^vmin
        # y = u2 * p^v2 = (u2 * p^(v2-vmin)) * p^vmin
        # x+y = (u1*p^(v1-vmin) + u2*p^(v2-vmin)) * p^vmin
        vmin = min(self.v, other.v)
        M = p ** prec
        a = (self.u_mod * pow(p, self.v - vmin, M)) % M
        b = (other.u_mod * pow(p, other.v - vmin, M)) % M
        s = (a + b) % M
        if s == 0:
            return Qp.zero(self.ctx)
        # extract valuation of the unit-level sum, then offset by vmin
        vs = vp_int(s, p)
        u = (s // (p**vs)) % M
        return Qp(self.ctx, v=vmin + vs, u_mod=u)

    def neg(self) -> "Qp":
        if self.is_zero():
            return self
        return Qp(self.ctx, v=self.v, u_mod=(-self.u_mod) % self.ctx.modulus())

    def sub(self, other: "Qp") -> "Qp":
        return self.add(other.neg())

    def mul(self, other: "Qp") -> "Qp":
        if self.ctx != other.ctx:
            raise ValueError("Context mismatch")
        if self.is_zero() or other.is_zero():
            return Qp.zero(self.ctx)
        v = self.v + other.v
        u = (self.u_mod * other.u_mod) % self.ctx.modulus()
        # normalize if divisible by p (shouldn't happen for units, but guard)
        while u % self.ctx.p == 0 and u != 0:
            u //= self.ctx.p
            v += 1
        return Qp(self.ctx, v=v, u_mod=u)

    def inv(self) -> "Qp":
        if self.is_zero():
            raise ZeroDivisionError("division by zero")
        N = self.ctx.modulus()
        uinv = pow(self.u_mod % N, -1, N)
        v = -self.v
        return Qp(self.ctx, v=v, u_mod=uinv)

    def div(self, other: "Qp") -> "Qp":
        return self.mul(other.inv())

    def val(self) -> int:
        return self.v if not self.is_zero() else 10**9

    def abs(self) -> float:
        # |x|_p = p^{-v_p(x)}, and |0|_p = 0
        if self.is_zero():
            return 0.0
        return (self.ctx.p) ** (-self.v)

    def digits(self, depth: Optional[int]=None) -> list[int]:
        """Return base-p digits of the unit u_mod up to 'depth' (<= prec)."""
        N = self.ctx.modulus()
        u = self.u_mod % N
        k = self.ctx.prec if depth is None else min(depth, self.ctx.prec)
        out = []
        for _ in range(k):
            out.append(u % self.ctx.p)
            u //= self.ctx.p
        return out

    def __repr__(self) -> str:
        if self.is_zero(): return f"Qp(0; p={self.ctx.p}, prec={self.ctx.prec})"
        return f"Qp(u≡{self.u_mod} (mod p^{self.ctx.prec}) * p^{self.v}; p={self.ctx.p})"
