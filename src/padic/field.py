"""
padic.field
~~~~~~~~~~~
Finite-precision p-adic arithmetic over Q_p.

Every nonzero element x ∈ Q_p is represented as  x = u · p^v
where v ∈ Z is the *valuation* v_p(x) and u is a *unit* (gcd(u,p)=1)
stored modulo p^prec.  Zero is the special sentinel (u_mod=0).

Precision semantics
-------------------
- All computations are truncated to ctx.prec significant p-adic digits.
- Operations that lose information at the low-valuation end return a result
  correct to the available precision without error.
- If an integer n cannot be faithfully encoded at the requested precision
  (i.e. all significant digits are shifted out), a PrecisionError is raised
  rather than silently returning zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


class PrecisionError(ArithmeticError):
    """Raised when an element cannot be represented at the current precision."""


def vp_int(n: int, p: int) -> int:
    """Return v_p(n) — the p-adic valuation of integer n.

    For n = 0 a large sentinel (10^9) is returned to represent +∞.
    Callers that handle zero specially should check ``n == 0`` before calling.
    """
    if n == 0:
        return 10**9
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def _is_prime(n: int) -> bool:
    """Return True iff n is a prime integer (n >= 2)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@dataclass(frozen=True)
class QpContext:
    """Immutable context for a p-adic computation: prime p and working precision.

    Parameters
    ----------
    p : int
        The prime base.
    prec : int
        Number of significant base-p digits retained in the unit part.
    """
    p: int
    prec: int

    def __post_init__(self) -> None:
        if self.p < 2:
            raise ValueError(f"p must be a prime ≥ 2, got {self.p}")
        if not _is_prime(self.p):
            raise ValueError(
                f"p must be prime, got {self.p} (composite numbers do not yield "
                "well-defined p-adic fields; p-adic arithmetic is only defined for prime p)."
            )
        if self.prec < 1:
            raise ValueError(f"prec must be ≥ 1, got {self.prec}")

    def modulus(self) -> int:
        """Return p^prec, the working modulus for unit residues."""
        return self.p ** self.prec


@dataclass
class Qp:
    """Finite-precision p-adic number: x = u_mod · p^v (u_mod ≢ 0 mod p, or zero).

    Attributes
    ----------
    ctx : QpContext
        The p and precision for this element.
    v : int
        Valuation v_p(x).  Sentinel 10^9 is used for the zero element.
    u_mod : int
        Unit residue modulo p^prec.  0 encodes the zero element.
    """
    ctx: QpContext
    v: int
    u_mod: int

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def zero(ctx: QpContext) -> "Qp":
        """Return the additive identity in Q_p."""
        return Qp(ctx, v=10**9, u_mod=0)

    @staticmethod
    def from_int(ctx: QpContext, n: int) -> "Qp":
        """Embed an integer n into Q_p at the given context.

        Raises PrecisionError if n ≠ 0 but all significant digits are lost
        at the current precision (i.e. v_p(n) ≥ prec).
        """
        if n == 0:
            return Qp.zero(ctx)
        p = ctx.p
        v = vp_int(abs(n), p)
        if v >= ctx.prec:
            raise PrecisionError(
                f"Cannot represent {n} in Q_{p} at prec={ctx.prec}: "
                f"v_p(|n|)={v} ≥ prec (all digits lost). Increase ctx.prec."
            )
        u = (abs(n) // (p ** v)) % ctx.modulus()
        if n < 0:
            u = (-u) % ctx.modulus()
        return Qp(ctx, v=v, u_mod=u)

    @staticmethod
    def from_rational(ctx: QpContext, num: int, den: int) -> "Qp":
        """Embed the rational num/den into Q_p.

        Parameters
        ----------
        ctx : QpContext
        num : int  numerator
        den : int  denominator (must be nonzero)

        Returns the p-adic representation of num/den truncated to ctx.prec
        significant digits.
        """
        if den == 0:
            raise ZeroDivisionError("denominator is zero")
        if num == 0:
            return Qp.zero(ctx)
        p = ctx.p
        v_num = vp_int(abs(num), p)
        v_den = vp_int(abs(den), p)
        v = v_num - v_den
        num_red = abs(num) // (p ** v_num)
        den_red = abs(den) // (p ** v_den)
        N = ctx.modulus()
        inv = pow(den_red, -1, N)
        u = (num_red * inv) % N
        if (num < 0) ^ (den < 0):
            u = (-u) % N
        # Normalize: strip trailing p-factors from u (shouldn't occur unless
        # the arithmetic above introduced them, but guard defensively).
        while u % p == 0 and u != 0:
            u //= p
            v += 1
        return Qp(ctx, v=v, u_mod=u)

    # ------------------------------------------------------------------
    # Core predicates
    # ------------------------------------------------------------------

    def is_zero(self) -> bool:
        """Return True iff this element is the zero of Q_p."""
        return self.u_mod == 0

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Two Qp elements are equal iff they share context, valuation, and unit residue.

        Zero elements are equal regardless of sentinel valuation.
        """
        if not isinstance(other, Qp):
            return NotImplemented
        if self.ctx != other.ctx:
            return False
        if self.is_zero() and other.is_zero():
            return True
        if self.is_zero() or other.is_zero():
            return False
        return self.v == other.v and self.u_mod == other.u_mod

    def __hash__(self) -> int:
        if self.is_zero():
            return hash((self.ctx, "zero"))
        return hash((self.ctx, self.v, self.u_mod))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_ctx(self, other: "Qp", op: str = "op") -> None:
        if self.ctx != other.ctx:
            raise ValueError(
                f"Context mismatch in {op}: "
                f"left={self.ctx}, right={other.ctx}"
            )

    def add(self, other: "Qp") -> "Qp":
        """Return self + other in Q_p (truncated to ctx.prec digits)."""
        self._check_ctx(other, "add")
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        p = self.ctx.p
        prec = self.ctx.prec
        # Align both operands to the smaller valuation vmin:
        #   x = (u1 · p^(v1−vmin)) · p^vmin
        #   y = (u2 · p^(v2−vmin)) · p^vmin
        #   x + y = (shifted_sum) · p^vmin
        vmin = min(self.v, other.v)
        M = p ** prec
        a = (self.u_mod * pow(p, self.v - vmin, M)) % M
        b = (other.u_mod * pow(p, other.v - vmin, M)) % M
        s = (a + b) % M
        if s == 0:
            return Qp.zero(self.ctx)
        vs = vp_int(s, p)
        u = (s // (p ** vs)) % M
        return Qp(self.ctx, v=vmin + vs, u_mod=u)

    def neg(self) -> "Qp":
        """Return the additive inverse -self."""
        if self.is_zero():
            return self
        return Qp(self.ctx, v=self.v, u_mod=(-self.u_mod) % self.ctx.modulus())

    def sub(self, other: "Qp") -> "Qp":
        """Return self − other."""
        return self.add(other.neg())

    def mul(self, other: "Qp") -> "Qp":
        """Return self × other."""
        self._check_ctx(other, "mul")
        if self.is_zero() or other.is_zero():
            return Qp.zero(self.ctx)
        v = self.v + other.v
        u = (self.u_mod * other.u_mod) % self.ctx.modulus()
        # Guard: strip accidental p-factors (product of units is a unit, but
        # modular truncation can produce them in edge cases).
        while u % self.ctx.p == 0 and u != 0:
            u //= self.ctx.p
            v += 1
        return Qp(self.ctx, v=v, u_mod=u)

    def inv(self) -> "Qp":
        """Return the multiplicative inverse self^{-1}.

        Raises ZeroDivisionError for the zero element.
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot invert the zero element of Q_p")
        N = self.ctx.modulus()
        uinv = pow(self.u_mod % N, -1, N)
        return Qp(self.ctx, v=-self.v, u_mod=uinv)

    def div(self, other: "Qp") -> "Qp":
        """Return self / other."""
        return self.mul(other.inv())

    # ------------------------------------------------------------------
    # Metric / valuation
    # ------------------------------------------------------------------

    def val(self) -> int:
        """Return v_p(self); returns the sentinel 10^9 for the zero element."""
        return self.v if not self.is_zero() else 10**9

    def abs(self) -> float:
        """Return |self|_p = p^{−v_p(self)}.  |0|_p = 0."""
        if self.is_zero():
            return 0.0
        return float(self.ctx.p) ** (-self.v)

    # ------------------------------------------------------------------
    # Digit representation
    # ------------------------------------------------------------------

    def digits(self, depth: Optional[int] = None) -> list:
        """Return the base-p digits of the unit part u_mod.

        Parameters
        ----------
        depth : int, optional
            Number of digits to return (default: ctx.prec).

        Returns
        -------
        list of int
            Digits [d_0, d_1, …] where u_mod = d_0 + d_1·p + d_2·p² + …
            Note: this ignores valuation — see ``digits_with_valuation`` in
            padic.btree for a valuation-aware alternative.
        """
        N = self.ctx.modulus()
        u = self.u_mod % N
        k = self.ctx.prec if depth is None else min(depth, self.ctx.prec)
        out = []
        for _ in range(k):
            out.append(u % self.ctx.p)
            u //= self.ctx.p
        return out

    def canonical_mod(self, extra_prec: int = 0) -> Tuple[int, int]:
        """Return the canonical (v, u_mod) pair for this element.

        The element is  u_mod · p^v  where  0 < u_mod < p^prec  and
        p ∤ u_mod.  For zero the sentinel (10^9, 0) is returned.
        extra_prec is accepted for backward compatibility but has no effect.
        """
        if self.is_zero():
            return (10**9, 0)
        return (self.v, self.u_mod)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_zero():
            return f"Qp(0; p={self.ctx.p}, prec={self.ctx.prec})"
        return (
            f"Qp(u≡{self.u_mod} (mod {self.ctx.p}^{self.ctx.prec}) "
            f"* {self.ctx.p}^{self.v}; p={self.ctx.p})"
        )
