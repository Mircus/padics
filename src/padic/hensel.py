"""
padic.hensel
~~~~~~~~~~~~
Hensel lifting for roots of integer polynomials in Z_p.

Algorithm
---------
``hensel_lift_simple`` implements *linear* Hensel lifting (the modulus grows
by one factor of p per step).  Given a root  a₀ ≡ 0 (mod p)  of  f, the
lift satisfies:

    a_{k+1} ≡ a_k − f(a_k) / f'(a_k)  (mod p^{k+1})

This gives precision growing as  p, p², p³, …  (linear in the number of
steps, **not** quadratic).  Quadratic (Newton–Hensel) lifting — where the
modulus doubles each step — is on the roadmap.

Precision contract
------------------
- ``target_prec`` controls the lifting modulus p^target_prec.
- The returned Qp element is built with the *caller's* ctx (ctx.prec digits).
- If target_prec > ctx.prec the lifted digits are faithfully computed but
  then truncated to ctx.prec on return (information is intentionally discarded).
- If target_prec < ctx.prec the returned element has only target_prec
  verified digits; the remaining ctx.prec − target_prec digits are
  implementation-dependent residues.
- Recommended practice: set target_prec == ctx.prec.
"""

from __future__ import annotations
from typing import Callable
from .field import QpContext, Qp, vp_int


def hensel_lift_simple(
    ctx: QpContext,
    fZ: Callable[[int], int],
    fZprime: Callable[[int], int],
    a0_mod_p: int,
    target_prec: int,
) -> Qp:
    """Lift a root of f ∈ Z[X] from Z/pZ to Z/p^target_prec Z, then to Qp.

    Parameters
    ----------
    ctx : QpContext
        Context for the returned Qp element.
    fZ : callable int → int
        The polynomial evaluated over Z (arbitrary precision).
    fZprime : callable int → int
        The formal derivative of fZ evaluated over Z.
    a0_mod_p : int
        A root of f modulo p (i.e. fZ(a0_mod_p) ≡ 0 mod p).
    target_prec : int
        Number of p-adic digits to lift to.

    Returns
    -------
    Qp
        The lifted root as a Qp element at ctx.prec precision.

    Raises
    ------
    ValueError
        If the precondition f(a0) ≡ 0 (mod p) fails.
    ArithmeticError
        If f'(a) becomes non-invertible mod the current modulus during lifting
        (Hensel's lemma conditions violated).
    """
    p = ctx.p

    # Validate Hensel preconditions at mod p
    a0 = a0_mod_p % p
    if fZ(a0) % p != 0:
        raise ValueError(
            f"Hensel precondition failed: f({a0}) ≡ {fZ(a0) % p} ≢ 0 (mod {p}). "
            f"a0_mod_p must be a root of f modulo p."
        )
    if fZprime(a0) % p == 0:
        raise ValueError(
            f"Hensel precondition failed: f'({a0}) ≡ 0 (mod {p}). "
            f"The derivative must be invertible mod p (simple root required)."
        )

    N = p ** target_prec
    a = a0
    mod = p

    while mod < N:
        mod *= p
        fa = fZ(a) % mod
        fpa = fZprime(a) % mod
        try:
            inv = pow(fpa, -1, mod)
        except ValueError:
            raise ArithmeticError(
                f"f'({a}) = {fpa} is not invertible mod {mod} during Hensel "
                f"lifting (step mod={mod}).  Check that f has a simple root."
            ) from None
        t = (-fa * inv) % mod
        a = (a + t) % mod

    # Build Qp from the lifted residue.
    if a == 0:
        return Qp.zero(ctx)
    v = vp_int(a, p)
    u = (a // (p ** v)) % (p ** ctx.prec)
    return Qp(ctx, v=v, u_mod=u)
