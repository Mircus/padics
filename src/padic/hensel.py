
from typing import Callable, Tuple
from .field import QpContext, Qp

def hensel_lift_simple(ctx: QpContext,
                       fZ: Callable[[int], int],
                       fZprime: Callable[[int], int],
                       a0_mod_p: int,
                       target_prec: int) -> Qp:
    """Hensel lift a root of f in Z/pZ to Z/p^N Z, then to Qp at given precision.
    Assumes f(a0) ≡ 0 (mod p) and f'(a0) not ≡ 0 (mod p).
    Returns a Qp element a with digits determined up to target_prec.
    """
    p = ctx.p
    N = p ** target_prec
    a = a0_mod_p % p
    mod = p
    # Iteratively lift: a_{k+1} = a_k - f(a_k)/f'(a_k) mod p^{2^k}, but we can do linear steps doubling not required here.
    while mod < N:
        mod *= p
        # Solve f(a) + f'(a) * t ≡ 0 (mod mod), with t modulo p^k step
        fa = fZ(a) % mod
        fpa = fZprime(a) % mod
        # We need inverse of f'(a) modulo mod/power; but since mod increases by p each step, we can invert mod p then lift.
        inv = pow(fpa, -1, mod)
        t = (-fa * inv) % mod
        a = (a + t) % mod
    # Build Qp from the lifted residue.  Check zero BEFORE calling vp_int to
    # avoid computing p^{sentinel} (a billion-digit number).
    from .field import vp_int
    if a == 0:
        return Qp.zero(ctx)
    v = vp_int(a, p)
    u = (a // (p**v)) % (p**ctx.prec)
    return Qp(ctx, v=v, u_mod=u)
