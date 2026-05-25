from padic import QpContext, Qp, QpBall, hensel_lift_simple

# ---------------------------------------------------------------------------
# Qp arithmetic
# ---------------------------------------------------------------------------

def test_add_mul_roundtrip():
    ctx = QpContext(3, prec=8)
    x = Qp.from_rational(ctx, 7, 12)   # v = -1 (negative valuation)
    y = Qp.from_int(ctx, 10)
    z = x.add(y).sub(y)
    assert z.u_mod == x.u_mod and z.v == x.v, f"Expected v={x.v}, got v={z.v}"

def test_inv():
    ctx = QpContext(5, prec=8)
    x = Qp.from_rational(ctx, 7, 3)
    inv = x.inv()
    one = x.mul(inv)
    assert one.abs() > 0 and one.val() == 0  # unit (valuation 0)

def test_add_negative_valuation():
    """x = 1/p, y = p  →  x+y = 1/p + p = (1 + p^2)/p; val = -1."""
    ctx = QpContext(5, prec=6)
    x = Qp.from_rational(ctx, 1, 5)   # 1/5, v=-1, u=1
    y = Qp.from_int(ctx, 5)           # 5,   v=1,  u=1
    s = x.add(y)
    assert s.val() == -1, f"Expected val=-1, got {s.val()}"
    # 1/5 + 5 = 26/5; unit part = 26 = 1 + 5^2 (mod 5^6)
    expected_u = (1 + 5**2) % (5**6)
    assert s.u_mod == expected_u, f"Expected u_mod={expected_u}, got {s.u_mod}"

def test_add_both_negative_valuation():
    """2/p + 3/p = 5/p = 1; val = 0."""
    ctx = QpContext(5, prec=6)
    x = Qp.from_rational(ctx, 2, 5)  # 2/5
    y = Qp.from_rational(ctx, 3, 5)  # 3/5
    s = x.add(y)
    assert s.val() == 0 and s.u_mod == 1, f"Expected (v=0,u=1), got (v={s.val()},u={s.u_mod})"

def test_add_cancellation():
    """x + (-x) = 0."""
    ctx = QpContext(3, prec=8)
    x = Qp.from_rational(ctx, 5, 9)   # v=-2
    assert x.add(x.neg()).is_zero()

def test_from_rational_zero():
    """from_rational(ctx, 0, n) must return zero without hanging."""
    ctx = QpContext(7, prec=4)
    z = Qp.from_rational(ctx, 0, 3)
    assert z.is_zero()

def test_mul_negative_valuation():
    """(1/p) * (1/p) = 1/p^2."""
    ctx = QpContext(5, prec=6)
    x = Qp.from_rational(ctx, 1, 5)
    r = x.mul(x)
    assert r.val() == -2 and r.u_mod == 1

# ---------------------------------------------------------------------------
# QpBall.refine
# ---------------------------------------------------------------------------

def test_refine_covers_parent():
    """Every point in the parent ball must be in exactly one child."""
    ctx = QpContext(5, prec=6)
    center = Qp.from_int(ctx, 7)
    parent = QpBall(center, 1)
    children = parent.refine()
    assert len(children) == 5
    # sample 25 integers and check each is in exactly one child
    for k in range(25):
        pt = Qp.from_int(ctx, k)
        if parent.contains(pt):
            hits = [c.contains(pt) for c in children]
            assert sum(hits) == 1, f"pt={k} in {sum(hits)} children (expected 1)"

def test_refine_shifts_correct():
    """Child d should contain center + d*p^n."""
    ctx = QpContext(5, prec=6)
    center = Qp.from_rational(ctx, 1, 5)   # 1/5, v=-1
    n = 1
    ball = QpBall(center, n)
    children = ball.refine()
    p = ctx.p
    for d in range(p):
        if d == 0:
            shift_elem = Qp.zero(ctx)
        else:
            shift_elem = Qp(ctx, v=n, u_mod=d)
        expected_center = center.add(shift_elem)
        child = children[d]
        assert child.contains(expected_center), (
            f"Child {d} does not contain its own expected center"
        )

def test_refine_zero_center():
    """Refine on B(0, 1): 5 children with distinct residues 0..4 mod p."""
    ctx = QpContext(5, prec=6)
    zero = Qp.zero(ctx)
    ball = QpBall(zero, 0)
    children = ball.refine()
    assert len(children) == 5
    # child d should contain Qp(d) for d=0,1,2,3,4
    for d in range(5):
        pt = Qp.from_int(ctx, d)
        assert children[d].contains(pt), f"Child {d} should contain integer {d}"
        # and NOT contain any other digit
        for d2 in range(5):
            if d2 != d:
                pt2 = Qp.from_int(ctx, d2)
                assert not children[d].contains(pt2), (
                    f"Child {d} should NOT contain integer {d2}"
                )

def test_repr_no_crash():
    """QpBall.__repr__ must not raise."""
    ctx = QpContext(3, prec=4)
    b = QpBall(Qp.from_int(ctx, 1), 2)
    assert "p^(-2)" in repr(b)

# ---------------------------------------------------------------------------
# Hensel lift
# ---------------------------------------------------------------------------

def test_hensel_lift_sqrt():
    """Lift sqrt(2) in Z_7: 2 is a QR mod 7 (3^2 = 9 ≡ 2), so root exists."""
    ctx = QpContext(7, prec=6)
    # f(x) = x^2 - 2, f'(x) = 2x
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a*a - 2,
        fZprime=lambda a: 2*a,
        a0_mod_p=3,       # 3^2 = 9 ≡ 2 (mod 7)
        target_prec=6,
    )
    # verify root^2 ≡ 2 (mod 7^6)
    check = root.mul(root).sub(Qp.from_int(ctx, 2))
    assert check.is_zero() or check.val() >= ctx.prec, (
        f"Hensel root squared ≠ 2: val={check.val()}"
    )
