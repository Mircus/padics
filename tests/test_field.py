"""
Comprehensive test suite for padic-ds.

Covers:
- Qp arithmetic (add, sub, mul, inv, div) with positive and negative valuations
- Ultrametric inequality
- QpBall (contains, intersect, refine)
- Hensel lifting
- BT distances (unit-only and valuation-aware)
- Metrics (pairwise distance)
- PadicKNNClassifier
- PrecisionError
- Qp equality / hashing
"""

import pytest
import numpy as np
from padic import (
    QpContext, Qp, PrecisionError, QpBall,
    hensel_lift_simple,
    bt_distance, bt_distance_full, lca_depth, digits_with_valuation,
    padic_dist, pairwise_padic_dist, pairwise_padic_dist_vec,
    PadicKNNClassifier,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def ctx5():
    return QpContext(5, prec=8)

@pytest.fixture
def ctx3():
    return QpContext(3, prec=8)

@pytest.fixture
def ctx7():
    return QpContext(7, prec=6)


# ===========================================================================
# 1. Qp construction and predicates
# ===========================================================================

def test_zero_is_zero(ctx5):
    z = Qp.zero(ctx5)
    assert z.is_zero()
    assert z.abs() == 0.0
    assert z.val() == 10**9

def test_from_int_positive(ctx5):
    x = Qp.from_int(ctx5, 25)   # 25 = 5^2 * 1
    assert x.v == 2
    assert x.u_mod == 1

def test_from_int_negative(ctx5):
    x = Qp.from_int(ctx5, -10)  # -10 = 5 * (-2); unit = -2 mod 5^8
    assert x.v == 1
    assert x.u_mod == (-2) % (5**8)

def test_from_int_precision_error(ctx5):
    # 5^8 has valuation 8 == prec; should raise
    with pytest.raises(PrecisionError):
        Qp.from_int(ctx5, 5**8)

def test_from_rational_one_over_p(ctx5):
    x = Qp.from_rational(ctx5, 1, 5)
    assert x.v == -1
    assert x.u_mod == 1

def test_from_rational_negative_den(ctx5):
    x = Qp.from_rational(ctx5, 2, -5)  # -2/5
    y = Qp.from_rational(ctx5, 2, 5)
    # -2/5 should be negation of 2/5
    assert x.add(y).is_zero()

def test_from_rational_zero(ctx5):
    z = Qp.from_rational(ctx5, 0, 7)
    assert z.is_zero()

def test_from_rational_zero_den(ctx5):
    with pytest.raises(ZeroDivisionError):
        Qp.from_rational(ctx5, 1, 0)


# ===========================================================================
# 2. Equality and hashing
# ===========================================================================

def test_equality_same(ctx5):
    x = Qp.from_int(ctx5, 7)
    y = Qp.from_int(ctx5, 7)
    assert x == y

def test_equality_zero(ctx5):
    assert Qp.zero(ctx5) == Qp.zero(ctx5)

def test_equality_distinct(ctx5):
    assert Qp.from_int(ctx5, 3) != Qp.from_int(ctx5, 4)

def test_equality_context_mismatch():
    x = Qp.from_int(QpContext(5, 4), 3)
    y = Qp.from_int(QpContext(7, 4), 3)
    assert x != y

def test_hash_equal_elements(ctx5):
    x = Qp.from_int(ctx5, 7)
    y = Qp.from_int(ctx5, 7)
    assert hash(x) == hash(y)

def test_hash_in_set(ctx5):
    x = Qp.from_int(ctx5, 7)
    y = Qp.from_int(ctx5, 7)
    z = Qp.from_int(ctx5, 11)
    s = {x, z}
    assert y in s  # y == x, same hash


# ===========================================================================
# 3. Arithmetic — basic
# ===========================================================================

def test_add_commutativity(ctx3):
    x = Qp.from_rational(ctx3, 7, 12)
    y = Qp.from_int(ctx3, 10)
    assert x.add(y) == y.add(x)

def test_add_associativity(ctx5):
    x = Qp.from_int(ctx5, 2)
    y = Qp.from_int(ctx5, 3)
    z = Qp.from_int(ctx5, 7)
    assert x.add(y).add(z) == x.add(y.add(z))

def test_add_negative_valuation(ctx5):
    """1/p + p = (1 + p^2)/p; val = -1."""
    x = Qp.from_rational(ctx5, 1, 5)
    y = Qp.from_int(ctx5, 5)
    s = x.add(y)
    assert s.val() == -1
    expected_u = (1 + 5**2) % (5**8)
    assert s.u_mod == expected_u

def test_add_both_negative_valuation(ctx5):
    """2/p + 3/p = 5/p = 1; val = 0."""
    x = Qp.from_rational(ctx5, 2, 5)
    y = Qp.from_rational(ctx5, 3, 5)
    s = x.add(y)
    assert s.val() == 0 and s.u_mod == 1

def test_add_cancellation(ctx3):
    x = Qp.from_rational(ctx3, 5, 9)
    assert x.add(x.neg()).is_zero()

def test_sub_roundtrip(ctx3):
    x = Qp.from_rational(ctx3, 7, 12)
    y = Qp.from_int(ctx3, 10)
    z = x.add(y).sub(y)
    assert z == x

def test_mul_commutativity(ctx5):
    x = Qp.from_int(ctx5, 3)
    y = Qp.from_int(ctx5, 7)
    assert x.mul(y) == y.mul(x)

def test_mul_valuation_additivity(ctx5):
    """v_p(x*y) = v_p(x) + v_p(y)."""
    x = Qp.from_rational(ctx5, 1, 5)   # v=-1
    y = Qp.from_int(ctx5, 25)          # v=2
    assert x.mul(y).val() == -1 + 2

def test_mul_negative_valuation(ctx5):
    """(1/p)^2 = 1/p^2."""
    x = Qp.from_rational(ctx5, 1, 5)
    r = x.mul(x)
    assert r.val() == -2 and r.u_mod == 1

def test_inv_roundtrip(ctx5):
    x = Qp.from_rational(ctx5, 7, 3)
    one = x.mul(x.inv())
    assert one.val() == 0 and one.u_mod == 1

def test_div(ctx5):
    x = Qp.from_int(ctx5, 6)
    y = Qp.from_int(ctx5, 2)
    three = x.div(y)
    assert three == Qp.from_int(ctx5, 3)

def test_inv_zero_raises(ctx5):
    with pytest.raises(ZeroDivisionError):
        Qp.zero(ctx5).inv()

def test_context_mismatch_add():
    x = Qp.from_int(QpContext(5, 6), 1)
    y = Qp.from_int(QpContext(7, 6), 1)
    with pytest.raises(ValueError):
        x.add(y)

def test_context_mismatch_mul():
    x = Qp.from_int(QpContext(5, 6), 1)
    y = Qp.from_int(QpContext(7, 6), 1)
    with pytest.raises(ValueError):
        x.mul(y)


# ===========================================================================
# 4. Ultrametric inequality  d(x,z) ≤ max(d(x,y), d(y,z))
# ===========================================================================

@pytest.mark.parametrize("a,b,c", [
    (1, 2, 3), (5, 10, 15), (1, 5, 25), (7, 14, 21), (1, 7, 49)
])
def test_ultrametric_inequality(ctx5, a, b, c):
    x = Qp.from_int(ctx5, a)
    y = Qp.from_int(ctx5, b)
    z = Qp.from_int(ctx5, c)
    dxz = padic_dist(x, z)
    dxy = padic_dist(x, y)
    dyz = padic_dist(y, z)
    assert dxz <= max(dxy, dyz) + 1e-12, \
        f"Ultrametric violated: d({a},{c})={dxz} > max(d({a},{b})={dxy}, d({b},{c})={dyz})"


# ===========================================================================
# 5. QpBall
# ===========================================================================

def test_ball_contains_center(ctx5):
    center = Qp.from_int(ctx5, 7)
    ball = QpBall(center, 3)
    assert ball.contains(center)

def test_ball_refine_covers_parent(ctx5):
    center = Qp.from_int(ctx5, 7)
    parent = QpBall(center, 1)
    children = parent.refine()
    assert len(children) == 5
    for k in range(25):
        try:
            pt = Qp.from_int(ctx5, k)
        except PrecisionError:
            continue
        if parent.contains(pt):
            hits = [c.contains(pt) for c in children]
            assert sum(hits) == 1, f"pt={k} in {sum(hits)} children"

def test_ball_refine_shifts_correct(ctx5):
    center = Qp.from_rational(ctx5, 1, 5)
    n = 1
    ball = QpBall(center, n)
    children = ball.refine()
    p = ctx5.p
    for d in range(p):
        if d == 0:
            shift_elem = Qp.zero(ctx5)
        else:
            shift_elem = Qp(ctx5, v=n, u_mod=d)
        expected_center = center.add(shift_elem)
        assert children[d].contains(expected_center)

def test_ball_refine_zero_center(ctx5):
    zero = Qp.zero(ctx5)
    ball = QpBall(zero, 0)
    children = ball.refine()
    assert len(children) == 5
    for d in range(5):
        pt = Qp.from_int(ctx5, d)
        assert children[d].contains(pt)
        for d2 in range(5):
            if d2 != d:
                pt2 = Qp.from_int(ctx5, d2)
                assert not children[d].contains(pt2)

def test_ball_repr_no_crash(ctx3):
    b = QpBall(Qp.from_int(ctx3, 1), 2)
    r = repr(b)
    assert "p^(-2)" in r

def test_ball_intersect(ctx5):
    x = Qp.from_int(ctx5, 1)
    y = Qp.from_int(ctx5, 2)
    big_ball = QpBall(x, 0)     # radius 1 (contains all units)
    small_ball = QpBall(x, 3)   # radius 5^{-3}
    assert big_ball.intersect(small_ball) is not None

def test_ball_intersect_disjoint(ctx5):
    x = Qp.from_int(ctx5, 1)
    y = Qp.from_int(ctx5, 2)
    b1 = QpBall(x, 1)
    b2 = QpBall(y, 1)
    # 1 and 2 differ mod 5, so balls of radius 5^{-1} are disjoint
    assert b1.intersect(b2) is None


# ===========================================================================
# 6. Hensel lifting
# ===========================================================================

def test_hensel_lift_sqrt7(ctx7):
    """sqrt(2) in Z_7: 3^2 = 9 ≡ 2 (mod 7)."""
    root = hensel_lift_simple(
        ctx7,
        fZ=lambda a: a * a - 2,
        fZprime=lambda a: 2 * a,
        a0_mod_p=3,
        target_prec=6,
    )
    check = root.mul(root).sub(Qp.from_int(ctx7, 2))
    assert check.is_zero() or check.val() >= ctx7.prec

def test_hensel_lift_sqrt5():
    """sqrt(6) in Z_5: a0=1 since 1-6=-5≡0 mod 5, f'(1)=2 invertible."""
    ctx = QpContext(5, prec=6)
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a * a - 6,
        fZprime=lambda a: 2 * a,
        a0_mod_p=1,
        target_prec=6,
    )
    check = root.mul(root).sub(Qp.from_int(ctx, 6))
    assert check.is_zero() or check.val() >= ctx.prec

def test_hensel_bad_initial():
    """Should raise when f(a0) ≢ 0 mod p."""
    ctx = QpContext(5, prec=6)
    with pytest.raises(ValueError, match="precondition"):
        hensel_lift_simple(
            ctx,
            fZ=lambda a: a * a - 2,
            fZprime=lambda a: 2 * a,
            a0_mod_p=0,   # 0^2 - 2 = -2 ≢ 0 mod 5
            target_prec=6,
        )

def test_hensel_bad_derivative():
    """Should raise when f'(a0) ≡ 0 mod p (multiple root)."""
    ctx = QpContext(5, prec=6)
    with pytest.raises(ValueError, match="precondition"):
        hensel_lift_simple(
            ctx,
            fZ=lambda a: a * a,           # root = 0, double root
            fZprime=lambda a: 2 * a,
            a0_mod_p=0,
            target_prec=6,
        )


# ===========================================================================
# 7. BT distances
# ===========================================================================

def test_bt_distance_same(ctx5):
    x = Qp.from_int(ctx5, 3)
    assert bt_distance(ctx5, x, x) == 0

def test_bt_distance_zero_zero(ctx5):
    assert bt_distance(ctx5, Qp.zero(ctx5), Qp.zero(ctx5)) == 0

def test_bt_distance_unit_only_limitation(ctx5):
    """bt_distance ignores valuation: d_BT(x, p*x) == 0 by design."""
    x = Qp.from_int(ctx5, 3)
    px = Qp.from_int(ctx5, 15)  # 3 * 5
    # unit digits of 3 == unit digits of 15 (both have unit part 3)
    assert bt_distance(ctx5, x, px) == 0

def test_bt_distance_full_valuation_aware(ctx5):
    """bt_distance_full should distinguish x from p*x."""
    x = Qp.from_int(ctx5, 3)
    px = Qp.from_int(ctx5, 15)
    # Their valuation-aware digit sequences differ at the leading position
    assert bt_distance_full(ctx5, x, px) > 0

def test_digits_with_valuation_zero(ctx5):
    digs = digits_with_valuation(Qp.zero(ctx5), ctx5.prec)
    assert digs == [0] * ctx5.prec

def test_lca_depth_common_prefix():
    assert lca_depth([1, 2, 3, 4], [1, 2, 0, 4]) == 2

def test_lca_depth_empty():
    assert lca_depth([], [1, 2]) == 0


# ===========================================================================
# 8. Pairwise distance matrix
# ===========================================================================

def test_pairwise_dist_symmetric(ctx5):
    pts = [Qp.from_int(ctx5, k) for k in [1, 2, 5, 10]]
    D = pairwise_padic_dist(pts)
    assert D.shape == (4, 4)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), 0)

def test_pairwise_dist_ultrametric(ctx5):
    """All triangles in the pairwise matrix satisfy ultrametric inequality."""
    pts = [Qp.from_int(ctx5, k) for k in [1, 2, 5, 10, 25]]
    D = pairwise_padic_dist(pts)
    n = len(pts)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert D[i, k] <= max(D[i, j], D[j, k]) + 1e-12


# ===========================================================================
# 9. PadicKNNClassifier
# ===========================================================================

def test_knn_fit_predict(ctx3):
    X = [[Qp.from_int(ctx3, n)] for n in [1, 4, 10, 13, 40, 121]]
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = PadicKNNClassifier(ctx3, k=1)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert (pred == y).all()

def test_knn_predict_proba(ctx3):
    X = [[Qp.from_int(ctx3, n)] for n in [1, 4, 10, 13, 40, 121]]
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = PadicKNNClassifier(ctx3, k=3)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

def test_knn_context_mismatch_raises(ctx3):
    ctx7 = QpContext(7, 6)
    X = [[Qp.from_int(ctx3, 1)]]
    clf = PadicKNNClassifier(ctx7, k=1)
    clf.fit([[Qp.from_int(ctx7, 1)]], np.array([0]))
    with pytest.raises(ValueError, match="context"):
        clf.predict(X)

def test_knn_not_fitted_raises(ctx3):
    from sklearn.exceptions import NotFittedError
    clf = PadicKNNClassifier(ctx3, k=1)
    with pytest.raises(NotFittedError):
        clf.predict([[Qp.from_int(ctx3, 1)]])


# ===========================================================================
# 10. Edge cases: digits_with_valuation (negative valuations)
# ===========================================================================

@pytest.mark.parametrize("num,den,expected_v", [
    (1, 5,   -1),
    (1, 25,  -2),
    (1, 125, -3),
])
def test_digits_with_valuation_encodes_negative_valuation(ctx5, num, den, expected_v):
    """digits_with_valuation must produce a different sequence for v < 0 vs v = 0.

    Specifically, 1 (v=0) and 1/p^k (v=-k) must not share the same encoding,
    otherwise bt_distance_full collapses elements that differ by powers of p.
    """
    x = Qp.from_int(ctx5, 1)                    # v = 0
    y = Qp.from_rational(ctx5, num, den)         # v < 0
    assert y.val() == expected_v
    dx = digits_with_valuation(x, ctx5.prec)
    dy = digits_with_valuation(y, ctx5.prec)
    assert dx != dy, (
        f"digits_with_valuation failed to distinguish v=0 from v={expected_v}: "
        f"both returned {dx}"
    )


@pytest.mark.parametrize("base", [1, 2, 3, 7, 12])
def test_bt_distance_full_separates_x_from_px_all_valuations(ctx5, base):
    """bt_distance_full(x, p*x) > 0 for any nonzero x (v >= 0 and v < 0)."""
    x  = Qp.from_int(ctx5, base)
    px = Qp.from_int(ctx5, base * ctx5.p)
    assert bt_distance_full(ctx5, x, px) > 0, (
        f"bt_distance_full returned 0 for x={base} and p*x={base*ctx5.p}"
    )


def test_bt_distance_full_negative_valuation_separated(ctx5):
    """bt_distance_full must be >0 between 1/p (v=-1) and 1/p^2 (v=-2)."""
    a = Qp.from_rational(ctx5, 1, 5)    # v = -1
    b = Qp.from_rational(ctx5, 1, 25)   # v = -2
    assert bt_distance_full(ctx5, a, b) > 0


# ===========================================================================
# 11. Edge cases: pairwise_padic_dist_vec validation
# ===========================================================================

def test_pairwise_dist_vec_rejects_mismatched_lengths(ctx5):
    """pairwise_padic_dist_vec must raise ValueError for unequal vector lengths."""
    a = Qp.from_int(ctx5, 1)
    b = Qp.from_int(ctx5, 2)
    c = Qp.from_int(ctx5, 3)
    X = [[a, b], [c]]   # row 0 has length 2, row 1 has length 1
    with pytest.raises(ValueError, match="length"):
        pairwise_padic_dist_vec(X)


def test_pairwise_dist_vec_rejects_mismatched_contexts():
    """pairwise_padic_dist_vec must raise ValueError for mixed QpContexts."""
    ctx_a = QpContext(3, prec=5)
    ctx_b = QpContext(3, prec=6)
    X = [
        [Qp.from_int(ctx_a, 1)],
        [Qp.from_int(ctx_b, 1)],
    ]
    with pytest.raises(ValueError, match="[Cc]ontext"):
        pairwise_padic_dist_vec(X)


def test_pairwise_dist_vec_rejects_non_qp_elements(ctx5):
    """pairwise_padic_dist_vec must raise TypeError for non-Qp elements."""
    X = [[1.0, 2.0]]   # plain floats, not Qp
    with pytest.raises(TypeError):
        pairwise_padic_dist_vec(X)


def test_pairwise_dist_vec_valid_input(ctx5):
    """pairwise_padic_dist_vec must succeed for a valid uniform-context input."""
    X = [[Qp.from_int(ctx5, n), Qp.from_int(ctx5, n + 1)] for n in [1, 6, 11]]
    D = pairwise_padic_dist_vec(X)
    assert D.shape == (3, 3)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), 0)


# ===========================================================================
# 12. Edge cases: QpBall.refine precision guard
# ===========================================================================

def test_ball_refine_raises_at_precision_limit(ctx5):
    """refine must raise ValueError when n >= ctx.prec (would exceed precision)."""
    center = Qp.from_int(ctx5, 1)
    # ctx5.prec = 8; a ball at depth 8 cannot be refined further
    deep_ball = QpBall(center, ctx5.prec)
    with pytest.raises(ValueError, match="precision"):
        deep_ball.refine()


def test_ball_refine_raises_beyond_precision_limit(ctx5):
    """refine must also raise when n > ctx.prec."""
    center = Qp.from_int(ctx5, 1)
    beyond_ball = QpBall(center, ctx5.prec + 2)
    with pytest.raises(ValueError, match="precision"):
        beyond_ball.refine()


def test_ball_refine_succeeds_just_below_precision(ctx5):
    """refine must succeed for n = ctx.prec - 1 (the last valid depth)."""
    center = Qp.from_int(ctx5, 1)
    near_limit_ball = QpBall(center, ctx5.prec - 1)
    children = near_limit_ball.refine()
    assert len(children) == ctx5.p
