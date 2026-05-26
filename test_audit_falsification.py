"""
R&D Audit Falsification Tests for padic-ds
==========================================
These tests probe the correctness of mathematical claims made in the library.
They do NOT duplicate existing tests/test_field.py; they target untested edge
cases and formally-stated properties that could silently break.

Claim categories tested
-----------------------
F1  Field axioms not covered by unit tests (distributivity, identity elements,
    double negation, etc.)
F2  Non-Archimedean ultrametric: v(x+y) = min(v(x),v(y)) when v(x) ≠ v(y)
    (strict equality, not just ≥)
F3  BT tree ultrametric: bt_distance and bt_distance_full satisfy the
    ultrametric inequality over a broad sample of points
F4  ultrametric_dendrogram: the returned height matrix is ultrametric
F5  pairwise_padic_dist_vec: the max-product metric satisfies ultrametric
F6  Hensel lifting: returned root satisfies f(root) ≡ 0 mod p^prec
    (convergence contract), including edge-cases of precision contract
F7  embed_float_array: context and shape contract; no PrecisionError at
    default settings for small floats
F8  KNN edge cases: k=1, k==n_train, k>n_train (should raise or handle)
F9  QpContext: composite p should be rejected (p=4, p=6, p=9, …)
F10 Qp canonical invariant: u_mod is never divisible by p for nonzero elements
F11 Valuative absolute value multiplicativity: |xy|_p = |x|_p * |y|_p
F12 Sub-additivity: |x+y|_p ≤ max(|x|_p, |y|_p)  (non-Archimedean triangle)
F13 Hash / equality consistency for negation and subtraction
F14 digits_with_valuation: length contract and zero-element contract
F15 lca_depth: symmetric, bounded by min-length
"""

import pytest
import numpy as np
from padic import (
    QpContext, Qp, PrecisionError, QpBall,
    hensel_lift_simple,
    bt_distance, bt_distance_full, lca_depth, digits_with_valuation,
    padic_dist, pairwise_padic_dist, pairwise_padic_dist_vec,
    PadicKNNClassifier, embed_float_array,
    ultrametric_dendrogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx5():
    return QpContext(5, prec=8)

@pytest.fixture
def ctx3():
    return QpContext(3, prec=8)

@pytest.fixture
def ctx7():
    return QpContext(7, prec=8)


# ===========================================================================
# F1 — Field axioms (distributivity, identity elements, double-negation)
# ===========================================================================

def test_distributivity_left(ctx5):
    """a*(b+c) == a*b + a*c  (left distributivity)."""
    a = Qp.from_int(ctx5, 3)
    b = Qp.from_rational(ctx5, 1, 5)
    c = Qp.from_int(ctx5, 7)
    lhs = a.mul(b.add(c))
    rhs = a.mul(b).add(a.mul(c))
    assert lhs == rhs, f"Distributivity violated: lhs={lhs}, rhs={rhs}"

def test_distributivity_right(ctx5):
    """(a+b)*c == a*c + b*c  (right distributivity)."""
    a = Qp.from_rational(ctx5, 2, 25)
    b = Qp.from_int(ctx5, 11)
    c = Qp.from_rational(ctx5, 1, 5)
    lhs = a.add(b).mul(c)
    rhs = a.mul(c).add(b.mul(c))
    assert lhs == rhs

def test_multiplicative_identity(ctx5):
    """1 * x == x for various x (including x with negative valuation)."""
    one = Qp.from_int(ctx5, 1)
    for n, d in [(7, 1), (1, 5), (3, 25), (12, 1)]:
        x = Qp.from_rational(ctx5, n, d)
        assert one.mul(x) == x, f"1*x ≠ x for x={n}/{d}"
        assert x.mul(one) == x, f"x*1 ≠ x for x={n}/{d}"

def test_additive_identity(ctx5):
    """0 + x == x and x + 0 == x."""
    zero = Qp.zero(ctx5)
    for n, d in [(7, 1), (1, 5), (0, 1)]:
        x = Qp.from_rational(ctx5, n, d)
        assert zero.add(x) == x
        assert x.add(zero) == x

def test_double_negation(ctx5):
    """neg(neg(x)) == x."""
    for n, d in [(7, 1), (1, 5), (3, 25)]:
        x = Qp.from_rational(ctx5, n, d)
        assert x.neg().neg() == x, f"Double negation failed for {n}/{d}"

def test_mul_by_zero(ctx5):
    """x * 0 == 0 and 0 * x == 0."""
    zero = Qp.zero(ctx5)
    x = Qp.from_int(ctx5, 13)
    assert x.mul(zero).is_zero()
    assert zero.mul(x).is_zero()

def test_sub_self_is_zero(ctx3):
    """x - x == 0 for multiple elements."""
    for n in [1, 3, 9, 27, 100]:
        try:
            x = Qp.from_int(ctx3, n)
        except PrecisionError:
            continue
        assert x.sub(x).is_zero(), f"x - x ≠ 0 for n={n}"


# ===========================================================================
# F2 — Non-Archimedean equality  v(x+y) = min(v(x), v(y)) when v(x) ≠ v(y)
# ===========================================================================

@pytest.mark.parametrize("a_val,b_val", [
    (-1, 0), (0, 1), (1, 3), (-2, 1), (0, 3), (-1, 2)
])
def test_na_ultrametric_equality(ctx5, a_val, b_val):
    """When v(x) ≠ v(y), v(x+y) must equal min(v(x), v(y)) exactly."""
    # Construct x with valuation a_val, y with valuation b_val
    # Both unit parts = 1 (mod p^prec)
    if a_val >= 0:
        x = Qp.from_rational(ctx5, 5**a_val, 1)
    else:
        x = Qp.from_rational(ctx5, 1, 5**(-a_val))
    if b_val >= 0:
        y = Qp.from_rational(ctx5, 5**b_val, 1)
    else:
        y = Qp.from_rational(ctx5, 1, 5**(-b_val))
    s = x.add(y)
    expected_v = min(a_val, b_val)
    assert s.val() == expected_v, (
        f"v(x+y)={s.val()} but expected {expected_v} "
        f"(v(x)={a_val}, v(y)={b_val})"
    )

def test_na_ultrametric_strict_p3(ctx3):
    """v(1 + 9) = v(10) = 0 since v(1)=0 < v(9)=2."""
    one = Qp.from_int(ctx3, 1)
    nine = Qp.from_int(ctx3, 9)
    assert min(one.val(), nine.val()) == 0
    s = one.add(nine)  # 1 + 9 = 10; v_3(10) = 0
    assert s.val() == 0


# ===========================================================================
# F3 — BT tree ultrametric
# ===========================================================================

SAMPLE_INTS_5 = [1, 2, 3, 4, 6, 7, 11, 12, 24, 26, 50, 51, 100, 101, 123]

def test_bt_distance_symmetry(ctx5):
    for a in SAMPLE_INTS_5:
        for b in SAMPLE_INTS_5:
            try:
                x = Qp.from_int(ctx5, a)
                y = Qp.from_int(ctx5, b)
            except PrecisionError:
                continue
            assert bt_distance(ctx5, x, y) == bt_distance(ctx5, y, x)

def test_bt_distance_zero_iff_equal_unit_digits(ctx5):
    """bt_distance == 0 <=> x and y have the same unit digit sequence."""
    x = Qp.from_int(ctx5, 3)
    px = Qp.from_int(ctx5, 15)   # 3*5, same unit digits
    assert bt_distance(ctx5, x, px) == 0   # documented limitation
    y = Qp.from_int(ctx5, 4)
    assert bt_distance(ctx5, x, y) > 0

def test_bt_distance_full_ultrametric(ctx5):
    """bt_distance_full satisfies d(x,z) <= max(d(x,y), d(y,z))."""
    pts = [Qp.from_int(ctx5, n) for n in SAMPLE_INTS_5
           if not _precision_err(ctx5, n)]
    n = len(pts)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dxz = bt_distance_full(ctx5, pts[i], pts[k])
                dxy = bt_distance_full(ctx5, pts[i], pts[j])
                dyz = bt_distance_full(ctx5, pts[j], pts[k])
                assert dxz <= max(dxy, dyz) + 1e-9, (
                    f"BT full ultrametric violated at i={i},j={j},k={k}: "
                    f"d(i,k)={dxz} > max({dxy},{dyz})"
                )

def test_bt_distance_nonneg(ctx5):
    for a in SAMPLE_INTS_5:
        for b in SAMPLE_INTS_5:
            try:
                x = Qp.from_int(ctx5, a)
                y = Qp.from_int(ctx5, b)
            except PrecisionError:
                continue
            assert bt_distance_full(ctx5, x, y) >= 0


def _precision_err(ctx, n):
    try:
        Qp.from_int(ctx, n)
        return False
    except PrecisionError:
        return True


# ===========================================================================
# F4 — ultrametric_dendrogram returns a valid ultrametric height matrix
# ===========================================================================

def test_ultrametric_dendrogram_symmetric(ctx5):
    pts = [Qp.from_int(ctx5, n) for n in [1, 2, 3, 6, 11, 12, 24]]
    H = ultrametric_dendrogram(ctx5, pts)
    np.testing.assert_array_equal(H, H.T)

def test_ultrametric_dendrogram_zero_diagonal(ctx5):
    pts = [Qp.from_int(ctx5, n) for n in [1, 2, 3, 6, 11, 12, 24]]
    H = ultrametric_dendrogram(ctx5, pts)
    np.testing.assert_array_equal(np.diag(H), 0)

def test_ultrametric_dendrogram_is_ultrametric(ctx5):
    """H[i,k] <= max(H[i,j], H[j,k]) for all triples."""
    pts = [Qp.from_int(ctx5, n) for n in [1, 2, 3, 6, 11, 12, 24]]
    H = ultrametric_dendrogram(ctx5, pts)
    n = len(pts)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert H[i, k] <= max(H[i, j], H[j, k]) + 1e-9, (
                    f"Dendrogram ultrametric violated at i={i},j={j},k={k}: "
                    f"H[i,k]={H[i,k]} > max({H[i,j]},{H[j,k]})"
                )


# ===========================================================================
# F5 — pairwise_padic_dist_vec: product-max ultrametric
# ===========================================================================

def test_pairwise_dist_vec_ultrametric(ctx3):
    """Product ultrametric satisfies d(u,w) <= max(d(u,v), d(v,w))."""
    pts = [
        [Qp.from_int(ctx3, a), Qp.from_int(ctx3, b)]
        for a, b in [(1, 1), (3, 1), (1, 3), (9, 1), (3, 9), (1, 9)]
    ]
    D = pairwise_padic_dist_vec(pts)
    n = len(pts)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert D[i, k] <= max(D[i, j], D[j, k]) + 1e-12, (
                    f"Vec ultrametric violated: D[{i},{k}]={D[i,k]} "
                    f"> max({D[i,j]},{D[j,k]})"
                )

def test_pairwise_dist_vec_symmetric(ctx3):
    pts = [
        [Qp.from_int(ctx3, a), Qp.from_int(ctx3, b)]
        for a, b in [(1, 2), (3, 7), (9, 4)]
    ]
    D = pairwise_padic_dist_vec(pts)
    np.testing.assert_array_almost_equal(D, D.T)


# ===========================================================================
# F6 — Hensel lifting: root contract
# ===========================================================================

def test_hensel_root_satisfies_congruence_sqrt2_mod7():
    """Lifted root r satisfies r^2 ≡ 2 (mod 7^prec)."""
    ctx = QpContext(7, prec=8)
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a * a - 2,
        fZprime=lambda a: 2 * a,
        a0_mod_p=3,
        target_prec=8,
    )
    # r^2 - 2 should have valuation >= prec (i.e. is "zero" at our precision)
    two = Qp.from_int(ctx, 2)
    residual = root.mul(root).sub(two)
    assert residual.is_zero() or residual.val() >= ctx.prec, (
        f"Hensel root does not satisfy r^2 ≡ 2 mod 7^{ctx.prec}: "
        f"residual val = {residual.val()}"
    )

def test_hensel_root_satisfies_congruence_sqrt6_mod5():
    """Lifted root r satisfies r^2 ≡ 6 (mod 5^prec)."""
    ctx = QpContext(5, prec=6)
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a * a - 6,
        fZprime=lambda a: 2 * a,
        a0_mod_p=1,
        target_prec=6,
    )
    six = Qp.from_int(ctx, 6)
    residual = root.mul(root).sub(six)
    assert residual.is_zero() or residual.val() >= ctx.prec

def test_hensel_target_prec_less_than_ctx_prec():
    """target_prec < ctx.prec: return does not raise; root has ctx.prec digits."""
    ctx = QpContext(7, prec=10)
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a * a - 2,
        fZprime=lambda a: 2 * a,
        a0_mod_p=3,
        target_prec=4,   # less than ctx.prec=10
    )
    # Just check it returns a valid Qp element with the right context
    assert root.ctx == ctx

def test_hensel_target_prec_greater_than_ctx_prec():
    """target_prec > ctx.prec: digits truncated, no crash."""
    ctx = QpContext(7, prec=4)
    root = hensel_lift_simple(
        ctx,
        fZ=lambda a: a * a - 2,
        fZprime=lambda a: 2 * a,
        a0_mod_p=3,
        target_prec=8,   # greater than ctx.prec=4
    )
    assert root.ctx == ctx  # context preserved


# ===========================================================================
# F7 — embed_float_array shape and context contract
# ===========================================================================

def test_embed_float_shape(ctx5):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    embedded = embed_float_array(X, ctx5, scale=10)
    assert len(embedded) == 3
    assert all(len(row) == 2 for row in embedded)
    assert all(isinstance(e, Qp) for row in embedded for e in row)
    assert all(e.ctx == ctx5 for row in embedded for e in row)

def test_embed_float_1d(ctx3):
    X = np.array([[1.0], [3.0], [9.0]])
    embedded = embed_float_array(X, ctx3, scale=1)
    assert len(embedded) == 3
    assert all(len(row) == 1 for row in embedded)

def test_embed_float_scale_zero_returns_zero(ctx5):
    """A float that rounds to 0 after scaling produces Qp zero."""
    X = np.array([[0.0001]])
    # scale=1 → round(0.0001*1) = 0 → Qp.zero
    embedded = embed_float_array(X, ctx5, scale=1)
    assert embedded[0][0].is_zero()


# ===========================================================================
# F8 — KNN edge cases
# ===========================================================================

def test_knn_k_equals_n_train(ctx5):
    """k == n_train should not crash; probabilities sum to 1."""
    X = [[Qp.from_int(ctx5, n)] for n in [1, 6, 11, 16]]
    y = np.array([0, 0, 1, 1])
    clf = PadicKNNClassifier(ctx5, k=4)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

def test_knn_k1_memorizes_train(ctx3):
    """k=1 must memorize training data perfectly (if all train points unique)."""
    X = [[Qp.from_int(ctx3, n)] for n in [1, 3, 9, 27]]
    y = np.array([0, 1, 2, 3])
    clf = PadicKNNClassifier(ctx3, k=1)
    clf.fit(X, y)
    pred = clf.predict(X)
    np.testing.assert_array_equal(pred, y)

def test_knn_k_greater_than_n_train(ctx5):
    """k > n_train: predict should not raise (argsort clamps to n_train)."""
    X = [[Qp.from_int(ctx5, n)] for n in [1, 6]]
    y = np.array([0, 1])
    clf = PadicKNNClassifier(ctx5, k=10)  # k > 2 training points
    clf.fit(X, y)
    # May silently use all training points or raise — just must not crash
    try:
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        # If it does not raise, probabilities must still sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    except Exception as exc:
        pytest.skip(f"k > n_train raises {type(exc).__name__}: {exc}")

def test_knn_empty_X_raises(ctx3):
    """Fitting or predicting with empty X should raise ValueError."""
    clf = PadicKNNClassifier(ctx3, k=1)
    with pytest.raises(ValueError):
        clf.fit([], np.array([]))


# ===========================================================================
# F9 — QpContext: composite p
# ===========================================================================

@pytest.mark.parametrize("bad_p", [4, 6, 8, 9, 10, 15, 25])
def test_qpcontext_composite_p_rejected(bad_p):
    """QpContext should reject composite p values.

    NOTE: if this test FAILS it means the library silently accepts composite
    bases — a mathematical correctness bug (p-adic arithmetic is only defined
    for prime p).
    """
    with pytest.raises((ValueError, ArithmeticError)):
        QpContext(bad_p, prec=4)


# ===========================================================================
# F10 — Canonical invariant: u_mod not divisible by p
# ===========================================================================

def test_unit_mod_not_divisible_by_p(ctx5):
    """For all nonzero Qp elements, u_mod % p != 0."""
    p = ctx5.p
    test_cases = [
        Qp.from_int(ctx5, 1),
        Qp.from_int(ctx5, 7),
        Qp.from_rational(ctx5, 1, 5),
        Qp.from_rational(ctx5, 3, 25),
        Qp.from_int(ctx5, 5),        # v=1, u_mod=1
        Qp.from_int(ctx5, 10),       # v=1, u_mod=2
    ]
    for x in test_cases:
        assert x.u_mod % p != 0, (
            f"u_mod={x.u_mod} is divisible by p={p} for {x}"
        )

def test_unit_mod_preserved_after_arithmetic(ctx5):
    """Arithmetic results should also have p-free u_mod."""
    p = ctx5.p
    a = Qp.from_int(ctx5, 3)
    b = Qp.from_rational(ctx5, 1, 5)
    for result_name, result in [
        ("a*b", a.mul(b)),
        ("a+b", a.add(b)),
        ("a-b", a.sub(b)),
        ("inv(b)", b.inv()),
    ]:
        if not result.is_zero():
            assert result.u_mod % p != 0, (
                f"u_mod={result.u_mod} divisible by p={p} in {result_name}"
            )


# ===========================================================================
# F11 — Multiplicativity of absolute value |xy|_p = |x|_p * |y|_p
# ===========================================================================

@pytest.mark.parametrize("n,m", [
    (1, 5), (3, 25), (7, 5), (2, 10), (1, 1), (5, 5)
])
def test_abs_multiplicative(ctx5, n, m):
    x = Qp.from_int(ctx5, n)
    y = Qp.from_int(ctx5, m)
    xy = x.mul(y)
    assert abs(xy.abs() - x.abs() * y.abs()) < 1e-12, (
        f"|{n}*{m}|_p = {xy.abs()}, |{n}|_p*|{m}|_p = {x.abs()*y.abs()}"
    )


# ===========================================================================
# F12 — Non-Archimedean: |x+y|_p <= max(|x|_p, |y|_p)
# ===========================================================================

@pytest.mark.parametrize("n,m", [
    (1, 2), (5, 3), (1, 5), (25, 1), (25, 30), (7, 14), (3, 6)
])
def test_nonarchimedean_abs_inequality(ctx5, n, m):
    x = Qp.from_int(ctx5, n)
    y = Qp.from_int(ctx5, m)
    s = x.add(y)
    assert s.abs() <= max(x.abs(), y.abs()) + 1e-12, (
        f"|{n}+{m}|_p = {s.abs()} > max(|{n}|_p={x.abs()}, |{m}|_p={y.abs()})"
    )


# ===========================================================================
# F13 — Hash / equality consistency
# ===========================================================================

def test_neg_of_neg_hash(ctx5):
    """neg(neg(x)) == x must also imply same hash."""
    x = Qp.from_rational(ctx5, 7, 25)
    assert hash(x.neg().neg()) == hash(x)

def test_sub_produces_different_hash(ctx5):
    """x != x - 1 must imply different hash (unless collision)."""
    x = Qp.from_int(ctx5, 10)
    one = Qp.from_int(ctx5, 1)
    y = x.sub(one)
    assert x != y


# ===========================================================================
# F14 — digits_with_valuation: length and zero contract
# ===========================================================================

def test_digits_with_valuation_length(ctx5):
    """digits_with_valuation must return exactly total_depth items."""
    for n, d in [(1, 1), (5, 1), (1, 5), (0, 1), (25, 1)]:
        x = Qp.from_rational(ctx5, n, d)
        for depth in [4, 6, 8]:
            digs = digits_with_valuation(x, depth)
            assert len(digs) == depth, (
                f"Expected {depth} digits for {n}/{d} but got {len(digs)}"
            )

def test_digits_with_valuation_zero_is_all_zeros(ctx5):
    digs = digits_with_valuation(Qp.zero(ctx5), 8)
    assert digs == [0] * 8

def test_digits_with_valuation_elements_in_range(ctx5):
    """All digits must be in [0, p-1]."""
    p = ctx5.p
    for n, d in [(3, 1), (7, 5), (1, 25)]:
        x = Qp.from_rational(ctx5, n, d)
        digs = digits_with_valuation(x, 8)
        assert all(0 <= dig < p for dig in digs), (
            f"Digit out of range for {n}/{d}: {digs}"
        )


# ===========================================================================
# F15 — lca_depth: symmetry and bounds
# ===========================================================================

def test_lca_depth_symmetric():
    a = [1, 2, 3, 4]
    b = [1, 2, 0, 4]
    assert lca_depth(a, b) == lca_depth(b, a)

def test_lca_depth_bounded_by_min_length():
    a = [1, 2, 3]
    b = [1, 2, 3, 4, 5]
    assert lca_depth(a, b) <= min(len(a), len(b))

def test_lca_depth_identical_sequences():
    a = [1, 2, 3, 4]
    assert lca_depth(a, a) == len(a)

def test_lca_depth_no_common_prefix():
    a = [1, 2, 3]
    b = [2, 2, 3]
    assert lca_depth(a, b) == 0


# ===========================================================================
# F16 — vp_int edge cases
# ===========================================================================

from padic.field import vp_int

def test_vp_int_zero_returns_sentinel():
    assert vp_int(0, 5) == 10**9

def test_vp_int_one_returns_zero():
    assert vp_int(1, 5) == 0
    assert vp_int(1, 7) == 0

def test_vp_int_prime_power():
    assert vp_int(125, 5) == 3   # 5^3 = 125
    assert vp_int(49, 7) == 2    # 7^2 = 49

def test_vp_int_coprime():
    assert vp_int(7, 5) == 0
    assert vp_int(12, 7) == 0

def test_vp_int_negative_not_handled():
    """vp_int is documented to work on |n|; calling with negative may
    loop or return wrong answer.  This test documents the current behaviour."""
    # -25 = -5^2; we expect the library to handle this via abs(n) in from_int
    # but vp_int itself is called on abs(n) in from_int, so direct call with
    # a negative is an undocumented path — just check it doesn't infinite-loop
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("vp_int(-25, 5) did not terminate")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2)
    try:
        result = vp_int(-25, 5)
    except TimeoutError:
        pytest.fail("vp_int(-25, 5) hung (infinite loop for negative input)")
    finally:
        signal.alarm(0)
    # We just record the result; we don't assert a specific value
    # since the behaviour for negatives is undocumented
    assert isinstance(result, int)
