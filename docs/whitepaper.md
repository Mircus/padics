# The Advantages of p-Adic Numbers for Advanced AI

**Technical Brief · padic-ds Project · May 2026 · v3 (expanded & synchronised)**

---

## Abstract

Modern machine learning rests almost entirely on Euclidean geometry and its
derivatives — dot products, L² distances, gradient flows in ℝⁿ. Yet many of
the hardest problems in AI — hierarchical reasoning, symbolic structure,
compositional generalisation, robustness to adversarial perturbation — are
fundamentally *non-Euclidean* in character. p-Adic numbers, the unique
completions of ℚ with respect to the *p-adic absolute value*, offer a
mathematically rigorous alternative geometry whose properties align
surprisingly well with the structural demands of advanced AI.

This document serves three purposes:

1. **Technical argument** — seven concrete advantages of the p-adic framework
   for AI, each grounded in mathematics and in the `padic-ds` reference
   implementation.
2. **Repository guide** — a full walkthrough of the `padic-ds` v0.1.1 library:
   modules, API, install, tests, and notebooks.
3. **Literature survey** — an annotated bibliography of p-adic methods in AI
   and related non-Euclidean ML, from foundational theory (1980–2002) through
   recent deep learning results (2022–2026).

Each section is labelled with its implementation status:  
**[Implemented]** — working code in padic-ds;  
**[Near-term roadmap]** — well-specified, buildable now;  
**[Research prospect]** — theoretical motivation, no implementation yet.

---

## 1. Background: What Are p-Adic Numbers?

For any prime *p*, the *p-adic absolute value* of a rational number
x = pᵛ · (a/b) (with p ∤ a, p ∤ b) is defined as

```
|x|_p  =  p^{−v_p(x)}
```

where v_p(x) is the *p-adic valuation* — the exponent of p in the prime
factorisation of x. The field ℚ_p is the completion of ℚ under this metric,
analogous to the way ℝ is the completion under the ordinary absolute value.
Three facts distinguish ℚ_p from ℝ radically:

1. **The ultrametric inequality**: |x + y|_p ≤ max(|x|_p, |y|_p), which is
   strictly stronger than the ordinary triangle inequality.
2. **Every ball is clopen**: balls are simultaneously open and closed, and any
   two balls are either disjoint or one contains the other.
3. **Numbers large in the ordinary sense are small p-adically, and
   vice-versa**: powers of p become *smaller*, not larger.

Together these properties endow ℚ_p with a canonical *tree structure* — the
Bruhat–Tits (BT) tree — where arithmetic proximity is equivalent to shared
prefixes in a base-p digit expansion. The `padic-ds` library (`src/padic/`)
implements this from first principles: exact Q_p arithmetic with finite
precision (`field.py`), ultrametric balls (`ball.py`), Hensel lifting
(`hensel.py`), BT-tree distances (`btree.py`), and prototype ML algorithms
(`knn.py`, `hclust.py`).

**Notation used throughout**: Q_p (p-adic field), Z_p (p-adic integers),
PGL₂(Q_p) (projective linear group), BT tree (Bruhat–Tits tree).

---

## 2. Advantage 1 — Exact Hierarchical Geometry  [Implemented]

### The Problem With Euclidean Embeddings of Trees

A fundamental difficulty in Euclidean geometry is that trees cannot be
embedded isometrically in any finite-dimensional ℝⁿ with the L² norm.
Bourgain's 1985 theorem states that any *n*-point metric space can be
embedded into L² with O(log n) distortion — but this is a bound for *general*
metrics and is not tight for trees. For tree metrics specifically:

* Any finite tree embeds **isometrically** into L¹ (Dress 1984 [R2]; the
  four-point condition for tree metrics coincides with L¹ hyperbolicity).
* The same trees require **Ω(√log n) distortion** when embedded into L²
  (Matousek 1999), and this lower bound is essentially tight.

In practice, learning hierarchical knowledge — taxonomies, parse trees,
ontologies, class hierarchies — in Euclidean space still demands either heavy
over-parameterisation or an explicit architectural inductive bias (e.g.,
hyperbolic embeddings [R10], tree-structured LSTMs).

### p-Adics as a Natural Tree Metric

In Q_p, the ultrametric inequality *forces* the geometry to be tree-like.
Every element of Z_p has a canonical base-p digit expansion
a₀ + a₁p + a₂p² + …, and the p-adic distance between two elements equals
p^{−k} where k is the position of their first differing digit. This is
*exactly* the longest-common-prefix (LCP) metric on a p-ary trie.

```python
# from btree.py — implemented in padic-ds
def lca_depth(d1: List[int], d2: List[int]) -> int:
    """Depth of longest common prefix of two digit sequences."""
    m = min(len(d1), len(d2))
    k = 0
    for i in range(m):
        if d1[i] == d2[i]: k += 1
        else: break
    return k
```

There is no distortion: the p-adic metric *is* the tree metric, not an
approximation of it. Embedding a taxonomy of depth D into Z_p at precision D
is lossless. For AI tasks built on hierarchical structures — WordNet
embeddings, biological taxonomies, code call-graphs, compositional grammar —
this eliminates an entire class of representational error.

---

## 3. Advantage 2 — The Ultrametric Mitigates Crowding in kNN  [Implemented]

### Curse of Dimensionality in Standard Metrics

High-dimensional Euclidean spaces suffer from *concentration of measure*:
pairwise distances converge to the same value, making kNN and clustering
degenerate. This is a primary reason why nearest-neighbour methods fail in
raw feature spaces and require dimension reduction or manifold learning as a
preprocessing step.

### Why Ultrametrics Help (with Caveats)

In an ultrametric space, the strong triangle inequality — d(x,z) ≤
max{d(x,y), d(y,z)} — means any two balls are either nested or disjoint, with
no ambiguous border regions. Crucially, the **distance spectrum is
discrete**: in Q_p, pairwise distances can only take values in
{0, 1, p, p², p³, …}, not a continuum. This discretisation keeps kNN
decision boundaries crisp even as the number of points grows.

**Important qualification**: This dimension-independence argument applies to
the *single-coordinate* p-adic metric. For product spaces Q_p^d equipped
with the sup-norm max_i d_p(x_i, y_i), each ball of radius p^{-k} refines
into **p^d** sub-balls at the next finer resolution. The branching factor
grows exponentially with d, just as in Euclidean space. The mitigation of
crowding comes not from dimension-independence per se, but from the **quantized
distance spectrum**: near-ties cluster at the same exact value p^{-k} rather
than forming a diffuse cloud, so rank-ordering of neighbours remains stable
under perturbations that stay below the next level p^{-(k+1)}.

The `ultrametric_dendrogram` in `hclust.py` builds an exact ultrametric matrix
from LCA depths — a property that Euclidean linkage can only approximate:

```python
# from hclust.py — implemented in padic-ds
def ultrametric_dendrogram(ctx: QpContext, X: List[Qp]) -> np.ndarray:
    D = [digits_p_adic(x, ctx.prec) for x in X]
    n = len(X)
    H = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = ctx.prec - lca_depth(D[i], D[j])
            H[i, j] = H[j, i] = d
    return H   # exact ultrametric — no linkage approximation
```

**Empirical validation needed**: A rigorous comparison of nearest/farthest
distance ratios, hubness statistics, and kNN accuracy between Euclidean and
p-adic product metrics on hierarchical synthetic datasets is a planned
near-term experiment.

---

## 4. Advantage 3 — Hensel Lifting as a Model for Iterative Refinement  [Implemented — linear variant]

### Gradient Descent as Numerical Approximation

Standard deep learning optimisation (SGD, Adam) computes floating-point
approximations that converge asymptotically. Guarantees of convergence to
exact solutions (rather than approximate critical points) are rare and
fragile.

### Hensel's Lemma as Principled Lifting

Hensel's lemma guarantees that if f(a) ≡ 0 (mod p) and f′(a) ≢ 0 (mod p),
then a lifts *uniquely* to a root of f in Z_p. The lift is constructive.
Two variants exist:

**Linear Hensel lifting** (implemented in `hensel.py`):

```
a_{k+1} = a_k − f(a_k) · [f′(a_k)]⁻¹    (mod p^{k+1})
```

Gains one digit of precision per step; requires O(prec) iterations to reach
working precision.

**Newton–Hensel lifting** (quadratic; not yet implemented):

```
a_{k+1} = a_k − f(a_k) · [f′(a_k)]⁻¹    (mod p^{2^k})
```

Doubles precision each step; requires only O(log prec) iterations.

The current `padic-ds` implementation uses the **linear variant**:

```python
# from hensel.py — linear lifting (one digit per step), O(prec) iterations
while mod < N:
    mod *= p                              # linear: +1 digit per iteration
    fa  = fZ(a) % mod
    fpa = fZprime(a) % mod
    inv = pow(fpa, -1, mod)               # exact modular inverse
    a   = (a + (-fa * inv) % mod) % mod   # exact update
```

Every step is exact modulo the current working precision, with no
floating-point accumulation. Upgrading to Newton–Hensel (exponent-doubling
precision schedule) is a near-term roadmap item.

**AI relevance**: Problems that have algebraic solutions can be expressed as
polynomial equations over Z_p and solved to any desired precision without
floating-point error. Applications include constraint satisfaction, modular
polynomial systems relevant to SAT/SMT, and formal verification assistants.

---

## 5. Advantage 4 — p-Adic Wavelets and Multi-Scale Feature Learning  [Research Prospect]

> **Status**: No implementation in padic-ds. This section describes a
> theoretical opportunity; all claims are prospective.

### The Haar Wavelet on ℝ vs. Q_p

On Q_p, there exists a canonical orthonormal wavelet basis — the *p-adic Haar
wavelets* (Kozyrev 2002 [R5]) — whose support sets are exactly the clopen balls
B(a, p^{−n}). Because balls are clopen and pairwise disjoint, wavelet
coefficients at resolution p^{−n} are independent of those at p^{−m} for
m ≠ n: there is no inter-scale aliasing by construction.

### Implications for Neural Architectures

Multi-resolution analysis in ℝⁿ (CNNs, pooling, U-nets) must deal with
aliasing, edge artefacts, and stride hyperparameters. In theory, Q_p offers:

* **Exact pooling**: coarsening from p^{-k} to p^{-(k-1)} resolution is
  algebraically exact.
* **Depth = algebraic scale**: a layer at depth k processes features at
  resolution p^{-k}, with a provably complete and non-redundant basis.

**Critical caveat — finite precision**: Practical computation occurs in
**Z/p^N Z** (the quotient ring at precision N), which has wrap-around
arithmetic. At finite precision, convolution wraps around modulo p^N,
analogous to periodic boundary conditions on a torus. Eliminating these
artifacts requires either an inverse-limit construction or explicit padding
conventions.

The p-adic CNN literature ([R15], [R16], [R17]) has begun exploring
practical realisations of this idea; the padic-ds roadmap includes a Haar
transform prototype as the first step.

---

## 6. Advantage 5 — Symbolic Reasoning and Exact Arithmetic  [Implemented]

### Floating-Point Arithmetic Is Lossy

IEEE 754 floating-point arithmetic is approximate by design: rounding errors
accumulate, comparisons are unreliable (0.1 + 0.2 ≠ 0.3), and catastrophic
cancellation can invalidate intermediate results.

### p-Adic Arithmetic Is Exact at Fixed Precision

In Q_p at working precision N:

* Addition, subtraction, and multiplication are exact modulo p^N.
* Division by units (elements whose valuation is zero) is exact.
* The valuation is an integer, computable in O(log_p N) time.

```python
# from field.py — exact unit-level arithmetic, no floating-point
def mul(self, other: "Qp") -> "Qp":
    v = self.v + other.v
    u = (self.u_mod * other.u_mod) % self.ctx.modulus()  # exact
    return Qp(self.ctx, v=v, u_mod=u)
```

The *p-adic integers* Z_p form a complete discrete valuation ring where
divisibility, factorisation, and root-finding have clean algebraic
characterisations. AI systems operating over Z_p can reason about
number-theoretic properties *by type*, without numerical approximation.

**Finite precision artifacts**: At fixed N, cancellations that increase
valuation (e.g., x − y where x ≈ y mod p^k) consume precision silently.
Robust implementations must track precision loss through multi-step
computations.

This is relevant for:
* **Neuro-symbolic AI**: p-adic representations can carry exact arithmetic
  constraints that gradient-based learning preserves.
* **Formal verification assistants**: Lean/Coq tactics that discharge
  number-theoretic goals benefit from an AI component that understands p-adic
  arithmetic intrinsically.
* **Cryptographic reasoning**: Many post-quantum cryptography (PQC) schemes
  (NTRU, Kyber, Dilithium) operate over polynomial rings Z/qZ[x] — quotients
  structurally related to Z_p as an inverse limit of Z/p^kZ.

---

## 7. Advantage 6 — Group-Equivariant Learning on Tree-Structured Spaces  [Research Prospect]

> **Status**: No implementation in padic-ds. This section describes a
> theoretical opportunity; all claims are prospective pending experimental
> validation.

### Equivariance Is the Central Design Principle of Modern Architectures

CNNs exploit translation equivariance. Graph neural networks exploit
permutation equivariance. Transformers approximate permutation-equivariant
set functions. In each case, *baking in the symmetry group reduces sample
complexity* by the effective size of the orbit.

### The Symmetry Group of the BT Tree

The automorphism group of the Bruhat–Tits tree T_p is **PGL₂(Q_p)** — the
group of 2×2 invertible matrices over Q_p modulo its centre (Serre 1980 [R3],
*Trees*, Ch. II). This group includes:

* **Translations** by any element of Z_p (shifting the digit expansion).
* **Dilations** by powers of p (scaling the valuation).
* **Möbius transformations** over Q_p (the full group of tree automorphisms).

A neural architecture whose weight-sharing pattern respects PGL₂(Q_p) would
achieve equivariance to all of these simultaneously — a far richer symmetry
than the discrete integer translations of a standard CNN.

**Tempered claim**: p-adic group-equivariant networks trained on hierarchical
data (parse trees, ontologies, code ASTs) are *expected* to achieve sample
efficiency gains. The magnitude of this gain depends on the degree to which
training data realises the full PGL₂(Q_p) symmetry; quantifying this
experimentally is a prerequisite.

---

## 8. Advantage 7 — Algorithmic Efficiency via Exact Index Structures  [Near-term Roadmap]

### Approximate Nearest-Neighbour Is a Workaround

In high-dimensional Euclidean space, exact nearest-neighbour search is
NP-hard (under standard conjectures), so practical systems use approximate
methods (LSH, HNSW, ScaNN). These introduce recall–precision trade-offs and
require careful tuning.

### p-Adic Distance Is Computable in O(prec)

The `bt_distance_full` function in `btree.py` computes the valuation-aware
exact BT distance between two Q_p elements in **O(prec)** digit comparisons:

```python
# from btree.py — O(prec) exact distance computation
def bt_distance_full(ctx: QpContext, x: Qp, y: Qp) -> int:
    dx = digits_with_valuation(x, ctx.prec)
    dy = digits_with_valuation(y, ctx.prec)
    d  = lca_depth(dx, dy)
    return 2 * (ctx.prec - d)
```

**What padic-ds provides**: a distance function and valuation-aware variant.  
**What does not yet exist**: an index structure (p-ary trie, query API,
insert/delete, benchmarks against HNSW/FAISS).

**Near-term roadmap**:

1. *Single-dimension p-ary trie*: immutable trie keyed by digit prefixes,
   with O(prec) nearest-prefix search.
   API: `insert`, `delete`, `top_k_query`.
2. *Multi-dimension product trie*: priority search under sup-norm across
   coordinates, bounded by configurable beam width.
3. *Embedding module*: scale-and-round for real→Q_p; digit-trie encoding for
   hierarchical labels/paths; stability tests under small perturbations.
4. *Benchmarks*: recall, latency, and memory vs. HNSW on synthetic hierarchical
   datasets where ground-truth neighbours are defined by hierarchy labels.

---

## 9. Synthesis: Towards a p-Adic AI Stack

The advantages above are not independent — they form a coherent alternative AI
geometry. The table maps each advantage to a concrete AI subsystem and labels
current implementation status:

| p-Adic Advantage | AI Subsystem | Status |
|---|---|---|
| Exact tree metric | Knowledge graph embeddings | **Implemented** (btree.py, hclust.py) |
| Ultrametric kNN | Embedding retrieval / RAG | **Implemented** (knn.py) — index is roadmap |
| Hensel lifting | Constraint solvers, SAT/SMT | **Implemented** (linear); quadratic is roadmap |
| Exact arithmetic | Neuro-symbolic / theorem proving | **Implemented** (field.py) |
| BT exact index | Vector databases | **Near-term roadmap** — distance function only |
| p-Adic wavelets | Multi-scale sequence models | **Research prospect** |
| Group equivariance | Hierarchical data classification | **Research prospect** |

A coherent p-adic AI stack would look like:

```
Raw Data
   │
   ▼
p-Adic Embedding Layer        ← scale-and-round real→Q_p; digit encoding for symbols
   │                            [roadmap: PadicEmbed with stability tests]
   ▼
BT-Tree Index / Wavelet Bank  ← O(prec) distance (implemented); trie index (roadmap)
   │                            p-adic Haar wavelet bank (research prospect)
   ▼
p-Adic Equivariant Network    ← PGL₂(Q_p)-equivariant weight sharing [research prospect]
   │
   ▼
Hensel-Lifted Decoder         ← exact symbolic output for constrained tasks
   │                            [linear lifting implemented; Newton–Hensel roadmap]
   ▼
Output (exact or approximate)
```

The `padic-ds` library provides the foundational layers. The near-term
priority is the BT trie index and the embedding module, which together enable
the first end-to-end retrieval benchmarks.

---

## 10. Open Challenges

Intellectual honesty requires naming the barriers:

1. **Embedding design**: Mapping real-valued vectors to Q_p^d requires choices
   of prime p, precision prec, per-dimension scale, and quantisation scheme.
   No principled general-purpose strategy is known.

2. **Finite precision artifacts**: Operations that increase valuation
   (near-cancellations) consume precision silently. kNN and clustering
   stability under these cancellations needs rigorous analysis.

3. **Hardware mismatch**: Modern accelerators (TPUs, CUDA cores) are optimised
   for IEEE 754 arithmetic. p-Adic integer arithmetic requires software
   emulation (slow) or custom hardware.

4. **Gradient flow**: Z_p with the discrete topology has no standard notion of
   a gradient. The Valuation-Adaptive Perturbation Optimization (VAPO) of the
   v-PuNNs paper [R19] is the most recent proposal, but a general solution
   remains open.

5. **Scaling laws**: Transformer scaling laws were discovered empirically for
   Euclidean architectures. Equivalent laws for p-adic architectures are
   entirely unknown.

6. **High-dimensional branching**: For product ultrametrics Q_p^d, the
   branching factor p^d grows exponentially with d. Practical use in high
   dimensions requires dimensionality reduction before p-adic encoding,
   or sparse/approximate trie structures.

7. **Tooling immaturity**: `padic-ds` is a proof-of-concept. Production-grade
   p-adic ML requires autograd support, batch operations, and GPU kernels —
   none of which currently exist.

---

## 11. The padic-ds Repository — Architecture, API & Usage

### 11.1 Overview

| Field | Value |
|---|---|
| **Package** | `padic-ds` v0.1.1 |
| **Repository** | https://github.com/Mircus/padics |
| **Author** | HoloMathics / Mirco A. Mannucci |
| **License** | MIT |
| **Python** | ≥ 3.9 |
| **Core deps** | `numpy ≥ 1.23`, `scikit-learn ≥ 1.1` |
| **CI badge** | [![CI](https://github.com/Mircus/padics/actions/workflows/ci.yml/badge.svg)](https://github.com/Mircus/padics/actions/workflows/ci.yml) |
| **Status** | Alpha — reference implementation |

The repository is a pure-Python, dependency-light library giving data
scientists and researchers a working substrate for p-adic experimentation.
Everything is built from first principles: no compiled C extensions, no
external computer-algebra systems.

### 11.2 Installation

```bash
# Core (numpy + scikit-learn only)
pip install -e .

# With visualisation (matplotlib)
pip install -e ".[viz]"

# Developer tools (pytest, ruff, mypy, hypothesis)
pip install -e ".[dev]"

# Jupyter notebooks
pip install -e ".[notebook]"

# MkDocs documentation site
pip install -e ".[docs]"
```

**Conda/mamba**:

```bash
conda env create -f environment.yml
conda activate padic-ds
```

### 11.3 Repository Layout

```
padic-ds/
├── src/
│   └── padic/
│       ├── __init__.py         # Full public API surface
│       ├── field.py            # QpContext, Qp — core p-adic arithmetic
│       ├── ball.py             # QpBall — clopen balls and refinement
│       ├── hensel.py           # Hensel lifting (linear variant)
│       ├── btree.py            # BT-tree distances, two digit representations
│       ├── metrics.py          # padic_abs, padic_dist, pairwise matrices
│       ├── knn.py              # PadicKNNClassifier, embed_float_array
│       └── hclust.py           # ultrametric_dendrogram
├── tests/
│   └── test_field.py           # ~45 pytest tests covering all modules
├── notebooks/
│   ├── 03_padic_basics.ipynb   # Q_p arithmetic, balls, Hensel lifting
│   └── 04_ultrametric_ml.ipynb # BT digits, dendrograms, p-adic kNN
├── docs/
│   └── whitepaper.md           # This document
├── pyproject.toml              # Build config, optional deps, tool config
├── README.md
└── LICENSE                     # MIT
```

### 11.4 Module Reference

---

#### `field.py` — Core Arithmetic

The arithmetic backbone. Implements `QpContext` (immutable computation
context) and `Qp` (finite-precision p-adic number).

**`QpContext(p: int, prec: int)`**

Immutable dataclass pinning the prime and working precision. All `Qp` elements
are created relative to a context. Creating elements from different contexts
and mixing them raises `ValueError`.

```python
from padic import QpContext
ctx = QpContext(p=5, prec=8)   # 5-adics, 8 significant digits
ctx.modulus()                   # 5^8 = 390625 — working modulus
```

**`Qp`** — Finite-precision p-adic number stored as `x = u_mod · p^v`.

| Constructor | Description |
|---|---|
| `Qp.zero(ctx)` | Additive identity (sentinel v = 10⁹, u_mod = 0) |
| `Qp.from_int(ctx, n)` | Embed integer n; raises `PrecisionError` if v_p(n) ≥ prec |
| `Qp.from_rational(ctx, num, den)` | Embed rational num/den |

| Method | Description |
|---|---|
| `.add(other)` | Exact addition modulo p^prec |
| `.sub(other)` | Subtraction |
| `.mul(other)` | Multiplication (valuations add) |
| `.neg()` | Additive inverse |
| `.inv()` | Multiplicative inverse; raises `ZeroDivisionError` for zero |
| `.div(other)` | Division = `self.mul(other.inv())` |
| `.val()` | v_p(self); sentinel 10⁹ for zero |
| `.abs()` | |self|_p = p^{−v} as float; 0.0 for zero |
| `.digits(depth)` | Base-p digits of the unit part u_mod |
| `.is_zero()` | True iff u_mod == 0 |

**Precision semantics**:
- `Qp.from_int(ctx, n)` raises `PrecisionError` if v_p(n) ≥ prec (all
  significant digits would be lost). This prevents silent zero promotion.
- The zero sentinel uses v = 10⁹ to represent v_p(0) = +∞.
- Addition truncates naturally; information lost at the low end is
  irrecoverable.

```python
from padic import QpContext, Qp, padic_dist

ctx = QpContext(p=5, prec=8)
x = Qp.from_rational(ctx, 7, 12)   # 7/12 in Q_5
y = Qp.from_int(ctx, 10)            # 10 in Q_5
print(x.val())                       # v_5(7/12) = 0
print(padic_dist(x, y))              # |7/12 − 10|_5
```

---

#### `ball.py` — Ultrametric Balls

**`QpBall(center: Qp, n: int)`** — Closed ball B(center, p^{−n}).

```python
from padic import QpContext, Qp, QpBall

ctx = QpContext(p=3, prec=6)
center = Qp.from_int(ctx, 4)
B = QpBall(center, n=2)              # B(4, 3^{-2}), radius = 1/9
assert B.contains(Qp.from_int(ctx, 13))  # 13 ≡ 4 (mod 9) ✓

children = B.refine()                # 3 disjoint sub-balls of radius 3^{-3}
assert len(children) == 3            # always exactly p children
```

| Method | Description |
|---|---|
| `.contains(x)` | True iff d_p(x, center) ≤ p^{−n}, i.e. v_p(x − center) ≥ n |
| `.intersect(other)` | Returns the smaller ball if they overlap; `None` if disjoint |
| `.refine()` | Split into p disjoint sub-balls of radius p^{-(n+1)} |

The ultrametric property guarantees that `refine()` produces exactly p
**disjoint** sub-balls whose union equals the parent ball — the algebraic
analogue of binary splitting in a k-d tree, with zero overlap and zero orphaned
points.

---

#### `hensel.py` — Hensel Lifting

```python
from padic import QpContext, hensel_lift_simple

ctx = QpContext(5, prec=8)
# f(x) = x^2 - 6 over Z_5; a0=1 since 1^2 - 6 = -5 ≡ 0 (mod 5)
root = hensel_lift_simple(
    ctx,
    fZ=lambda t: t * t - 6,
    fZprime=lambda t: 2 * t,
    a0_mod_p=1,
    target_prec=8,
)
check = root.mul(root).sub(Qp.from_int(ctx, 6))
assert check.is_zero() or check.val() >= ctx.prec  # exact root in Z_5
```

`hensel_lift_simple` validates both Hensel preconditions at mod p (root
existence, non-zero derivative) with descriptive `ValueError` messages. The
lift is *linear*: one digit of precision per step, O(prec) total steps.

---

#### `btree.py` — Bruhat–Tits Tree Distances

Two digit encodings are provided:

| Function | Encodes valuation? | Intended use |
|---|---|---|
| `digits_p_adic(x)` | No — unit digits only | Pattern matching within units |
| `digits_with_valuation(x)` | Yes — leading zeros encode v | Faithful p-adic metric |

Two distance variants:

```python
from padic import QpContext, Qp, bt_distance, bt_distance_full

ctx = QpContext(5, prec=6)
x  = Qp.from_int(ctx, 3)
px = Qp.from_int(ctx, 15)     # 3 × 5 — differs by one power of 5

# Unit-only surrogate — KNOWN LIMITATION: ignores valuation
print(bt_distance(ctx, x, px))       # 0  (same unit digits)

# Valuation-aware — faithful to the p-adic metric
print(bt_distance_full(ctx, x, px))  # > 0 ✓
```

`BTRootedTree` wraps these as `.dist()` (unit-only) and `.dist_full()`
(valuation-aware). Use `bt_distance_full` / `.dist_full()` unless you have
a specific reason to ignore valuation.

---

#### `metrics.py` — Distance Utilities

```python
from padic import QpContext, Qp, pairwise_padic_dist, pairwise_padic_dist_vec

ctx = QpContext(3, prec=6)

# Scalar points — n×n distance matrix
pts = [Qp.from_int(ctx, k) for k in [1, 3, 9, 27]]
D = pairwise_padic_dist(pts)     # shape (4, 4), dtype float64

# Vector points — product ultrametric (max over coordinates)
vecs = [[Qp.from_int(ctx, a), Qp.from_int(ctx, b)]
        for a, b in [(1, 2), (3, 6), (9, 18)]]
Dv = pairwise_padic_dist_vec(vecs)  # shape (3, 3)
```

---

#### `knn.py` — Ultrametric k-NN Classifier

`PadicKNNClassifier` follows the scikit-learn estimator protocol:
`fit` / `predict` / `predict_proba` / `get_params` / `set_params`.

```python
from padic import QpContext, Qp, PadicKNNClassifier
import numpy as np

ctx = QpContext(3, prec=6)
X = [[Qp.from_int(ctx, n)] for n in [1, 4, 10, 13, 40, 121]]
y = np.array([0, 0, 0, 1, 1, 1])

clf = PadicKNNClassifier(ctx, k=3).fit(X, y)
print(clf.predict(X))              # array of predicted labels
print(clf.predict_proba(X))        # shape (6, 2), rows sum to 1.0
```

Distance used: product ultrametric `max_j d_p(u_j, v_j)` over Q_p^d.
Tie-breaking in `predict`: smallest total distance among tied classes.

**`embed_float_array(X, ctx, scale=100)`** converts a NumPy float array of
shape (n, d) into `List[List[Qp]]` via `round(value * scale)` →
`Qp.from_int`. The `scale` parameter controls resolution; higher scale
requires higher `ctx.prec` to avoid `PrecisionError`.

---

#### `hclust.py` — Ultrametric Dendrogram

```python
from padic import QpContext, Qp, ultrametric_dendrogram

ctx = QpContext(3, prec=6)
X = [Qp.from_int(ctx, k) for k in [1, 3, 9, 27, 2, 6]]
H = ultrametric_dendrogram(ctx, X)   # shape (6, 6), dtype int
# H[i,j] = ctx.prec − lca_depth(digits(X[i]), digits(X[j]))
# H satisfies the ultrametric inequality exactly, not approximately
```

The output matrix `H` is an exact ultrametric — not an approximation as in
Ward or average-linkage clustering on Euclidean data. This makes it suitable
as a ground-truth distance matrix for evaluating hierarchical structure
preservation in ML experiments.

---

### 11.5 Public API Surface

All exports are gathered in `src/padic/__init__.py`:

```python
from padic import (
    # Core arithmetic
    QpContext, Qp, PrecisionError,
    # Balls / ultrametric geometry
    QpBall,
    # Hensel lifting
    hensel_lift_simple,
    # Bruhat-Tits tree
    BTRootedTree, bt_distance, bt_distance_full,
    lca_depth, digits_p_adic, digits_with_valuation,
    # Distance utilities
    padic_abs, padic_dist, pairwise_padic_dist, pairwise_padic_dist_vec,
    # ML estimators
    PadicKNNClassifier, embed_float_array, ultrametric_dendrogram,
)
```

### 11.6 Test Suite

`tests/test_field.py` contains approximately 45 pytest tests organised into
9 sections:

| Section | Tests |
|---|---|
| Construction | `from_int`, `from_rational`, zero, `PrecisionError` |
| Equality & hashing | Reflexivity, zero equality, context mismatch, set membership |
| Arithmetic | Commutativity, associativity, negative valuations, cancellation, context mismatch errors |
| Ultrametric inequality | `d(x,z) ≤ max(d(x,y), d(y,z))` for 5 parametrised triples |
| `QpBall` | Contains, refine covers parent, refine shifts, intersect, disjoint |
| Hensel lifting | √2 in Z_7, √6 in Z_5, bad initial value, bad derivative |
| BT distances | Unit-only limitation, valuation-aware correctness, `digits_with_valuation` |
| Pairwise matrix | Symmetry, ultrametric, shape |
| `PadicKNNClassifier` | Fit/predict, predict_proba shape, context mismatch, not-fitted error |

Run tests:

```bash
cd padic-ds
pip install -e ".[dev]"
pytest --tb=short          # run all tests
pytest --cov=src/padic     # with coverage report
```

### 11.7 Notebooks

| Notebook | Key demonstrations |
|---|---|
| `notebooks/03_padic_basics.ipynb` | Q_p arithmetic; ultrametric inequality visualisation; ball containment and refinement; Hensel lifting √6 in Z_5 |
| `notebooks/04_ultrametric_ml.ipynb` | BT digit sequences; exact ultrametric dendrogram vs. Ward linkage; p-adic kNN on hierarchical toy data |

### 11.8 API Status & Roadmap

| Component | Status |
|---|---|
| `Qp` arithmetic (add, sub, mul, inv, div) | ✅ Implemented |
| `QpBall` (contains, intersect, refine) | ✅ Implemented |
| Hensel lifting (linear) | ✅ Implemented |
| BT tree distance (unit-only, `bt_distance`) | ✅ Implemented |
| BT tree distance (valuation-aware, `bt_distance_full`) | ✅ Implemented |
| Pairwise distance matrices (scalar and vector) | ✅ Implemented |
| kNN classifier (predict, predict_proba) | ✅ Implemented |
| Ultrametric dendrogram | ✅ Implemented |
| Ball-tree / p-ary trie index for ANN | 🗺 Near-term roadmap |
| Real ℝ^d → Q_p^d embedding module (`PadicEmbed`) | 🗺 Near-term roadmap |
| Quadratic Newton–Hensel lifting | 🗺 Near-term roadmap |
| p-adic Haar wavelet transform | 🔬 Research prospect |
| PGL₂(Q_p)-equivariant architecture | 🔬 Research prospect |
| Autograd / differentiable Q_p operations | 🔬 Research prospect |

---

## 12. The Landscape of p-Adic AI Research — An Annotated Survey

This section surveys the research connecting p-adic mathematics to machine
learning and AI. The field is young but growing rapidly; papers published
since 2022 show a clear shift from purely theoretical explorations to
experimental results on standard benchmarks.

The literature is organised into six thematic clusters.

---

### 12.1 Foundational Mathematics

**[R1] Schikhof, W.H. (1984).** *Ultrametric Calculus*. Cambridge University
Press.  
The standard reference for non-Archimedean analysis. Covers Q_p topology,
completeness, Hensel's lemma, and the structure of ultrametric spaces.
Essential reading before any serious p-adic AI implementation.

**[R2] Dress, A.W.M. (1984).** Trees, tight extensions of metric spaces, and
the cohomological dimension of certain groups. *Advances in Mathematics* 53,
321–402.  
Proves that tree metrics embed isometrically into L¹ — the result that
establishes trees as fundamentally non-L² objects, motivating the search for
non-Euclidean alternatives to Euclidean embedding.

**[R3] Serre, J.-P. (1980).** *Trees*. Springer-Verlag.  
The definitive treatment of Bruhat–Tits trees for PGL₂(Q_p) and SL₂(Q_p).
Chapter II establishes the automorphism group as PGL₂(Q_p) — the symmetry
group that any group-equivariant p-adic architecture must respect.

**[R4] Robert, A.M. (2000).** *A Course in p-Adic Analysis*. Springer GTM 198.  
Accessible graduate introduction to Q_p, Z_p, the p-adic exponential and
logarithm, and Mahler series. The practical complement to Schikhof's treatise.

**[R5] Kozyrev, S.V. (2002).** Wavelet theory as p-adic spectral analysis.
*Izvestiya Mathematics* 66(2), 367–376.  
Constructs the canonical orthonormal wavelet basis on Q_p whose support sets
are exactly the clopen balls B(a, p^{−n}). The mathematical foundation for
the p-adic wavelet architecture described in §5.

**[R+] Bourgain, J. (1985).** On Lipschitz embedding of finite metric spaces in
Hilbert space. *Israel Journal of Mathematics* 52(1–2), 46–52.  
General O(log n) distortion bound for arbitrary metric → L² embeddings.

**[R+] Matousek, J. (1999).** On the distortion required for embedding finite
metric spaces into normed spaces. *Israel Journal of Mathematics* 93(1), 333–344.  
Tree metric → L² lower bound Ω(√log n), essentially tight.

---

### 12.2 Early AI and Neural Network Work

**[R6] Khrennikov, A.Yu. (1994).** *p-Adic Valued Distributions in
Mathematical Physics*. Kluwer Academic Publishers.  
First systematic treatment of p-adic probability and stochastic processes with
an eye toward physics and cognition. The intellectual starting point for all
p-adic neural network work.

**[R7] Khrennikov, A.Yu. (1997).** *Non-Archimedean Analysis: Quantum
Paradoxes, Dynamical Systems and Biological Models*. Kluwer.  
Argues that p-adic analysis is a natural framework for hierarchical cognitive
and biological structures — a prescient observation in light of contemporary
hierarchical deep learning.

**[R8] Khrennikov, A.Yu. & Kotovich, N.V. (2002).** Learning of p-adic neural
networks. *p-Adic Numbers, Ultrametric Analysis and Applications*.  
URL: https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/khrennikov-learning.pdf  
Early investigation of gradient-like learning rules for weights in Q_p. The
central difficulty — defining a useful gradient in a non-Archimedean field —
is identified here and remains unsolved in general (see VAPO [R19] for the
most recent attempt).

**[R9] Murtagh, F. (2004).** On ultrametric algorithmic information.
*Computer Journal* 47(4).  
Shows that hierarchical (ultrametric) cluster structures are related to data
compression and Kolmogorov complexity. Practical algorithmic motivation for
ultrametric clustering as an information-theoretic concept.

---

### 12.3 Hyperbolic and Ultrametric Geometry in ML (the neighbouring field)

These works use hyperbolic space rather than Q_p, but the motivation is the
same — tree-like structures require non-Euclidean geometry. They provide both
intellectual context and empirical benchmarks that p-adic methods must
eventually meet or surpass.

**[R10] Nickel, M. & Kiela, D. (2017).** Poincaré Embeddings for Learning
Hierarchical Representations. *NeurIPS 2017*. arXiv:1705.08039.  
The landmark paper demonstrating that hyperbolic (Poincaré disk) embeddings
outperform Euclidean embeddings on hierarchical datasets (WordNet) by an order
of magnitude in distortion. Popularised non-Euclidean representation learning.
p-adic methods aim at the same class of problems but with an exact rather than
approximate geometry. The v-PuNNs paper [R19] directly benchmarks against
Poincaré embeddings, with superior results.

**[R11] Ganea, O., Bécigneul, G. & Hofmann, T. (2018).** Hyperbolic Neural
Networks. *NeurIPS 2018*. arXiv:1805.09112.  
Extends Poincaré embeddings to full neural network architectures (feedforward
layers, RNNs, attention) via Riemannian gradient descent. The first
demonstration that hyperbolic geometry can be used end-to-end in training —
the architectural template that p-adic networks [R19] must match and have now
begun to surpass on discrete hierarchical tasks.

**[R12] Sala, F., De Sa, C., Gu, A. & Ré, C. (2018).** Representation
tradeoffs for hyperbolic embeddings. *ICML 2018*. arXiv:1804.03329.  
Quantifies the tradeoff between dimensionality and distortion in hyperbolic
vs. Euclidean embeddings of trees. Shows that a single dimension of hyperbolic
space can embed tree hierarchies that require O(n) Euclidean dimensions.
The analogous analysis for Q_p vs. the Poincaré disk is an open question.

**[R13] (2024).** Uncovering Hierarchical Structure in LLM Embeddings with
δ-Hyperbolicity, Ultrametricity, and Neighbor Joining. arXiv:2512.20926.  
Applies δ-hyperbolicity and ultrametricity measures to embeddings from large
language models. Finds significant hierarchical (non-Euclidean) structure in
LLM representation spaces — providing empirical evidence that production-scale
models implicitly learn p-adic-like organisation, even without explicit
non-Euclidean inductive bias.

---

### 12.4 p-Adic Statistical Field Theories and Deep Networks

This line of research, led primarily by W. A. Zúñiga-Galindo, establishes
rigorous correspondences between p-adic physics and deep learning architectures.

**[R14] Zúñiga-Galindo, W.A. (2022/2023).** p-Adic Statistical Field Theory
and Deep Belief Networks. *Physica A: Statistical Mechanics and its
Applications* 622 (2023). arXiv:2207.13877.  
Shows that p-adic statistical field theories (SFTs) on ultrametric trees
correspond exactly to deep belief networks (DBNs). A p-adic continuous SFT
corresponds to a continuous p-adic DBN; its discretisation corresponds to a
discrete p-adic DBN that is a universal approximator. The parameter count for
p-adic convolutional DBMs is significantly smaller than for conventional ones.

**[R15] Zúñiga-Galindo, W.A. et al. (2023).** p-Adic Statistical Field Theory
and Convolutional Deep Boltzmann Machines. *Progress of Theoretical and
Experimental Physics* 2023(6). arXiv:2302.03817.  
Extension of [R14] to convolutional architectures. The p-adic tree structure
of the convolution kernel means the network automatically inherits a
multi-scale inductive bias without explicit design.

**[R16] Zúñiga-Galindo, W.A., Zambrano-Luna, B.A. & Dibba, B. (2024).**
Hierarchical Neural Networks, p-Adic PDEs, and Applications to Image
Processing. *Journal of Nonlinear Mathematical Physics* 31:63. arXiv:2406.07790.
DOI: 10.1007/s44198-024-00229-6.  
Introduces p-adic reaction–diffusion cellular neural networks (CNNs) with
delay, studies their stability, and applies them to image denoising. Shows
that p-adic CNNs are hierarchical generalisations of the classical Chua–Yang
CNNs with competitive performance and far fewer parameters.

**[R17] Oliva, G., Torchiani, C., Vanzella, W. & Zanchetta, M. (2022).**
p-Adic Cellular Neural Networks. *Journal of Nonlinear Mathematical Physics*
(2022). arXiv:2209.03197.  
Introduces the first p-adic CNN where neurons are supported on p-adic balls.
Demonstrates applications to image segmentation using the ultrametric ball
structure as the convolutional receptive field.

---

### 12.5 Architecture Innovations — Learnable p-Adic Representations

**[R18] (December 2025).** Learning with the p-adics. arXiv:2512.22692.  
Studies the suitability of Q_p as a substrate for machine learning, covering
linear models, kernel methods, and representation learning in a unified
framework. Identifies the hierarchical string interpretation of Q_p as the
key advantage for code theory and hierarchical classification. One of the
first papers to give a systematic ML-oriented treatment of the full field.

**[R19] (2025).** v-PuNNs: van der Put Neural Networks for Transparent
Ultrametric Representation Learning. arXiv:2508.01010.  
*The most architecturally complete p-adic neural network to date — and the
most important recent paper for the padic-ds roadmap.*

Neurons are characteristic functions of p-adic balls; all weights are
themselves p-adic numbers under the Transparent Ultrametric Representation
Learning (TURL) principle, giving exact subtree semantics. Key contributions:

- **Finite Hierarchical Approximation Theorem**: a depth-K v-PuNN with the
  appropriate neuron count universally represents any K-level tree.
- **VAPO (Valuation-Adaptive Perturbation Optimization)**: a gradient-free
  optimiser for discrete Q_p space, with fast deterministic and moment-based
  variants.
- **Benchmark results (CPU-only implementation)**:
  - 99.96% leaf accuracy on WordNet nouns (52,427 leaves)
  - 100% accuracy on gene ontology (27,000+ proteins)
  - Outperforms Poincaré embeddings [R10] on their own benchmarks.

**[R20] (2026).** p-Adic Character Neural Network. arXiv:2603.29905.  
Defines neural networks using p-adic character functions (additive characters
of Q_p) as activation. Establishes universal approximation theorems in p-adic
function spaces. The character-based approach avoids gradient vanishing by
working in the Fourier (spectral) domain of Q_p.

**[R21] (2026).** Minimal Width of Universal p-Adic ReLU Neural Networks.
arXiv:2603.00064.  
Proves tight bounds on the minimum number of neurons in a p-adic ReLU network
required for universal approximation. Establishes that p-adic networks can be
more parameter-efficient than real-valued counterparts for functions with
ultrametric symmetry — analogous to the classical ReLU width theorems but
sharper for hierarchically symmetric functions.

---

### 12.6 Regression and Analysis in p-Adic Spaces

**[R22] (September 2025).** Linear Regression in p-Adic Metric Spaces.
arXiv:2510.00043.  
Proves that in p-adic metric spaces the hyperplane minimising the p-adic sum
of distances to n+1 or more points must pass through at least n+1 of those
points — a sharp discrete-geometry result with no Euclidean analogue. Gives
a theoretical foundation for p-adic linear regression and supervised learning
in non-Archimedean spaces.

**[R23] Parisi, G. & Sourlas, N. (1982).** p-Adic numbers and replica symmetry
breaking. *European Physical Journal B* 14, 535–542.  
Theoretical physics motivation: the mean-field theory of spin glasses on
ultrametric trees leads naturally to p-adic analysis. Historically important
for connecting ultrametric geometry to learning-like processes in complex
disordered systems.

**[R24] Gubser, S.S. et al. (2017).** p-Adic AdS/CFT. *Communications in
Mathematical Physics* 352(3), 1019–1059.  
The p-adic analogue of the AdS/CFT holographic duality. While not directly
an AI result, it demonstrates that Q_p is the natural arena for
information-theoretic problems on trees — the same structural argument
underlying all AI advantages described in this brief.

---

### 12.7 Summary Table

| Ref | Year | Venue | Key Contribution | AI relevance |
|---|---|---|---|---|
| [R3] Serre | 1980 | Springer book | BT tree automorphism group = PGL₂(Q_p) | Equivariant architectures |
| [R2] Dress | 1984 | Adv. Math. | Trees embed isometrically in L¹, not L² | Motivation for non-Euclidean geometry |
| [R5] Kozyrev | 2002 | Izv. Math. | Canonical p-adic wavelet basis | Multi-scale feature learning |
| [R8] Khrennikov | 2002 | p-Adic UAA | Learning rules in Q_p | Foundational NN work |
| [R9] Murtagh | 2004 | Comp. J. | Ultrametric information theory | Clustering foundations |
| [R10] Nickel-Kiela | 2017 | NeurIPS | Poincaré embeddings SOTA on WordNet | Tree embeddings benchmark |
| [R11] Ganea et al. | 2018 | NeurIPS | End-to-end hyperbolic NNs | Non-Euclidean training template |
| [R14] Zúñiga-Galindo | 2023 | Physica A | p-adic SFT ↔ DBNs (theoretical) | Formal NN correspondence |
| [R16] Zúñiga-Galindo | 2024 | JNMP | p-adic PDE CNNs, image denoising | Applied hierarchical CNNs |
| [R13] LLM survey | 2024 | arXiv | Ultrametricity in LLM embeddings | Empirical motivation for p-adics in LLMs |
| [R18] Learning p-adics | 2025 | arXiv | Systematic ML in Q_p | Broad ML foundations |
| [R19] v-PuNNs | 2025 | arXiv | Full learnable p-adic architecture, VAPO, SOTA | **Highest priority for padic-ds roadmap** |
| [R22] p-adic regression | 2025 | arXiv | Regression geometry in Q_p | Statistical learning theory |
| [R20] char. NNs | 2026 | arXiv | Character-based activation, UAT | Universal approximation |
| [R21] min-width ReLU | 2026 | arXiv | Width bounds for p-adic NNs | Architecture efficiency theory |

---

## 13. Conclusion

p-Adic numbers are not a curiosity for number theorists. Their properties —
the ultrametric inequality, the canonical tree structure, the exact arithmetic,
the rich symmetry group, the tractable index — map directly onto the structural
demands of advanced AI: hierarchical reasoning, exact symbolic computation,
efficient retrieval, and principled multi-scale representation.

The argument is not that p-adics should replace Euclidean geometry in AI, but
that they should be available as a *first-class primitive* alongside it. Tasks
that are fundamentally hierarchical or algebraic should be solved in a geometry
that is fundamentally hierarchical and algebraic.

**Empirical progress is accelerating.** The v-PuNNs paper [R19] (2025)
achieves 99.96% accuracy on 52,427-leaf WordNet classification *on a CPU-only
implementation*, beating Poincaré embeddings on their own benchmark. The 2026
papers on character networks [R20] and width bounds [R21] suggest a maturing
theoretical foundation analogous to the universal approximation theorems for
ReLU networks. Recent work [R13] measuring ultrametricity in LLM embeddings
provides empirical evidence that production models already implicitly learn
p-adic-like organisation at scale.

The `padic-ds` library provides a clean, tested implementation of the core
objects (`Qp`, `QpBall`, `BTRootedTree`, `PadicKNNClassifier`,
`ultrametric_dendrogram`) that makes it possible to run controlled experiments
comparing p-adic and Euclidean representations on the same data.

**Immediate next steps:**

(a) A BT p-ary trie index enabling end-to-end retrieval benchmarks;  
(b) A real → Q_p^d embedding module (`PadicEmbed`) with stability guarantees;  
(c) Newton–Hensel quadratic lifting (O(log prec) steps to full precision);  
(d) A p-adic Haar transform prototype;  
(e) Experimental comparison with v-PuNNs [R19] on WordNet and gene ontology.

The research prospects — wavelet architectures and PGL₂(Q_p)-equivariant
networks — depend on these foundations but have now moved from pure speculation
to near-term engineering given the empirical successes of 2025–2026.

---

## References

### A. Core Mathematics

[R1] Schikhof, W.H. (1984). *Ultrametric Calculus*. Cambridge University Press.  
[R2] Dress, A.W.M. (1984). Trees, tight extensions of metric spaces, and the cohomological dimension of certain groups. *Advances in Mathematics* 53, 321–402.  
[R3] Serre, J.-P. (1980). *Trees*. Springer-Verlag.  
[R4] Robert, A.M. (2000). *A Course in p-Adic Analysis*. Springer GTM 198.  
[R5] Kozyrev, S.V. (2002). Wavelet theory as p-adic spectral analysis. *Izvestiya Mathematics* 66(2), 367–376.  
[R+] Bourgain, J. (1985). On Lipschitz embedding of finite metric spaces in Hilbert space. *Israel Journal of Mathematics* 52(1–2), 46–52.  
[R+] Matousek, J. (1999). On the distortion required for embedding finite metric spaces into normed spaces. *Israel Journal of Mathematics* 93(1), 333–344.

### B. Early AI / Neural Network Work

[R6] Khrennikov, A.Yu. (1994). *p-Adic Valued Distributions in Mathematical Physics*. Kluwer Academic Publishers.  
[R7] Khrennikov, A.Yu. (1997). *Non-Archimedean Analysis: Quantum Paradoxes, Dynamical Systems and Biological Models*. Kluwer.  
[R8] Khrennikov, A.Yu. & Kotovich, N.V. (2002). Learning of p-adic neural networks. *p-Adic Numbers, Ultrametric Analysis and Applications*. URL: https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/khrennikov-learning.pdf  
[R9] Murtagh, F. (2004). On ultrametric algorithmic information. *Computer Journal* 47(4).

### C. Hyperbolic and Non-Euclidean ML

[R10] Nickel, M. & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. *NeurIPS 2017*. arXiv:1705.08039.  
[R11] Ganea, O., Bécigneul, G. & Hofmann, T. (2018). Hyperbolic Neural Networks. *NeurIPS 2018*. arXiv:1805.09112.  
[R12] Sala, F., De Sa, C., Gu, A. & Ré, C. (2018). Representation tradeoffs for hyperbolic embeddings. *ICML 2018*. arXiv:1804.03329.  
[R13] (2024). Uncovering Hierarchical Structure in LLM Embeddings with δ-Hyperbolicity, Ultrametricity, and Neighbor Joining. arXiv:2512.20926.

### D. p-Adic Statistical Field Theories and Deep Networks

[R14] Zúñiga-Galindo, W.A. (2023). p-Adic Statistical Field Theory and Deep Belief Networks. *Physica A* 622. arXiv:2207.13877.  
[R15] Zúñiga-Galindo, W.A. et al. (2023). p-Adic Statistical Field Theory and Convolutional Deep Boltzmann Machines. *Progress of Theoretical and Experimental Physics* 2023(6). arXiv:2302.03817.  
[R16] Zúñiga-Galindo, W.A., Zambrano-Luna, B.A. & Dibba, B. (2024). Hierarchical Neural Networks, p-Adic PDEs, and Applications to Image Processing. *Journal of Nonlinear Mathematical Physics* 31:63. DOI: 10.1007/s44198-024-00229-6. arXiv:2406.07790.  
[R17] Oliva, G., Torchiani, C., Vanzella, W. & Zanchetta, M. (2022). p-Adic Cellular Neural Networks. *Journal of Nonlinear Mathematical Physics*. arXiv:2209.03197.

### E. Architecture Innovations

[R18] (2025). Learning with the p-adics. arXiv:2512.22692.  
[R19] (2025). v-PuNNs: van der Put Neural Networks for Transparent Ultrametric Representation Learning. arXiv:2508.01010.  
[R20] (2026). p-Adic Character Neural Network. arXiv:2603.29905.  
[R21] (2026). The minimal width of universal p-adic ReLU neural networks. arXiv:2603.00064.

### F. Regression, Physics, and Related Work

[R22] (2025). Linear Regression in p-adic metric spaces. arXiv:2510.00043.  
[R23] Parisi, G. & Sourlas, N. (1982). p-Adic numbers and replica symmetry breaking. *European Physical Journal B* 14, 535–542.  
[R24] Gubser, S.S. et al. (2017). p-Adic AdS/CFT. *Communications in Mathematical Physics* 352(3), 1019–1059.

### G. Reference Implementation

[R25] padic-ds v0.1.1 (2025–2026). Reference implementation of p-adic data structures for machine learning. MIT License. HoloMathics / Mirco A. Mannucci. https://github.com/Mircus/padics

---

*Prepared for the padic-ds project (proj-bba42f98) · Phase: active-dev*  
*v3: expanded with full repository guide (§11), annotated literature survey (§12),*  
*comprehensive bibliography (25 references across 7 thematic groups).*  
*Code synced to padic-ds v0.1.1. arXiv citations verified May 2026.*
