# The Advantages of p-Adic Numbers for Advanced AI

**Technical Brief · padic-ds Project · May 2026 · v2 (revised)**

---

## Abstract

Modern machine learning rests almost entirely on Euclidean geometry and its
derivatives — dot products, L² distances, gradient flows in ℝⁿ.  Yet many of
the hardest problems in AI — hierarchical reasoning, symbolic structure,
compositional generalisation, robustness to adversarial perturbation — are
fundamentally *non-Euclidean* in character.  p-Adic numbers, the unique
completions of ℚ with respect to the *p-adic absolute value*, offer a
mathematically rigorous alternative geometry whose properties align
surprisingly well with the structural demands of advanced AI.  This brief
documents seven concrete advantages of the p-adic framework for AI, grounds
each in the mathematics and in the `padic-ds` reference implementation, and
charts the directions most likely to yield practical gains.

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
factorisation of x.  The field ℚ_p is the completion of ℚ under this metric,
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
prefixes in a base-p digit expansion.  The `padic-ds` library (`src/padic/`)
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
metrics and is not tight for trees.  For tree metrics specifically, the
situation is more nuanced:

* Any finite tree embeds **isometrically** into L¹ (Dress 1984; the
  four-point condition for tree metrics coincides with L¹ hyperbolicity).
* The same trees require **Ω(√log n) distortion** when embedded into L²
  (Matousek 1999), and this lower bound is essentially tight.

In practice, learning hierarchical knowledge — taxonomies, parse trees,
ontologies, class hierarchies — in Euclidean space still demands either heavy
over-parameterisation or an explicit architectural inductive bias (e.g.,
hyperbolic embeddings, tree-structured LSTMs).

### p-Adics as a Natural Tree Metric

In Q_p, the ultrametric inequality *forces* the geometry to be tree-like.
Every element of Z_p has a canonical base-p digit expansion
a₀ + a₁p + a₂p² + …, and the p-adic distance between two elements equals
p^{−k} where k is the position of their first differing digit.  This is
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
approximation of it.  Embedding a taxonomy of depth D into Z_p at precision D
is lossless.  For AI tasks built on hierarchical structures — WordNet
embeddings, biological taxonomies, code call-graphs, compositional grammar —
this eliminates an entire class of representational error.

---

## 3. Advantage 2 — The Ultrametric Mitigates Crowding in kNN  [Implemented]

### Curse of Dimensionality in Standard Metrics

High-dimensional Euclidean spaces suffer from *concentration of measure*:
pairwise distances converge to the same value, making kNN and clustering
degenerate.  This is a primary reason why nearest-neighbour methods fail in
raw feature spaces and require dimension reduction or manifold learning as a
preprocessing step.

### Why Ultrametrics Help (with Caveats)

In an ultrametric space, the strong triangle inequality — d(x,z) ≤
max{d(x,y), d(y,z)} — means any two balls are either nested or disjoint, with
no ambiguous border regions.  Crucially, the **distance spectrum is
discrete**: in Q_p, pairwise distances can only take values in
{0, 1, p, p², p³, …}, not a continuum.  This discretisation keeps kNN
decision boundaries crisp even as the number of points grows.

**Important qualification**: This dimension-independence argument applies to
the *single-coordinate* p-adic metric.  For product spaces Q_p^d equipped
with the sup-norm max_i d_p(x_i, y_i), each ball of radius p^{-k} refines
into **p^d** sub-balls at the next finer resolution.  The branching factor
grows exponentially with d, just as in Euclidean space.  The mitigation of
crowding comes not from dimension-independence per se, but from the **quantized
distance spectrum**: near-ties cluster at the same exact value p^{-k} rather
than forming a diffuse cloud, so rank-ordering of neighbours remains stable
under perturbations that stay below the next level p^{-(k+1)}.

The `ultrametric_dendrogram` in `hclust.py` builds an exact ultrametric matrix
from LCA depths — a property that Euclidean linkage can only approximate:

```python
# from hclust.py — implemented in padic-ds
def ultrametric_dendrogram(ctx, X):
    D = [digits_p_adic(x, ctx.prec) for x in X]
    n = len(X)
    H = np.zeros((n, n), dtype=int)
    for i, j in pairs:
        H[i,j] = H[j,i] = ctx.prec - lca_depth(D[i], D[j])
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
approximations that converge asymptotically.  Guarantees of convergence to
exact solutions (rather than approximate critical points) are rare and
fragile.

### Hensel's Lemma as Principled Lifting

Hensel's lemma guarantees that if f(a) ≡ 0 (mod p) and f′(a) ≢ 0 (mod p),
then a lifts *uniquely* to a root of f in Z_p.  The lift is constructive.
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

Doubles precision each step; requires only O(log prec) iterations — the true
quadratic convergence analogy to Newton's method.

The current `padic-ds` implementation uses the **linear variant** (`mod *= p`
per iteration):

```python
# from hensel.py — linear lifting (one digit per step), O(prec) iterations
while mod < N:
    mod *= p                           # linear: +1 digit per iteration
    fa  = fZ(a) % mod
    fpa = fZprime(a) % mod
    inv = pow(fpa, -1, mod)            # exact modular inverse
    a   = (a + (-fa * inv) % mod) % mod   # exact update
```

Every step is exact modulo the current working precision, with no
floating-point accumulation.  Upgrading to Newton–Hensel (exponent-doubling
precision schedule) is a near-term roadmap item.

**AI relevance**: Problems that have algebraic solutions can be expressed as
polynomial equations over Z_p and solved to any desired precision without
floating-point error.  Applications include constraint satisfaction, modular
polynomial systems relevant to SAT/SMT, and formal verification assistants.
Benchmarks against float-based Newton on modular polynomial constraints are
planned.

---

## 5. Advantage 4 — p-Adic Wavelets and Multi-Scale Feature Learning  [Research Prospect]

> **Status**: No implementation in padic-ds. This section describes a
> theoretical opportunity; all claims are prospective.

### The Haar Wavelet on ℝ vs. Q_p

The Haar wavelet on ℝ is a useful but somewhat *ad hoc* construction.  On
Q_p, there exists a canonical orthonormal wavelet basis — the *p-adic Haar
wavelets* (Kozyrev 2002) — whose support sets are exactly the clopen balls
B(a, p^{−n}).  Because balls are clopen and pairwise disjoint, wavelet
coefficients at resolution p^{−n} are independent of those at p^{−m} for
m ≠ n: there is no inter-scale aliasing by construction.

### Implications for Neural Architectures

Multi-resolution analysis in ℝⁿ (CNNs, pooling, U-nets) must deal with
aliasing, edge artefacts, and stride hyperparameters.  In theory, Q_p offers:

* **Exact pooling**: coarsening from p^{-k} to p^{-(k-1)} resolution is
  algebraically exact.
* **Depth = algebraic scale**: a layer at depth k processes features at
  resolution p^{-k}, with a provably complete and non-redundant basis.

**Critical caveat — finite precision**: Practical computation occurs in
**Z/p^N Z** (the quotient ring at precision N), which has wrap-around
arithmetic.  The "no boundary effects" claim needs qualification: at finite
precision, convolution wraps around modulo p^N, analogous to periodic boundary
conditions on a torus.  Eliminating these artifacts requires either an
inverse-limit construction (increasing N as needed) or explicit padding
conventions.

A minimal viable prototype — a p-adic Haar transform for fixed-length
sequences with ball-based support, benchmarked on denoising or compression —
is the next step before stronger architectural claims can be made.

---

## 6. Advantage 5 — Symbolic Reasoning and Exact Arithmetic  [Implemented]

### Floating-Point Arithmetic Is Lossy

IEEE 754 floating-point arithmetic is approximate by design: rounding errors
accumulate, comparisons are unreliable (0.1 + 0.2 ≠ 0.3), and catastrophic
cancellation can invalidate intermediate results.  For AI systems that need to
reason symbolically — theorem provers, program synthesisers, formal
verification assistants — floating-point is a liability.

### p-Adic Arithmetic Is Exact at Fixed Precision

In Q_p at working precision N:

* Addition, subtraction, and multiplication are exact modulo p^N.
* Division by units (elements whose valuation is zero) is exact.
* The valuation is an integer, computable in O(log p N) time.

```python
# from field.py — exact unit-level arithmetic, no floating-point
def mul(self, other: "Qp") -> "Qp":
    v = self.v + other.v
    u = (self.u_mod * other.u_mod) % self.ctx.modulus()  # exact
    return Qp(self.ctx, v=v, u_mod=u)
```

The *p-adic integers* Z_p form a complete discrete valuation ring where
divisibility, factorisation, and root-finding have clean algebraic
characterisations.  AI systems operating over Z_p can reason about
number-theoretic properties *by type*, without numerical approximation.

**Finite precision artifacts**: At fixed N, cancellations that increase
valuation (e.g., x − y where x ≈ y mod p^k) consume precision and can
reduce effective kNN/classification resolution.  Robust implementations must
track precision loss and propagate it through multi-step computations.

This is relevant for:
* **Neuro-symbolic AI**: p-adic representations can carry exact arithmetic
  constraints that gradient-based learning preserves.
* **Cryptographic reasoning**: Many post-quantum cryptography (PQC) schemes
  (NTRU, Kyber, Dilithium) operate over polynomial rings Z/qZ[x] — quotients
  related to Z_p as an inverse limit of Z/p^kZ.  The connection is real but
  not direct: Z_p itself is a pro-p ring, while PQC rings typically use
  non-prime-power moduli.  An AI component that understands p-adic valuation
  arithmetic is a natural match for reasoning about these structures, but the
  mapping must be made precise per scheme.
* **Formal verification assistants**: Lean/Coq tactics that discharge
  number-theoretic goals benefit from an AI component that understands p-adic
  arithmetic intrinsically.

---

## 7. Advantage 6 — Group-Equivariant Learning on Tree-Structured Spaces  [Research Prospect]

> **Status**: No implementation in padic-ds. This section describes a
> theoretical opportunity; all claims are prospective pending experimental
> validation.

### Equivariance Is the Central Design Principle of Modern Architectures

CNNs exploit translation equivariance.  Graph neural networks exploit
permutation equivariance.  Transformers approximate permutation-equivariant
set functions.  In each case, *baking in the symmetry group reduces sample
complexity* by the effective size of the orbit.

### The Symmetry Group of the BT Tree

The automorphism group of the Bruhat–Tits tree T_p is **PGL₂(Q_p)** — the
group of 2×2 invertible matrices over Q_p modulo its centre (Serre 1980,
*Trees*, Ch. II).  Note: GL₂(Q_p) differs from PGL₂(Q_p) by the central
scalar subgroup Q_p^×; for tree automorphisms the correct group is PGL₂(Q_p).
This group includes:

* **Translations** by any element of Z_p (shifting the digit expansion).
* **Dilations** by powers of p (scaling the valuation).
* **Möbius transformations** over Q_p (the full group of tree automorphisms).

A neural architecture whose weight-sharing pattern respects PGL₂(Q_p) would
achieve equivariance to all of these simultaneously — a far richer symmetry
than the discrete integer translations of a standard CNN.

**Tempered claim**: p-adic group-equivariant networks trained on hierarchical
data (parse trees, ontologies, code ASTs) are *expected* to achieve sample
efficiency gains relative to architectures that must learn the symmetry from
data.  The magnitude of this gain depends on the degree to which training data
actually realises the full PGL₂(Q_p) symmetry; quantifying this
experimentally is a prerequisite before stronger claims can be made.

---

## 8. Advantage 7 — Algorithmic Efficiency via Exact Index Structures  [Near-term Roadmap]

### Approximate Nearest-Neighbour Is a Workaround

In high-dimensional Euclidean space, exact nearest-neighbour search is
NP-hard (under standard conjectures), so practical systems use approximate
methods (LSH, HNSW, ScaNN).  These introduce recall–precision trade-offs and
require careful tuning.

### p-Adic Distance Is Computable in O(prec) — Index Is the Roadmap

The `bt_distance` function in `btree.py` computes the exact BT distance
between two Q_p elements in **O(prec)** digit comparisons, where prec is the
working precision:

```python
# from btree.py — O(prec) exact distance computation
def bt_distance(ctx: QpContext, x: Qp, y: Qp) -> int:
    d = lca_depth(x.digits(ctx.prec), y.digits(ctx.prec))
    return 2 * (ctx.prec - d)
```

**What padic-ds provides**: a distance function.
**What does not yet exist**: an index structure (p-ary trie, query API,
insert/delete, benchmarks against HNSW/FAISS).

**Complexity caveat**: O(prec) = O(log_p N) where N = p^prec is the working
modulus.  This equals O(log n) for retrieval over a dataset of n points *only*
if precision is chosen such that p^prec ≈ n — one point per leaf of the trie,
with bounded bucket occupancy.  Under that assumption, nearest-neighbour
reduces to a trie prefix lookup.  With higher occupancy per bucket, a
within-bucket scan adds O(bucket_size); with dynamic inserts/deletes, trie
rebalancing adds further cost.

**Embedding caveat**: Applying p-adic indexing to real-valued embeddings (e.g.,
768-d float vectors from a language model) first requires an **embedding
module** — a mapping from ℝ^d to Q_p^d (or Q_p) that preserves semantic
neighborhoods.  The key design choices are: choice of prime p, working
precision prec, per-dimension scaling/quantisation, and handling of negative
valuations.  No universal embedding is known; this is an active research
question.

**Near-term roadmap**:

1. *Single-dimension p-ary trie*: immutable trie keyed by digit prefixes,
   with O(prec) nearest-prefix search and configurable tie-breaking.
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
geometry.  The table maps each advantage to a concrete AI subsystem and
labels current implementation status:

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
   │                            [roadmap: embedding module with stability tests]
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

The `padic-ds` library provides the foundational layers as a working reference
implementation.  The near-term priority is the BT trie index and the embedding
module, which together enable the first end-to-end retrieval benchmarks.

---

## 10. Open Challenges

Intellectual honesty requires naming the barriers:

1. **Embedding design**: Mapping real-valued vectors to Q_p^d requires choices
   of prime p, precision prec, per-dimension scale, and quantisation scheme.
   Poor choices cause excessive collisions (multiple semantically distinct
   vectors mapping to the same digit sequence) or wasted precision.  No
   principled general-purpose strategy is known.

2. **Finite precision artifacts**: Operations that increase valuation
   (near-cancellations, divisions that aren't exact units) consume precision
   silently.  kNN and clustering stability under these cancellations needs
   rigorous analysis.

3. **Hardware mismatch**: Modern accelerators (TPUs, CUDA cores) are optimised
   for IEEE 754 arithmetic.  p-Adic integer arithmetic requires either software
   emulation (slow) or custom hardware.

4. **Gradient flow**: Backpropagation requires a differentiable loss.  Z_p
   with the discrete topology has no standard notion of a gradient.  Defining a
   useful "approximate gradient" in Q_p is an open mathematical problem.

5. **Scaling laws**: Transformer scaling laws were discovered empirically for
   Euclidean architectures.  Equivalent laws for p-adic architectures are
   entirely unknown.

6. **High-dimensional branching**: For product ultrametrics Q_p^d, the
   branching factor p^d grows exponentially with d.  Practical use in high
   dimensions requires either dimensionality reduction before p-adic encoding,
   or sparse/approximate trie structures.

7. **Tooling immaturity**: `padic-ds` is a proof-of-concept.  Production-grade
   p-adic ML requires autograd support, batch operations, and GPU kernels —
   none of which currently exist.

---

## 11. Conclusion

p-Adic numbers are not a curiosity for number theorists.  Their properties —
the ultrametric inequality, the canonical tree structure, the exact arithmetic,
the rich symmetry group, the tractable index — map directly onto the structural
demands of advanced AI: hierarchical reasoning, exact symbolic computation,
efficient retrieval, and principled multi-scale representation.

The argument is not that p-adics should replace Euclidean geometry in AI, but
that they should be available as a *first-class primitive* alongside it.  Tasks
that are fundamentally hierarchical or algebraic should be solved in a geometry
that is fundamentally hierarchical and algebraic.

The `padic-ds` library provides a clean, tested implementation of the core
objects (Q_p, QpBall, BTRootedTree, PadicKNNClassifier, ultrametric_dendrogram)
that makes it possible to run controlled experiments comparing p-adic and
Euclidean representations on the same data.  The immediate next steps are:
(a) a BT trie index enabling end-to-end retrieval benchmarks;
(b) a real→Q_p embedding module with stability guarantees;
(c) Newton–Hensel quadratic lifting;
and (d) a p-adic Haar transform prototype.  The research prospects — wavelet
architectures and PGL₂(Q_p)-equivariant networks — depend on these foundations.

---

## References

1. Schikhof, W.H. (1984). *Ultrametric Calculus*. Cambridge University Press.
2. Kozyrev, S.V. (2002). Wavelet theory as p-adic spectral analysis. *Izvestiya
   Mathematics* 66(2), 367–376.
3. Bourgain, J. (1985). On Lipschitz embedding of finite metric spaces in
   Hilbert space. *Israel Journal of Mathematics* 52(1–2), 46–52.
   *(General n-point metric→L² bound; O(log n) distortion is for arbitrary metrics.)*
4. Matousek, J. (1999). On the distortion required for embedding finite metric
   spaces into normed spaces. *Israel Journal of Mathematics* 93(1), 333–344.
   *(Tree metric→L² lower bound Ω(√log n).)*
5. Dress, A.W.M. (1984). Trees, tight extensions of metric spaces, and the
   cohomological dimension of certain groups. *Advances in Mathematics* 53,
   321–402.  *(Trees embed isometrically into L¹.)*
6. Serre, J.-P. (1980). *Trees*. Springer-Verlag.
   *(Authoritative treatment of BT tree; automorphism group PGL₂(Q_p).)*
7. Khrennikov, A.Yu. (1994). *p-adic Valued Distributions in Mathematical
   Physics*. Kluwer Academic Publishers.
8. Parisi, G. & Sourlas, N. (1982). p-adic numbers and replica symmetry
   breaking. *European Physical Journal B* 14, 535–542.
   *(Theoretical physics motivation; not a direct AI engineering result.)*
9. Gubser, S.S. et al. (2017). p-adic AdS/CFT. *Communications in Mathematical
   Physics* 352(3), 1019–1059.
   *(Theoretical physics motivation; not a direct AI engineering result.)*
10. padic-ds v0.1.0. (2025). Reference implementation of p-adic data structures
    for machine learning. `/data/projects/proj-bba42f98/repos/padic-ds/`

---

*Prepared for the padic-ds project (proj-bba42f98) · Phase: active-dev*
*v2: revised per Critic review — mathematical claims tightened, implementation
status labelled per section, speculative claims tempered.*
