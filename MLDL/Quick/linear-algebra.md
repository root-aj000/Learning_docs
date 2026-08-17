---
title: Linear Algebra — Quick Revision
description: 10-minute revision of linear algebra for ML — vectors, matrices, dot products, eigen, PCA, SVD — all key formulas in one page.
tags: [math, linear-algebra, quick-rev, vectors, matrices, eigenvalues, svd]
---

# LINEAR ALGEBRA — QUICK REVISION

> Condensed from the full guide at `MLDL/maths/linear-algebra.md`. For revision: scan, recall, quiz yourself.

## THE BIG PICTURE (3 lines)

1. **Vectors** = lists of numbers = points/arrows. One vector = one data sample.
2. **Matrices** = grids of numbers = datasets (rows = samples, cols = features) AND transformations (rotate/scale space).
3. **Eigen/SVD** = "factor a matrix into its essential parts" = PCA = finding the directions of most variance in data.

## NOTATION QUICK-REF

| Symbol | Meaning |
| :--- | :--- |
| $\mathbf{v}$ | vector (column) |
| $\|\mathbf{v}\|_2$ | L2 length (Euclidean) |
| $\mathbf{a} \cdot \mathbf{b}$ | dot product (scalar) |
| $A_{ij}$ | entry: row $i$, column $j$ |
| $A^T$ | transpose (swap rows/cols) |
| $\det(A)$ | determinant (area scale) |
| $A^{-1}$ | inverse (undo transformation) |
| $\text{rank}(A)$ | number of independent rows/cols |
| $\lambda$, $\mathbf{v}$ | eigenvalue, eigenvector |
| $A = PDP^{-1}$ | eigendecomposition |
| $A = U\Sigma V^T$ | SVD |

## VECTORS — ALL KEY FORMULAS

| Operation | Formula | Memory hook |
| :--- | :--- | :--- |
| Addition | $\mathbf{a} + \mathbf{b} = (a_1{+}b_1, a_2{+}b_2)$ | tip-to-tail |
| Scaling | $c\mathbf{v} = (cv_1, cv_2)$ | stretch |
| **Dot product** | $\mathbf{a}\cdot\mathbf{b} = \sum a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ | agreement; 0 ⟺ perpendicular |
| L1 norm | $\|\mathbf{v}\|_1 = \sum |v_i|$ | manhattan |
| L2 norm | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | euclidean |
| Cosine similarity | $\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | angle only, scale-free |

**Worked dot product (10 sec):** $[2, 3] \cdot [5, 10] = 10 + 30 = 40$ — this single number IS what a neuron computes ($\mathbf{w}\cdot\mathbf{x} + b$).

**Neuron = dot + bias:** $z = \mathbf{w} \cdot \mathbf{x} + b$ — the heart of every neural network layer.

## MATRICES — ALL KEY FORMULAS

| Operation | Formula / Rule | Example / Memory hook |
| :--- | :--- | :--- |
| Matrix × vector | $y_i = \sum_j A_{ij} x_j$ (each row dots with x) | transforms a point |
| Matrix × matrix | $AB_{ik} = \sum_j A_{ij} B_{jk}$ | dims: $(m\times n)(n\times p) = m\times p$ — inner dims must match |
| Transpose | $(A^T)_{ij} = A_{ji}$ | $(AB)^T = B^T A^T$ |
| Identity | $I$ = 1s on diagonal, 0s else | $AI = A$ |
| Determinant (2×2) | $\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad - bc$ | area scale; 0 = squashed |
| Inverse (2×2) | $A^{-1} = \frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ | exists ⟺ $\det \ne 0$ |
| Rank | # independent rows/cols | max rank = $\min(m, n)$ |
| Normal equation | $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ | closed-form linear regression |

**Worked matmul (30 sec):** $A(2\times3) \cdot B(3\times2) = C(2\times2)$; entry $C_{11}$ = row 1 of A × col 1 of B: $1(7)+2(9)+3(11) = 58$.

## SPAN, SYSTEMS, EIGEN — THE DEEP PART

**Linear combination:** $\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2$ — "reach w by stretching vectors". **Span** = all reachable points. **Independent** vectors: none is a combo of others (determinant ≠ 0).

**Solving $A\mathbf{x} = \mathbf{b}$ (Gaussian elimination, 3 steps):** write augmented $[A \mid \mathbf{b}]$ → eliminate below the diagonal → back-substitute. No solution if inconsistent row ($0 = c \ne 0$); infinite if free variable.

**Eigen equation (the #1 ML concept):**
$$A\mathbf{v} = \lambda \mathbf{v}$$
- $\mathbf{v}$ = direction that survives the transformation (only stretched, not rotated).
- $\lambda$ = the stretch factor.
- Solve: $\det(A - \lambda I) = 0$ → then back-substitute for $\mathbf{v}$.

**Worked (1 min):** $A = \begin{bmatrix}2&1\\1&2\end{bmatrix}$ → $\det(A-\lambda I) = (2-\lambda)^2 - 1 = 0$ → $\lambda = 3, 1$. Eigenvectors: $[1,1]$, $[1,-1]$. Verify: $A[1,1] = [3,3] = 3[1,1]$ ✓.

**Eigendecomposition:** $A = PDP^{-1}$ (P = eigenvectors in columns, D = eigenvalues diagonal). Valid only for diagonalizable (usually symmetric) matrices.

**SVD (works for ALL matrices):** $A = U\Sigma V^T$ — $U$ = left singular vectors, $\Sigma$ = singular values (diagonal), $V$ = right singular vectors. For symmetric matrices: SVD = eigendecomposition.

**PCA (4 steps):**
1. Center data: subtract each column mean.
2. Covariance matrix: $C = \frac{1}{n-1}X^TX$.
3. Eigendecomposition of $C$ → eigenvectors = **principal components** (directions of max variance), eigenvalues = variance explained.
4. Project: $X_{\text{new}} = X \cdot P_k$ (keep top-k components) → dimensionality reduction.

## TENSORS (deep learning shapes)

**Tensor** = multi-dimensional array. Order = number of dimensions.

| Order | Name | Shape | ML example |
| :--- | :--- | :--- | :--- |
| 0 | scalar | () | bias $b$ |
| 1 | vector | (n,) | one sample's features |
| 2 | matrix | (m, n) | dataset, weight matrix |
| 3 | 3-tensor | (b, h, w) | image batch (RGB too: b,h,w,c) |
| 4 | 4-tensor | (b, c, h, w) | CNN feature maps |

**Matrix calculus (1 line):** $\nabla_W (y - Wx)^2 \propto -2(y - Wx)x^T$ — derivatives of losses w.r.t. matrices are just outer products of vectors (this IS backprop's weight update).

## THE DIFFERENTIATION TABLES (zero-confusion)

**Dot vs. cross product:** dot → scalar, any dimension, "agreement"; cross → vector, 3D only, "perpendicular + area".

**L1 vs. L2 norm:** L1 sums absolute values (robust, sparsity — used in Lasso); L2 sums squares then sqrt (smooth, Euclidean — used in ridge/weight decay).

**Determinant vs. inverse:** determinant = number (area scaling, existence check); inverse = matrix (the undo operation). $\det = 0$ → both "squashed space" and "no inverse".

**Eigendecomposition vs. SVD:** eigen needs square diagonalizable matrices; SVD works on ANY matrix (this is why SVD is used in practice, e.g. collaborative filtering, recommender systems).

**Rank vs. determinant:** rank = count of independent directions (works on any shape); determinant = volume scaling (square only).

## WHERE LINEAR ALGEBRA APPEARS IN ML (one-line map)

| Concept | ML location |
| :--- | :--- |
| Dot product | every neuron, attention (Q·K), linear regression |
| Matrix mult | every layer, batch inference on GPU |
| Transpose | gradient propagation ($W^T$), attention |
| Determinant/inverse | normal equation, Gaussian PDF normalization |
| Rank | detecting redundant features, model capacity |
| Eigenvalues | PCA, convergence analysis of GD |
| SVD | PCA, recommender systems, low-rank approximations |
| Tensors | all of deep learning (input/output shapes) |
| LoRA | fine-tuning with low-rank weight updates ($\Delta W = BA$, tiny memory) |

## TOP 5 COMMON MISTAKES

1. Matmul dims: inner dimensions must match — $(m\times n)(n\times p)$, NOT $(m\times n)(m\times p)$.
2. $AB \ne BA$ — matrix multiplication is NOT commutative.
3. $\det = 0$ → no inverse — never divide by zero determinant.
4. Eigenvectors must be verified: $A\mathbf{v} = \lambda\mathbf{v}$, not just "the answer from software".
5. Confusing rows/cols: in ML, rows = samples, columns = features — always.

> Full detail + worked examples: `MLDL/maths/linear-algebra.md` — then `calculus` (gradients move weights), `statistics` (covariance), `probability` (distributions).