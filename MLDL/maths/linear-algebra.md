---
title: Linear Algebra for Machine Learning
description: Complete beginner-friendly linear algebra for ML — vectors, matrices, transformations, eigenvalues, PCA, SVD, tensors, with worked numeric examples and visualizations.
tags: [math, linear-algebra, vectors, matrices, eigenvalues, svd, pca, tensors, ml]
---

# LINEAR ALGEBRA FOR MACHINE LEARNING

> This document is **fully self-contained**. You do not need to search the internet, open another textbook, or guess anything. Every symbol is defined, every formula is derived step by step, every example shows the full arithmetic, and every concept has a picture. Read it top to bottom.

---

# Part 0: PREREQUISITES — read this first, nothing is skipped

Linear algebra is the *language* of machine learning: data is stored in vectors and matrices, and models are built from matrix operations. Before any of that, you need four small building blocks. They are reviewed below **in full** — if you already know them, skim; if you do not, the refresher is complete.

---

## 0.1 What is a set? (the container for all of math)

A **set** is simply a collection of distinct things, written with curly braces:

$$\{1, 2, 3, 4, 5, 6\}$$

- **Element** = one thing inside the set (e.g. $3$ is an element of the set above).
- **Order does not matter:** $\{1, 2\} = \{2, 1\}$ (they are the same set).
- **Membership symbol:** $3 \in \{1, 2, 3\}$ reads *"3 is in the set"*; $7 \notin \{1, 2, 3\}$ reads *"7 is not in the set."*
- **Special sets you will meet constantly:**
  - $\mathbb{R}$ = the set of all real numbers (every number on the number line: $-2$, $0$, $0.5$, $\pi$, ...)
  - $\mathbb{R}^n$ = the set of all *lists of n real numbers* — e.g. $\mathbb{R}^2$ is the set of all pairs $(x, y)$ — i.e. **the entire 2D plane**, and $\mathbb{R}^3$ is **the entire 3D space**.

**Why sets matter here:** vectors are *elements* of $\mathbb{R}^n$. When we say "a vector is in $\mathbb{R}^3$", we mean "it is a list of 3 numbers, so it lives in 3-dimensional space."

---

## 0.2 Ordered lists (what a vector actually is)

A **list of numbers** with a fixed order is written in parentheses or square brackets:

$$(3, 2) \quad \text{or} \quad \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

**Critical: order matters in a list** (unlike a set). $(3, 2)$ is *not* $(2, 3)$ — one points right-and-up, the other up-and-right. The first number is the $x$ (horizontal) coordinate, the second is the $y$ (vertical) coordinate.

**Quick geometry check (you need this):** the point $(3, 2)$ is found by starting at the origin $(0,0)$ (where the axes cross), moving 3 units right, then 2 units up. This is the *only* thing you need from coordinate geometry: **each number in a list = one step along one axis.**

---

## 0.3 Solving two equations with two unknowns (needed for linear systems)

Suppose we have two equations that must both be true at once:

$$x + y = 3 \qquad x - y = 1$$

**The goal:** find *one* pair $(x, y)$ that satisfies *both*.

**Method: elimination (add the equations).**
- Step 1: add the left sides and right sides: $(x + y) + (x - y) = 3 + 1$.
- Step 2: simplify: $2x = 4$.
- Step 3: divide both sides by 2: $x = 2$.
- Step 4: substitute back into the first equation: $2 + y = 3$, so $y = 1$.
- Step 5: verify in the second equation: $2 - 1 = 1$ ✓.

**Answer: $(x, y) = (2, 1)$** — the unique pair where both lines cross.

**Why this matters here:** a *system of linear equations* (Module 3) is exactly this idea with more equations and more unknowns, and matrices are the tool that solves them in one sweep.

---

## 0.4 Fractions, percentages, and square roots — the arithmetic you'll reuse

- **Fraction of a total:** if 3 of 8 items are red, the fraction is $\frac{3}{8} = 0.375 = 37.5\%$. Formula: $\text{fraction} = \frac{\text{count of interest}}{\text{total count}}$.
- **Squares and square roots:** $3^2 = 9$; $\sqrt{9} = 3$ (the number that squares to 9). $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.
- **Pythagoras (the most used fact in this document):** the straight-line distance from $(0,0)$ to $(a, b)$ is $\sqrt{a^2 + b^2}$. This one formula becomes the *L2 norm* in section 1.4 and the *Euclidean distance* used everywhere in ML.

**Worked example:** distance from the origin to the point $(3, 4)$: $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.

---

## 0.5 Notation table — every symbol used in this document

**Essential now (Modules 1–2):**

| Symbol | Name | Meaning | Example |
| :--- | :--- | :--- | :--- |
| $x$, $y$ (plain) | scalar | a single number | $x = 3$ |
| $\mathbf{x}$ (bold) | vector | an ordered list of numbers | $\mathbf{x} = [3, 2]$ |
| $\mathbb{R}^n$ | R-n | space of all n-number lists | $(3,2) \in \mathbb{R}^2$ |
| $A$ (capital, bold) | matrix | a 2D grid of numbers | $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$ |
| $A_{ij}$ | entry | the number in row $i$, column $j$ | $A_{21} = 3$ above |
| $A^T$ | transpose | rows ↔ columns swapped | $\begin{bmatrix}1&2\end{bmatrix}^T = \begin{bmatrix}1\\2\end{bmatrix}$ |
| $\|\mathbf{x}\|_2$ | L2 norm | length of the vector | $\|[3,4]\|_2 = 5$ |
| $\mathbf{a} \cdot \mathbf{b}$ | dot product | weighted-sum alignment | $[1,2]\cdot[3,4] = 11$ |

**Reference later (Modules 3–5):**

| Symbol | Name | Meaning | First appears |
| :--- | :--- | :--- | :--- |
| $\|\mathbf{x}\|_1$ | L1 norm | sum of absolute values | Module 1.4 |
| $\mathbf{a} \times \mathbf{b}$ | cross product | perpendicular vector (3D) / area (2D) | Module 1.3b |
| $\det(A)$ | determinant | area/volume scaling factor | Module 2.5 |
| $A^{-1}$ | inverse | the matrix that undoes $A$ | Module 2.5 |
| $I$ | identity matrix | matrix version of the number 1 | Module 2.4 |
| $\text{rank}(A)$ | rank | number of independent directions | Module 2.6 |
| $\lambda$ | lambda | eigenvalue (stretch factor) | Module 4.1 |
| $\Sigma$ | capital sigma | diagonal matrix of singular values | Module 4.2 |
| span | span | all reachable combinations | Module 3.1 |
| $\perp$ | perpendicular | at right angles | Module 1.5 |

---

# Part 1: The Roadmap — where this document is going

```
                          LINEAR ALGEBRA FOR ML
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
[MODULE 1]                      [MODULE 2]                      [MODULE 3]
Vectors & Spaces                Matrices & Operations           Systems & Transformations
  ├── Scalars vs Vectors          ├── What is a Matrix           ├── Linear Combinations & Span
  ├── Vector Addition & Scaling   ├── Matrix × Vector            ├── Independence & Dependence
  ├── Dot Product                 ├── Matrix Multiplication      ├── Solving Linear Systems
  ├── Cross Product               ├── Transpose, Identity,       ├── Gaussian Elimination
  ├── Norms (L1, L2)              │    Symmetric                 └── Linear Transformations
  └── Distance & Cosine           ├── Determinant & Inverse         (scale, rotate, shear)
       Similarity                 └── Rank
                                    │
    ┌───────────────────────────────┘
    ▼
[MODULE 4]                      [MODULE 5]
Eigen-Concepts & Decompositions  Tensors & Advanced Topics
  ├── Eigenvalues & Eigenvectors  ├── Tensors (multi-D arrays)
  ├── Eigendecomposition          ├── Matrix Calculus & Gradients
  ├── PCA (Principal Components)  ├── Low-Rank Adaptation (LoRA)
  └── SVD (Singular Values)       └── Embeddings & Attention
```

**How to use this roadmap:** Module 1 teaches the vocabulary (vectors). Module 2 teaches the verb (matrices act on vectors). Module 3 teaches solving problems and moving space around. Module 4 is where data science lives (PCA, SVD — the "smart" decompositions). Module 5 connects everything to modern deep learning.

---

# Part 2: COMPREHENSIVE EXPLANATION

---

# MODULE 1: VECTORS & VECTOR SPACES

---

## 1.1 Scalars vs. Vectors — the difference between a number and a list

### What is it?

- **Scalar:** a single number. It has only *magnitude* (size), no direction. Examples: temperature $25^\circ C$, price $\$300{,}000$, your age $22$.
- **Vector:** an *ordered list* of numbers. It has *both* magnitude (length) and direction. Examples: "walk 3 km North-East" (magnitude 3, direction NE), a house described by (size, bedrooms, price).

**The same information, two notations:**

$$\text{house} = \underbrace{[1500, \ 3, \ 300000]}_{\text{a 3D vector}} \qquad \text{vs} \qquad \underbrace{300000}_{\text{a scalar}}$$

**Geometric picture:** a 2D vector $\mathbf{x} = (3, 2)$ is drawn as an **arrow** starting at the origin $(0,0)$ and ending at the point $(3, 2)$. The arrow's *length* is the magnitude; the direction the arrow points is the direction.

![A vector as an arrow with components](/maths-images/linalg-vector-basics.png)

**How to compute the length (magnitude) of a vector** — the L2 norm, which you will use constantly:

$$\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

This is just Pythagoras extended to n dimensions. For $\mathbf{x} = (3, 2)$: $\|x\|_2 = \sqrt{9 + 4} = \sqrt{13} \approx 3.6$.

**Worked example — turn real data into a vector:** a student's features: hours studied = 4, attendance = 90%, previous score = 72. Vector: $\mathbf{x} = [4, 90, 72]$. This single list is exactly what an ML model receives as input.

> **TL;DR:** A scalar is one number. A vector is an ordered list of numbers = a point/arrow in space. ML uses vectors for everything (data samples, weights, gradients).

### Where, why, how in ML

- **Where:** every dataset row, every image, every word embedding, every neural network layer.
- **Why:** computers can't process concepts ("dog", "blue", "expensive") — only numbers. Vectors are the universal container.
- **How:** a single training example = one vector $\mathbf{x}$; the model computes a prediction from it; a *batch* of examples = a matrix (Module 2).

### How scalars differ from vectors (one table, zero confusion)

| | Scalar | Vector |
| :--- | :--- | :--- |
| Contents | 1 number | n numbers |
| Shape | dimensionless | $n \times 1$ (column) or $1 \times n$ (row) |
| Has direction? | no | yes |
| ML example | learning rate $\alpha = 0.01$ | feature vector $[4, 90, 72]$ |

---

## 1.2 Vector Addition & Scalar Multiplication — the two operations everything else is built from

### What is it?

**Addition:** add the vectors *component by component* (same-sized vectors only):

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Scalar multiplication:** multiply every component by the same number:

$$c \cdot \mathbf{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

> **TL;DR:** Vector addition = combine two arrows tip-to-tail. Scalar multiplication = stretch/shrink an arrow (flip if negative). Gradient descent = scale gradient by learning rate, then add to weights.

### Worked example (every number shown)

$\mathbf{a} = [1, 2]$, $\mathbf{b} = [3, 4]$, scalar $c = 2$:

- Addition: $\mathbf{a} + \mathbf{b} = [1+3, \ 2+4] = [4, 6]$.
- Scaling: $2\mathbf{a} = [2\times1, \ 2\times2] = [2, 4]$.
- Combined: $2\mathbf{a} + 0.5\mathbf{b} = [2, 4] + [1.5, 2] = [3.5, 6]$.

**Geometric meaning:**
- **Addition = "tip-to-tail":** place $\mathbf{b}$'s tail at $\mathbf{a}$'s head; the sum is the arrow from the origin to $\mathbf{b}$'s head (the parallelogram rule).
- **Scaling:** $2\mathbf{v}$ = same direction, twice as long; $-0.5\mathbf{v}$ = *opposite* direction, half as long.

![Vector addition: the parallelogram rule](/maths-images/linalg-vector-add.png)

![Vector scaling: same direction, new length](/maths-images/linalg-vector-scale.png)

### Where, why, how in ML

- **Where:** gradient descent updates, bias addition in neural layers, data preprocessing (shifting/scaling features).
- **Why:** training = repeatedly *scaling* a gradient (a vector) by a learning rate and *adding* it to the weights.
- **How:** the update rule $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha\,\mathbf{g}$ is exactly: scale the gradient vector by $-\alpha$ (scalar multiplication), then add (vector addition).

### How addition differs from scalar multiplication

- **Addition** moves you *between* positions (combines two lists into one).
- **Scalar multiplication** changes *length* (or flips direction if negative) without changing the direction line.
- **They combine** to form *linear combinations* — the single most important operation in linear algebra (Module 3).

---

## 1.3 The Dot Product — how much two vectors agree

### What is it?

The **dot product** multiplies corresponding components and sums them, producing a **single scalar**:

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**The geometric meaning (the part that matters):**

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta$$

where $\theta$ is the angle between the vectors. Since $\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ are the lengths, the dot product measures **how much the two vectors point in the same direction**:

- $\theta = 0°$ (same direction): $\cos 0 = 1$ → dot product = full product of lengths (maximum).
- $\theta = 90°$ (perpendicular): $\cos 90° = 0$ → dot product = **0**.
- $\theta = 180°$ (opposite): $\cos 180° = -1$ → dot product = negative maximum.

**The projection view:** $\|\mathbf{a}\| \cos\theta$ is the length of $\mathbf{a}$'s *shadow* cast onto $\mathbf{b}$ (the **projection**). The dot product = (shadow length) × (length of $\mathbf{b}$).

![Dot product geometry: angle and projection](/maths-images/linalg-dot-product.png)

> **TL;DR:** Dot product = weighted sum of matching components = similarity score. Zero = perpendicular. Positive = pointing same way. A neuron IS a dot product + bias.

### Worked examples (full arithmetic)

**Example 1 — compute:** $\mathbf{a} = [2, 3]$, $\mathbf{b} = [5, 10]$.

- $a_1 b_1 = 2 \times 5 = 10$
- $a_2 b_2 = 3 \times 10 = 30$
- Sum: $\mathbf{a} \cdot \mathbf{b} = 10 + 30 = 40$.

**Example 2 — the neuron's entire computation.** A single artificial neuron: features $\mathbf{x} = [2, 3]$ (hours studied, attendance score), weights $\mathbf{w} = [5, 10]$ (importance of each feature), bias $b = 1$. The neuron computes:

$$z = \mathbf{w} \cdot \mathbf{x} + b = 40 + 1 = 41$$

This *one* operation (dot product + bias) is the heart of every neural network layer.

**Example 3 — perpendicularity check:** $\mathbf{a} = [1, 2]$, $\mathbf{b} = [2, -1]$: dot $= 1(2) + 2(-1) = 2 - 2 = 0$ → the vectors are perpendicular. (The dot product is zero *exactly* when the vectors are at 90°.)

### Where, why, how in ML

- **Where:** every neuron, linear regression ($\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$), transformer attention (dot products between query and key vectors).
- **Why:** the dot product is simultaneously a *weighted sum* (combining features with weights) and a *similarity score* (how aligned two vectors are).
- **How:** attention scores in GPT are dot products: $score = \mathbf{q} \cdot \mathbf{k}$ — big dot product = the two words are strongly related.

### How the dot product differs from plain multiplication

- **Plain multiplication** scales one number by another.
- **Dot product** *combines two lists* into one number by matching corresponding positions — it only makes sense between two lists of the *same length*.

---

## 1.3b The Cross Product — perpendicular vectors and area (3D only)

### What is it?

The **cross product** $\mathbf{a} \times \mathbf{b}$ of two 3D vectors produces a **new 3D vector** that is *perpendicular to both* of them:

$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix}$$

**The geometric meaning:**
- The result points **perpendicular** to the plane spanned by $\mathbf{a}$ and $\mathbf{b}$ (direction by the right-hand rule).
- The result's **length equals the area of the parallelogram** formed by $\mathbf{a}$ and $\mathbf{b}$: $\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\|\,\|\mathbf{b}\| \sin\theta$.

> **TL;DR:** Cross product (3D only) = vector perpendicular to both inputs. Length = parallelogram area. Right-hand rule: index=a, middle=b, thumb=a×b. Rare in ML (used for 3D normals).

**Right-Hand Rule Visual:**
```
       a × b  (result points here)
            ▲
            │
            │
            │
    b ──────┼──────▶
            │
            │
            ▼
           a
```
Point fingers along **a**, curl toward **b**, thumb points to **a × b**.

### Worked example (full arithmetic)

$\mathbf{a} = [1, 0, 0]$ (x-axis), $\mathbf{b} = [0, 1, 0]$ (y-axis).

- Component 1: $a_2 b_3 - a_3 b_2 = 0(0) - 0(1) = 0$
- Component 2: $a_3 b_1 - a_1 b_3 = 0(0) - 1(0) = 0$
- Component 3: $a_1 b_2 - a_2 b_1 = 1(1) - 0(0) = 1$
- Result: $[0, 0, 1]$ — the z-axis. ✓ Perpendicular to both x and y; length 1 = area of the unit square.

![Cross product: perpendicular result, parallelogram area](/maths-images/linalg-cross-product.png)

### Where, why, how in ML

- **Where:** 3D computer vision (surface normals from two edge vectors), camera geometry, physics engines in robotics.
- **Why:** any time a plane is defined by two vectors, its *normal vector* (needed for lighting, reflection, orientation) is the cross product.
- **How:** compute the normal of a 3D surface triangle by crossing two of its edge vectors — then you know which way the surface faces.

### How the cross product differs from the dot product

| | Dot product | Cross product |
| :--- | :--- | :--- |
| Input | two vectors (any dimension) | two vectors (3D only) |
| Output | a **scalar** | a **vector** |
| Zero when | vectors are perpendicular | vectors are parallel |
| Meaning | agreement / projection | perpendicular direction + area |
| ML use | attention, neurons, regression | 3D geometry, normals |

---

## 1.4 Vector Norms (L1 and L2) — measuring the length of a vector

### What is it?

A **norm** is a way of measuring "how big" a vector is — its length. There is more than one reasonable way to measure length, and ML uses (at least) two:

**L2 norm (Euclidean length) — the "straight-line" measure:**

$$\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

**L1 norm (Manhattan length) — the "city-block" measure:**

$$\|\mathbf{x}\|_1 = |x_1| + |x_2| + \cdots + |x_n|$$

> **TL;DR:** L2 = straight-line distance (circle unit ball). L1 = city-block distance (diamond unit ball). L1 forces sparsity (Lasso); L2 shrinks smoothly (Ridge).

### Worked example (every number shown)

For $\mathbf{x} = [-3, 4]$:

- L2: $\sqrt{(-3)^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.
- L1: $|-3| + |4| = 3 + 4 = 7$.

**Interpretation of the difference:** L2 = the straight-line distance from origin to the point (5). L1 = the distance if you must travel *only along the grid lines*, like a taxi in Manhattan (go left 3 blocks, then up 4 blocks = 7 blocks total).

![L1 vs L2: city-block path vs straight line](/maths-images/linalg-norms.png)

**Why the different shapes matter — the "unit circles":** the set of all vectors with $\|x\| = 1$ under each norm looks different:
- L2 unit circle = a perfect **circle**.
- L1 unit circle = a **diamond** (corners on the axes).
- L∞ unit circle = a **square** (max absolute value = 1).

![Unit circles of L1, L2, L∞](/maths-images/linalg-unit-circles.png)

**Distance between two vectors:** the *distance* between $\mathbf{a}$ and $\mathbf{b}$ is the norm of their difference: $d = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2}$. This is how KNN and k-means measure "how far apart are these data points."

### Where, why, how in ML

- **Where:** regularization (L1 = Lasso, L2 = Ridge), error metrics (MAE uses L1, MSE/RMSE use L2), KNN, k-means, embedding similarity.
- **Why:** the *type* of length measure changes model behavior dramatically:
  - **L1 (Lasso):** penalizes by sum of absolute weights → pushes *unimportant* weights to exactly $0$ → automatic *feature selection* (the model deletes useless features). Reason: the diamond's corners touch the axes, so the optimal solution often lands exactly on an axis (a weight becomes 0).
  - **L2 (Ridge):** penalizes by sum of *squared* weights → shrinks weights toward 0 but rarely to exactly 0 → keeps all features, just smaller (stable, smooth).
- **How:** the loss becomes $\text{Loss} + \lambda\|\mathbf{w}\|_1$ (Lasso) or $\text{Loss} + \lambda\|\mathbf{w}\|_2^2$ (Ridge). $\lambda$ controls how hard the penalty pushes.

### How L1 differs from L2 (the decision table)

| | L1 (Manhattan) | L2 (Euclidean) |
| :--- | :--- | :--- |
| Formula | sum of absolute values | square root of sum of squares |
| Unit circle shape | diamond | circle |
| Differentiable at 0? | no (sharp corner) | yes (smooth) |
| Effect on small weights | forces to exactly 0 (sparse) | shrinks but keeps them |
| ML use | Lasso → feature selection | Ridge → stability, and general distance |

---

## 1.5 Distance and Cosine Similarity — how similar are two vectors?

### What is it?

Two common ways to compare vectors:

**1. Euclidean distance (uses L2):**
$$d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}$$
*Small distance = similar. Sensitive to the vectors' lengths (magnitudes).*

**2. Cosine similarity (uses the angle only):**
$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$
*Ignores lengths entirely — compares only the direction (angle).*

**Range of cosine similarity:** $+1$ = same direction, $0$ = perpendicular (unrelated), $-1$ = opposite.

![Cosine similarity: aligned, orthogonal, opposite](/maths-images/linalg-cosine-similarity.png)

> **TL;DR:** Euclidean distance cares about magnitude AND direction. Cosine similarity ONLY cares about direction (angle). For text embeddings: cosine = meaning similarity, distance = length difference.

### Worked example (full arithmetic)

Compare documents as vectors: Doc A = "space rockets" → $\mathbf{a} = [3, 1]$, Doc B = "space planets" → $\mathbf{b} = [3, 0]$, Doc C = "cooking pasta" → $\mathbf{c} = [0, 2]$.

- $\mathbf{a} \cdot \mathbf{b} = 3(3) + 1(0) = 9$. $\|\mathbf{a}\| = \sqrt{10} \approx 3.16$, $\|\mathbf{b}\| = 3$.
- Cosine: $\frac{9}{3.16 \times 3} = \frac{9}{9.49} \approx 0.95$ → docs A and B are *very similar* (both about space). ✓
- $\mathbf{a} \cdot \mathbf{c} = 3(0) + 1(2) = 2$. Cosine: $\frac{2}{3.16 \times 2} = \frac{2}{6.32} \approx 0.32$ → docs A and C are *barely related*. ✓

**Why cosine and not distance?** Consider a long article (1000 words) and a short one (100 words) about the same topic. Their *distance* is large (different lengths) but their *direction* is nearly identical → cosine ≈ 1 → correctly judged similar. **Cosine ignores document length; distance does not.**

### Where, why, how in ML

- **Where:** search engines, recommendation systems, sentence/word embeddings (Word2Vec, BERT embeddings), RAG retrieval.
- **Why:** in embedding space, *meaning* lives in direction, not magnitude.
- **How:** convert text → embedding vectors → rank results by cosine similarity. The transformer attention mechanism also uses normalized dot products, which are the same thing.

### How distance differs from cosine similarity (one table)

| | Euclidean distance | Cosine similarity |
| :--- | :--- | :--- |
| Sensitive to length? | yes | no |
| Range | $0$ to $\infty$ | $-1$ to $+1$ |
| Bigger = | farther apart | more similar |
| Best for | geometric positions | text/embedding similarity |

---

# MODULE 2: MATRICES & MATRIX OPERATIONS

---

## 2.1 What is a Matrix? — a 2D table of numbers

### What is it?

A **matrix** is a rectangular grid of numbers with a defined number of **rows** and **columns**:

$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}$$

- **Dimension:** $m \times n$ means $m$ rows and $n$ columns (read "m by n").
- **Entry $A_{ij}$:** the number at row $i$, column $j$ (row first, column second — always!).
- The example above is a $2 \times 3$ matrix (2 rows, 3 columns).

**Worked example — a dataset is a matrix.** 3 patients, each with 2 health metrics (age, blood pressure):

$$X = \begin{bmatrix} 25 & 120 \\ 45 & 135 \\ 60 & 150 \end{bmatrix}$$

- Rows = samples (patients). Columns = features (age, BP).
- $X_{2,1} = 45$ = "patient 2's age." $X_{3,2} = 150$ = "patient 3's blood pressure."

**Convention you must remember:** **rows = data samples, columns = features.** This is the universal ML layout (in the code, "X has shape (n_samples, n_features)").

### Where, why, how in ML

- **Where:** datasets, grayscale images (each pixel grid), neural network weight layers, covariance matrices.
- **Why:** matrices let us apply an operation to *many* samples at once — the entire reason GPUs are fast at deep learning.
- **How:** instead of looping over 1000 patients one by one, one matrix multiplication transforms all 1000 at once.

### How a matrix differs from a vector

| | Vector | Matrix |
| :--- | :--- | :--- |
| Shape | $n \times 1$ or $1 \times n$ (one dimension) | $m \times n$ (two dimensions) |
| Picture | arrow / list | grid / table |
| ML example | one patient's features | the whole dataset |

*A vector is just a matrix with only one column (or one row).*

---

## 2.2 Matrix × Vector Multiplication — the most important operation in ML

### What is it?

Multiplying a matrix $A$ ($m \times n$) by a vector $\mathbf{x}$ ($n \times 1$) gives a new vector $\mathbf{y}$ ($m \times 1$). **Each entry of $\mathbf{y}$ is a dot product:** entry $i$ of the result = dot product of *row $i$* of $A$ with $\mathbf{x}$:

$$\mathbf{y}_i = \sum_{j=1}^{n} A_{ij} x_j = A_{i1}x_1 + A_{i2}x_2 + \cdots + A_{in}x_n$$

**The geometric meaning (the deep insight):** multiplying by $A$ *combines the columns of $A$* using $\mathbf{x}$'s numbers as weights:

$$A\mathbf{x} = x_1 \cdot (\text{column 1}) + x_2 \cdot (\text{column 2}) + \cdots + x_n \cdot (\text{column } n)$$

### Worked example (full arithmetic)

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

- Entry 1 = row 1 · x = $2(2) + 1(1) = 4 + 1 = 5$.
- Entry 2 = row 2 · x = $1(2) + 3(1) = 2 + 3 = 5$.
- Result: $A\mathbf{x} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$.

**Column-combination check (same answer, different view):**
$A\mathbf{x} = 2 \cdot \begin{bmatrix}2\\1\end{bmatrix} + 1 \cdot \begin{bmatrix}1\\3\end{bmatrix} = \begin{bmatrix}4\\2\end{bmatrix} + \begin{bmatrix}1\\3\end{bmatrix} = \begin{bmatrix}5\\5\end{bmatrix}$ ✓ — same result, and now you *see* the geometric meaning: the output is built by stretching the columns with $x$'s weights.

![Matrix × vector = weighted combination of columns](/maths-images/linalg-matrix-vector.png)

**Dimension rule (memorize):** $A$ ($m \times n$) times $\mathbf{x}$ ($n \times 1$) requires the *inner* dimensions to match ($n = n$); the result has the *outer* dimensions ($m \times 1$).

### Where, why, how in ML

- **Where:** the core of every neural network layer: $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ — weights matrix times input vector plus bias.
- **Why:** the weight matrix stores *all* connections between input features and layer outputs; one multiplication applies them all.
- **How:** a layer with 4 inputs and 3 outputs is a $3 \times 4$ matrix; feeding a batch of 1000 inputs just stacks vectors into a matrix (next section).

---

## 2.3 Matrix Multiplication — the batch version of everything above

### What is it?

Multiplying two matrices $A$ ($m \times n$) and $B$ ($n \times p$) gives $C$ ($m \times p$). **Each entry of $C$ is a dot product of a row of $A$ with a column of $B$:**

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Dimension rule:** columns of $A$ MUST equal rows of $B$ (the inner dimensions match: $n = n$). Result size = (rows of $A$) × (columns of $B$) — the outer dimensions.

### Worked example — every single arithmetic step

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \ (2 \times 3), \qquad B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix} \ (3 \times 2)$$

Inner dimensions: $A$ has 3 columns, $B$ has 3 rows → OK. Result: $2 \times 2$.

- $C_{11}$ = row 1 of A · col 1 of B = $1(7) + 2(9) + 3(11) = 7 + 18 + 33 = 58$.
- $C_{12}$ = row 1 of A · col 2 of B = $1(8) + 2(10) + 3(12) = 8 + 20 + 36 = 64$.
- $C_{21}$ = row 2 of A · col 1 of B = $4(7) + 5(9) + 6(11) = 28 + 45 + 66 = 139$.
- $C_{22}$ = row 2 of A · col 2 of B = $4(8) + 5(10) + 6(12) = 32 + 50 + 72 = 154$.

$$AB = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}$$

![Matrix multiplication: row × column dot products](/maths-images/linalg-matmul.png)

**Critical property — order matters!** In general $AB \neq BA$:
- $AB$ requires (cols of A = rows of B).
- $BA$ requires (cols of B = rows of A) — a *different* size condition.
- Even when both are defined, the numbers differ. **Matrix multiplication is NOT commutative.** (Check with the example above: $BA$ would be a $3 \times 3$ matrix — a different shape entirely.)

**ASCII mental model:**

```
A (2×3)        B (3×2)          C (2×2)
┌─────────┐   ┌────────┐      ┌──────────┐
│ 1 2 3   │   │ 7   8  │      │ 58   64  │
│ 4 5 6   │ × │ 9  10  │  =   │ 139  154 │
└─────────┘   │ 11 12  │      └──────────┘
              └────────┘
   (rows of A)  (columns of B) → dot products fill C
```

### Where, why, how in ML

- **Where:** every forward pass of a neural network ($Y = XW + B$ — a whole *batch* of samples through a layer at once), transformer attention, convolutions.
- **Why:** batched matrix multiplication is exactly what GPUs are built for — thousands of dot products in parallel.
- **How:** input batch $X$ (128 samples × 4 features) times weights $W$ (4 × 3 outputs) = predictions (128 × 3) — all samples processed simultaneously.

### How matrix multiplication differs from scalar multiplication

- **Scalar × scalar:** one number, commutative ($2\times3 = 3\times2$).
- **Scalar × matrix:** scales every entry (commutative).
- **Matrix × matrix:** dot-product per entry, size constraints, **not commutative** — the single most common beginner mistake is assuming $AB = BA$. It is not.

---

## 2.4 Transpose, Identity, and Special Matrices

### What is it?

**Transpose ($A^T$):** flip the matrix across its main diagonal — rows become columns, columns become rows. The entry at $(i, j)$ moves to $(j, i)$.

**Worked example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

- $A$ was $2 \times 3$; $A^T$ is $3 \times 2$. The first row $(1, 2, 3)$ became the first column.

![Transpose: swap rows and columns](/maths-images/linalg-transpose.png)

**Identity matrix ($I$):** square matrix with 1s on the main diagonal and 0s elsewhere:

$$I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \qquad I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**The identity is the "1" of matrices:** multiplying by it changes nothing: $A \cdot I = A$ and $I \cdot A = A$ (when sizes allow).

**Symmetric matrix:** a square matrix equal to its own transpose: $A = A^T$. The entry mirror image across the diagonal is identical.

**Worked example:** $S = \begin{bmatrix} 4 & 7 \\ 7 & 2 \end{bmatrix}$ is symmetric ($S_{12} = S_{21} = 7$). Covariance matrices and Hessian matrices are always symmetric — an important fact we will use in Module 4.

### Where, why, how in ML

- **Where:** transposes appear whenever dimensions must be aligned (e.g. computing $X^T X$ in the normal equation), covariance matrices, computing gradients.
- **Why:** the transpose is the *alignment tool* — it rotates a matrix so its dimensions match what an operation needs.
- **How:** the normal equation for linear regression literally is $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ — two transposes inside one formula.

### How transpose differs from inverse (common confusion)

- **Transpose** $A^T$: just re-arranges entries (always exists, easy to compute).
- **Inverse** $A^{-1}$: an entirely different matrix that *undoes* $A$'s transformation (does not always exist, expensive to compute). Never mix the superscripts: $T$ = rearrange, $-1$ = undo.

---

## 2.5 Determinant & Matrix Inverse — "how much space gets stretched" and "how to undo it"

### What is it?

**Determinant ($\det(A)$ or $|A|$):** a single number that tells you how a matrix scales *area* (2D) or *volume* (3D):
- $|\det| = 2$ → the transformation doubles area.
- $\det = 0$ → the transformation *squashes* space into a lower dimension (area becomes 0) → the matrix is **singular** (no inverse exists).
- $\det < 0$ → the transformation also *flips* orientation (mirror).

**Why is the formula $ad - bc$?** For a $2 \times 2$ matrix, the columns are vectors $\begin{bmatrix}a\\c\end{bmatrix}$ and $\begin{bmatrix}b\\d\end{bmatrix}$. The parallelogram they form has area $|ad - bc|$. The sign tells you if the orientation flipped.

> **TL;DR:** Determinant = area scaling factor. $\det = ad - bc$ IS the (signed) area of the parallelogram formed by the two column vectors. $\det = 0$ = squashed flat = no inverse.

**Formula for a $2 \times 2$ matrix (memorize):**

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**Worked example:** $\det\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = 1(4) - 2(3) = 4 - 6 = -2$. (Area doubled *and* orientation flipped — hence the minus.)

![Determinant = area scaling; det = 0 squashes to a line](/maths-images/linalg-det-area.png)

**Visual: Why $ad - bc$ is the Area**
```
Column 1 = [a, c] = (a, c)
Column 2 = [b, d] = (b, d)
Parallelogram corners: (0,0), (a,c), (b,d), (a+b, c+d)
Area = |a*d - b*c|  ← the determinant!
```

**Inverse ($A^{-1}$):** the matrix that *undoes* $A$. By definition:

$$A \cdot A^{-1} = I \quad \text{and} \quad A^{-1} \cdot A = I$$

**Formula for a $2 \times 2$ inverse (memorize):**

$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Notice: the denominator $ad - bc$ **is** the determinant. If $\det = 0$, division is impossible → **no inverse exists** (the transformation squashed space, so it cannot be undone).

> **TL;DR:** Inverse = the "undo" matrix. $A^{-1}$ exists ⟺ $\det \neq 0$. Formula: swap diagonal, negate off-diagonal, divide by determinant.

**Worked example — compute and verify:**

$$A = \begin{bmatrix} 4 & 3 \\ 2 & 2 \end{bmatrix}$$

- Step 1 — determinant: $ad - bc = 4(2) - 3(2) = 8 - 6 = 2 \neq 0$ → inverse exists.
- Step 2 — swap $a$, $d$; negate $b$, $c$: $\begin{bmatrix} 2 & -3 \\ -2 & 4 \end{bmatrix}$.
- Step 3 — divide by 2: $A^{-1} = \begin{bmatrix} 1 & -1.5 \\ -1 & 2 \end{bmatrix}$.
- Step 4 — verify $A A^{-1} = I$:
  - Row 1 · col 1: $4(1) + 3(-1) = 4 - 3 = 1$ ✓
  - Row 1 · col 2: $4(-1.5) + 3(2) = -6 + 6 = 0$ ✓
  - Row 2 · col 1: $2(1) + 2(-1) = 2 - 2 = 0$ ✓
  - Row 2 · col 2: $2(-1.5) + 2(2) = -3 + 4 = 1$ ✓
  - Result: $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ ✓ Verified.

### The normal equation — the closed-form solution to linear regression

Given data matrix $X$ (samples × features) and target vector $\mathbf{y}$, the best weights are:

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

**Why this works (intuition):** we want $X\mathbf{w} \approx \mathbf{y}$. Multiplying both sides by $X^T$ "projects" the problem onto the features, and multiplying by $(X^T X)^{-1}$ undoes the feature matrix — leaving the weights that best fit the data. **If $X^T X$ has determinant 0** (redundant/duplicate features), the inverse doesn't exist, and we must fall back to gradient descent or a pseudo-inverse.

### Where, why, how in ML

- **Where:** closed-form linear regression, Gaussian distributions (the determinant appears in the normalizing constant), covariance analysis.
- **Why:** the inverse solves systems directly; the determinant tells us *whether* a unique solution exists at all.
- **How:** in practice, libraries never compute inverses directly for big problems (numerically unstable) — they use *solves* ($A^{-1}b$ computed without forming $A^{-1}$). Same math, safer numbers.

### How determinant differs from inverse

| | Determinant $\det(A)$ | Inverse $A^{-1}$ |
| :--- | :--- | :--- |
| Type | a number | a matrix |
| Meaning | how much area/volume is scaled | the undo-operation |
| $\det = 0$ | "squashed space" | "no inverse exists" |
| Cost | cheap | expensive (for big matrices) |

---

## 2.6 Rank of a Matrix — how much *real* information is inside

### What is it?

The **rank** of a matrix is the number of **linearly independent** rows (or columns) — in plain words: *the number of genuinely different directions the matrix's rows/columns span.*

**Worked example — a rank-deficient matrix:**

$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

Row 2 is exactly $2 \times$ row 1: $(2, 4) = 2(1, 2)$. The second row adds **zero new information** — it points along the same line as the first. **Rank = 1** (only one independent direction), even though the matrix has 2 rows.

**Full-rank example:** $\begin{bmatrix} 1 & 2 \\ 3 & 5 \end{bmatrix}$ — neither row is a multiple of the other → rank = 2 (full rank).

![Independent vs dependent vectors: rank 2 vs rank 1](/maths-images/linalg-rank.png)

**Rules of thumb (memorize these):**
- Max possible rank = min(rows, columns).
- Rank < that → the matrix is **rank-deficient** (contains redundancy / squashes space).
- A square matrix is invertible **exactly when** its rank is full (equivalently, $\det \neq 0$).

### Where, why, how in ML

- **Where:** LoRA (low-rank adaptation of LLMs), matrix completion, PCA/SVD (Module 4), detecting redundant features (multicollinearity).
- **Why:** rank tells us how much *storage and computation* a matrix truly needs. A $1000 \times 1000$ matrix of rank 10 contains the same information as 10 vectors.
- **How — LoRA (concrete numbers):** fine-tuning an LLM updates a huge weight matrix $W$ (e.g., $4096 \times 4096 = 16.7$M parameters). Instead, LoRA writes the update as $W + BA$ where $B$ is $4096 \times 8$ and $A$ is $8 \times 4096$ (rank 8). Total new params: $4096 \times 8 + 8 \times 4096 = 65,536$ — **99.6% fewer parameters** (65K vs 16.7M). The update's true rank is tiny, so we only learn the low-rank part.

### How rank differs from determinant

- **Determinant:** only defined for *square* matrices; one number.
- **Rank:** defined for *any* matrix; a count of independent directions. A non-square matrix has no determinant but always has a rank.

---

# MODULE 3: LINEAR COMBINATIONS, SYSTEMS, AND TRANSFORMATIONS

---

## 3.1 Linear Combinations, Span, and Independence

### What is it?

**Linear combination:** building a new vector by *scaling and adding* a collection of vectors:

$$\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

**Span:** the set of *all possible* linear combinations of a set of vectors. In plain words: *"every point I can reach by stretching and adding these vectors."*

- One vector → its span is a **line** through the origin (stretch it any amount).
- Two independent vectors → their span is the **entire 2D plane**.
- Three independent vectors → their span is **all of 3D space**.

**Linear independence:** a set of vectors is **independent** if *no* vector in the set can be written as a combination of the others. If one *can* (e.g. $\mathbf{v}_2 = 2\mathbf{v}_1$), the set is **dependent** — one vector is redundant.

### Worked examples (full arithmetic)

**Example 1 — build a combination:** $\mathbf{v}_1 = [1, 2]$, $\mathbf{v}_2 = [3, 1]$, weights $c_1 = 2$, $c_2 = -1$:

$$2\mathbf{v}_1 - \mathbf{v}_2 = [2, 4] - [3, 1] = [-1, 3]$$

**Example 2 — dependent or not?** Are $\mathbf{a} = [1, 2]$ and $\mathbf{b} = [2, 4]$ independent? Try to solve $\mathbf{b} = c\mathbf{a}$: $[2, 4] = c[1, 2]$ → $c = 2$ works. **Dependent** — same line, no new direction.

**Example 3 — independent?** Are $\mathbf{a} = [1, 0]$ and $\mathbf{b} = [0, 1]$ independent? $\mathbf{b} = c\mathbf{a}$ would require $[0, 1] = [c, 0]$ → impossible. **Independent** — they span the whole plane (with weights $c_1, c_2$ you can reach any point: $c_1[1,0] + c_2[0,1] = [c_1, c_2]$).

![Two independent vectors span the whole plane](/maths-images/linalg-span.png)

### Where, why, how in ML

- **Where:** feature analysis, detecting multicollinearity, understanding neural network layer outputs.
- **Why:** *dependent features* (e.g. "size in sq ft" and "size in sq meters") are the same information twice — they make models unstable and waste computation.
- **How:** compute the correlation matrix (see Statistics doc); if two features are near-perfectly correlated, drop one. The rank of the feature matrix tells you how many *genuinely independent* signals you have.

### How span differs from independence

- **Span** = the *set of reachable points* (a geometric region).
- **Independence** = a *property of the vectors* (whether any are redundant).
- **They connect:** the dimension of the span = the number of independent vectors in the set. More independent vectors → bigger span (line → plane → space).

---

## 3.2 Solving Systems of Linear Equations — where lines cross

### What is it?

A **system of linear equations** is several equations that must all be true at the same time:

$$3x + 2y = 7 \qquad x - y = 1$$

In matrix form, every linear system looks like:

$$A\mathbf{x} = \mathbf{b}$$

where $A$ holds the coefficients, $\mathbf{x}$ holds the unknowns, and $\mathbf{b}$ holds the right-hand sides:

$$\begin{bmatrix} 3 & 2 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 7 \\ 1 \end{bmatrix}$$

**Geometric meaning:** each equation is a line (2D) or a plane (3D). The *solution* is the point where they all meet.

![Two lines crossing: the solution to a 2×2 system](/maths-images/linalg-linear-system.png)

### Method 1 — elimination (already reviewed in prerequisites, quick recap)

1. Add/subtract equations to eliminate one variable.
2. Solve for the remaining variable.
3. Substitute back.

**Worked example:** $3x + 2y = 7$, $x - y = 1$.

- From equation 2: $x = 1 + y$.
- Substitute into equation 1: $3(1 + y) + 2y = 7$ → $3 + 3y + 2y = 7$ → $5y = 4$ → $y = 0.8$.
- Back: $x = 1 + 0.8 = 1.8$.
- Verify in equation 1: $3(1.8) + 2(0.8) = 5.4 + 1.6 = 7$ ✓.

### Method 2 — Gaussian elimination (the algorithm computers actually use)

**Goal:** turn $A\mathbf{x} = \mathbf{b}$ into an upper-triangular form (zeros below the diagonal), then solve from the bottom up.

**Worked example (full arithmetic):**

$$\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$$

**Step 1 — write the augmented matrix** (coefficients + right side):

$$\left[\begin{array}{cc|c} 2 & 1 & 5 \\ 1 & 3 & 5 \end{array}\right]$$

**Step 2 — make the entry below the first pivot (2) become 0.** Multiply row 2 by 2 (so its first entry matches), then subtract row 1 from it:
- New row 2 = $2 \times$ (row 2) − row 1 = $(2, 6, 10) - (2, 1, 5) = (0, 5, 5)$.

$$\left[\begin{array}{cc|c} 2 & 1 & 5 \\ 0 & 5 & 5 \end{array}\right]$$

**Step 3 — read off the bottom row:** $5y = 5$ → $y = 1$.

**Step 4 — substitute upward:** $2x + 1 = 5$ → $2x = 4$ → $x = 2$.

**Answer: $(x, y) = (2, 1)$.** Check row 2 of the original: $1(2) + 3(1) = 5$ ✓.

**The three possible outcomes (know these):**
1. **Unique solution** — lines cross at one point (what we just saw).
2. **No solution** — parallel lines (contradictory equations).
3. **Infinite solutions** — same line twice (dependent equations).

### Where, why, how in ML

- **Where:** solving regression weights, Kalman filters, optimization sub-problems, physics engines.
- **Why:** many ML problems reduce to "solve $A\mathbf{x} = \mathbf{b}$" — the *solve* is the workhorse.
- **How:** `numpy.linalg.solve(A, b)` executes Gaussian elimination (with clever pivoting for numerical safety). When no exact solution exists (more equations than unknowns), ML uses **least squares** — find $\mathbf{x}$ that makes $A\mathbf{x}$ *as close as possible* to $\mathbf{b}$ (this is what the normal equation does).

### How a system differs from a single equation

- **Single equation:** infinitely many solutions (a whole line of points).
- **System:** *multiple constraints at once* — the solution must satisfy all of them; typically one point (or none).

---

## 3.3 Linear Transformations — matrices as "machines that move space"

### What is it?

A **linear transformation** is a rule that takes every vector in space and moves it to a new vector, in a way that:
1. **The origin stays fixed** (the zero vector maps to the zero vector).
2. **Straight lines stay straight** (no bending).
3. **Grid lines stay parallel and evenly spaced** (no warping).

**Every linear transformation is a matrix multiplication.** The matrix *encodes the recipe*: its columns tell you where the "unit arrows" (1,0) and (0,1) land.

**The four classic transformations (all with matrices):**

| Transformation | Matrix | What happens to the unit square |
| :--- | :--- | :--- |
| **Scale** | $\begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ | stretched horizontally ×2, squashed vertically ÷2 |
| **Rotation 90°** | $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ | turns the whole square sideways |
| **Shear** | $\begin{bmatrix} 1 & 0.7 \\ 0 & 1 \end{bmatrix}$ | slides the top sideways, like pushing a book stack |
| **Reflection** | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | flips the square over the x-axis (mirror) |

![The unit square under identity, scale, rotation, and shear](/maths-images/linalg-transformations.png)

**Worked example — shear, computed:** apply $S = \begin{bmatrix} 1 & 0.7 \\ 0 & 1 \end{bmatrix}$ to the corner point $(1, 1)$:

$$S \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1(1) + 0.7(1) \\ 0(1) + 1(1) \end{bmatrix} = \begin{bmatrix} 1.7 \\ 1 \end{bmatrix}$$

The point moved from $(1,1)$ to $(1.7,1)$ — pushed right by the shear.

**Why the columns tell the whole story:** the matrix's first column $(1, 0)^T$ is where $(1,0)$ lands; the second column $(0.7, 1)^T$ is where $(0,1)$ lands. Every other point is a linear combination of these two destinations.

### Where, why, how in ML

- **Where:** neural network layers ($\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is a linear transformation *plus a shift*), data preprocessing, computer vision (rotation/flip augmentations).
- **Why:** a linear layer learns a *matrix* — i.e. it learns a transformation of feature space — and stacking many transformations with nonlinearities (activation functions) is what makes deep networks powerful.
- **How:** the "bias" term $\mathbf{b}$ is what allows shifting away from the origin (a pure linear transformation can never move the origin; adding bias can).

### How a linear transformation differs from a general function

- **Linear:** only scaling and adding (matrix multiplication) — origin fixed, lines straight.
- **Nonlinear** (e.g. $f(x) = x^2$, sigmoid, ReLU): can bend, warp, and move the origin. Neural networks *need* nonlinear activation functions — otherwise a stack of linear layers would collapse into one linear layer (the whole network would be expressible as a single matrix).

---

# MODULE 4: EIGENVALUES, EIGENVECTORS & DECOMPOSITIONS

---

## 4.1 Eigenvalues and Eigenvectors — the directions a matrix "only stretches"

### What is it?

When a matrix multiplies a typical vector, it rotates *and* stretches it. But **some special vectors are not rotated at all** — the matrix only changes their length. Those vectors are **eigenvectors**, and the stretch factor is the **eigenvalue**:

$$A\mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$ = eigenvector (direction that stays fixed)
- $\lambda$ (lambda) = eigenvalue (how much it stretches; negative = also flips)

**Real-world analogy:** stretch a rubber sheet diagonally. Most points move off their original lines, but points *along the stretch direction* stay on their line — they're just pulled. Those fixed directions are the eigenvectors.

> **TL;DR:** Eigenvector = direction that doesn't rotate (only stretches). Eigenvalue = the stretch factor. $A\mathbf{v} = \lambda \mathbf{v}$ means "matrix times vector = scalar times same vector."

**Why $\det(A - \lambda I) = 0$?** We want $A\mathbf{v} = \lambda \mathbf{v}$.
Rearrange: $A\mathbf{v} - \lambda \mathbf{v} = 0$ → $(A - \lambda I)\mathbf{v} = 0$.
For a non-zero $\mathbf{v}$, the matrix $(A - \lambda I)$ MUST squash space (send some non-zero vector to zero).
A matrix squashes space ⟺ its determinant is zero.
Therefore: $\det(A - \lambda I) = 0$. **This is the characteristic equation — it finds the $\lambda$ values that make $A - \lambda I$ singular.**

### Worked example — find them step by step

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

**Step 1 — the characteristic equation.** Eigenvalues solve $\det(A - \lambda I) = 0$:

$$A - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}$$

$$\det = (2 - \lambda)(2 - \lambda) - 1(1) = \lambda^2 - 4\lambda + 4 - 1 = \lambda^2 - 4\lambda + 3 = 0$$

**Step 2 — solve the quadratic:** $(\lambda - 3)(\lambda - 1) = 0$ → $\lambda_1 = 3$, $\lambda_2 = 1$.

**Step 3 — find the eigenvector for $\lambda_1 = 3$:** solve $(A - 3I)\mathbf{v} = 0$:

$$\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = 0 \quad \Rightarrow \quad -v_1 + v_2 = 0 \quad \Rightarrow \quad v_1 = v_2$$

So an eigenvector is $\mathbf{v}_1 = (1, 1)$ (any multiple works).

**Step 4 — verify:** $A(1,1) = (2+1, 1+2) = (3, 3) = 3(1,1)$ ✓ — indeed stretched by exactly $\lambda = 3$, direction unchanged.

**Step 5 — the other eigenvector, $\lambda_2 = 1$:** $(A - I)\mathbf{v} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\mathbf{v} = 0$ → $v_1 + v_2 = 0$ → $\mathbf{v}_2 = (1, -1)$. Verify: $A(1,-1) = (2-1, 1-2) = (1, -1) = 1(1,-1)$ ✓.

![Eigenvectors: directions that don't rotate under A](/maths-images/linalg-eigenvectors.png)

### Where, why, how in ML

- **Where:** PCA (next section), covariance analysis, stability analysis of optimization, Markov chains.
- **Why:** eigenvalues reveal *how a matrix stretches space along which directions* — for a covariance matrix, the largest eigenvalue points along the direction of greatest data spread.
- **How:** in PCA, the eigenvectors of the covariance matrix ARE the principal components, ordered by eigenvalue size.

### How eigenvalues differ from eigenvectors (never mix them)

| | Eigenvalue $\lambda$ | Eigenvector $\mathbf{v}$ |
| :--- | :--- | :--- |
| Type | a number | a vector (direction) |
| Tells you | how much stretching | which direction |
| Analogy | the stretch factor | the stretched direction |

---

## 4.2 Eigendecomposition — breaking a matrix into its eigen-parts

### What is it?

If a matrix $A$ has $n$ independent eigenvectors, it can be **diagonalized** — rewritten as:

$$A = P D P^{-1}$$

- $P$ = matrix whose **columns are the eigenvectors**
- $D$ = diagonal matrix whose entries are the **eigenvalues** (in matching order)
- $P^{-1}$ = the inverse of $P$

**Why this decomposition is powerful:** powers of $A$ become trivial:

$$A^k = P D^k P^{-1}$$

Instead of multiplying $A$ by itself $k$ times (expensive), just raise the *diagonal* eigenvalues to the $k$-th power (cheap) and rebuild.

### Worked example — verify the decomposition

Using $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ from before ($\lambda_1 = 3$, $\mathbf{v}_1 = (1,1)$; $\lambda_2 = 1$, $\mathbf{v}_2 = (1,-1)$):

$$P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad D = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$$

**Step 1 — compute $P^{-1}$** using the 2×2 inverse formula ($\det P = 1(-1) - 1(1) = -2$):

$$P^{-1} = \frac{1}{-2} \begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix}$$

**Step 2 — verify $P D P^{-1} = A$:**

$$P D = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix}$$

$$(PD)P^{-1} = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix} \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = A \quad \blacksquare$$

### How eigendecomposition differs from SVD (the #1 confusion in this subject)

| | Eigendecomposition $A = PDP^{-1}$ | SVD $A = U\Sigma V^T$ |
| :--- | :--- | :--- |
| Applies to | square matrices only | **any** matrix (even rectangular) |
| Requires | independent eigenvectors | always exists |
| Matrices involved | $P$ (eigenvectors), $D$ (eigenvalues) | $U, V$ (rotations), $\Sigma$ (singular values) |
| Connection | eigenvalue = singular value for symmetric positive-definite $A$ | generalizes eigenvalues to non-square cases |

**When they agree:** for a symmetric matrix, the SVD's singular values are just the absolute values of the eigenvalues, and $U = V = P$.

### Where, why, how in ML

- **Where:** matrix powers (Markov chains, PageRank), stability analysis, and as the conceptual basis for PCA and SVD.
- **Why:** decompositions turn "apply a complicated matrix repeatedly" into "scale along a few fixed directions" — the same logic powering fast matrix math in libraries.
- **How:** `numpy.linalg.eig(A)` computes $P$ and $D$ in one call.

---

## 4.3 Principal Component Analysis (PCA) — finding the directions of most variance

### What is it?

**PCA** is a dimensionality-reduction technique: it finds the *few directions* along which the data varies the most, then projects the data onto those directions. The result: fewer numbers per sample, with minimal information loss.

**The one-sentence recipe:** *compute the covariance matrix of the data; its eigenvectors (ordered by eigenvalue size) are the principal components; the largest eigenvalue's direction carries the most information.*

> **TL;DR:** PCA = find directions of maximum variance (eigenvectors of covariance). Project data onto top-k directions to reduce dimensions. Keeps 99% variance with ~10% of features.

### Worked example — tiny 2D dataset

Data points (2 features): $(1, 1)$, $(2, 1)$, $(3, 3)$, $(4, 4)$, $(5, 6)$.

**Step 1 — center the data** (subtract the mean of each feature). Mean = $(\frac{1+2+3+4+5}{5}, \frac{1+1+3+4+6}{5}) = (3, 3)$. Centered: $(-2,-2), (-1,-2), (0,0), (1,1), (2,3)$.

**Step 2 — covariance matrix** (how features co-vary; see Statistics doc for the formula):
- Variance of feature 1: $\frac{(-2)^2 + (-1)^2 + 0 + 1 + 4}{5} = \frac{10}{5} = 2$.
- Variance of feature 2: $\frac{4 + 4 + 0 + 1 + 9}{5} = \frac{18}{5} = 3.6$.
- Covariance: $\frac{(-2)(-2) + (-1)(-2) + 0 + (1)(1) + (2)(3)}{5} = \frac{4+2+0+1+6}{5} = \frac{13}{5} = 2.6$.

$$C = \begin{bmatrix} 2 & 2.6 \\ 2.6 & 3.6 \end{bmatrix}$$

**Step 3 — eigenvalues/eigenvectors of $C$** (using the recipe from 4.1): the larger eigenvalue is $\lambda_1 \approx 5.53$ with eigenvector $\mathbf{v}_1 \approx (0.62, 0.79)$ — this is **PC1**, the direction of maximum variance. The second, $\lambda_2 \approx 0.07$ with $\mathbf{v}_2 \approx (-0.79, 0.62)$, is **PC2**.

**Step 4 — project:** each data point becomes one number: its coordinate along PC1. The two features collapse to one, while keeping ~99% of the total variance ($\frac{5.53}{5.53 + 0.07} \approx 99\%$).

![PCA: principal component arrows on 2D data](/maths-images/linalg-pca.png)

**Rule to remember:** the *fraction of variance kept* by the top $k$ components is $\frac{\lambda_1 + \cdots + \lambda_k}{\lambda_1 + \cdots + \lambda_n}$.

### Where, why, how in ML

- **Where:** data visualization (project 1000 features down to 2 for plotting), compression, noise reduction, speeding up models.
- **Why:** the "curse of dimensionality" — high-dimensional data is slow to train and prone to overfitting; PCA keeps the informative part.
- **How:** `sklearn.decomposition.PCA(n_components=k)` does all four steps internally.

### How PCA differs from SVD

- **PCA** is a *data analysis* method: it finds directions of variance (uses the covariance matrix's eigenvectors).
- **SVD** is a *matrix factorization* method: works on any matrix directly.
- **They connect:** PCA *is* SVD applied to the centered data matrix — the PCA components are the SVD's right singular vectors. Same math, two names, two stories.

> **WHICH ONE DO I USE? (Decision Box)**
> ```
> Is your matrix rectangular (not square)?
>   YES → SVD (PCA needs square covariance matrix)
>   NO (square) → Is it a covariance matrix (symmetric, positive semi-definite)?
>     YES → PCA (eigenvectors = principal components, directly interpretable)
>     NO → SVD (works on any square matrix; eigendecomp may fail if not diagonalizable)
> ```

---

## 4.4 Singular Value Decomposition (SVD) — the Swiss-army knife of matrices

### What is it?

**SVD** factorizes *any* matrix $A$ (even rectangular!) into three pieces:

$$A = U \Sigma V^T$$

- $U$ = matrix of **left singular vectors** (columns) — a rotation
- $\Sigma$ = diagonal matrix of **singular values** (the stretching factors, always ≥ 0)
- $V^T$ = matrix of **right singular vectors** — another rotation

**The geometric story (one picture, three steps):** multiplying by $A$ = *rotate* (by $V^T$), then *stretch along the axes* (by $\Sigma$), then *rotate again* (by $U$).

![SVD: unit circle → rotate → scale → rotate = ellipse](/maths-images/linalg-svd.png)

> **TL;DR:** SVD = any matrix = Rotation × Stretch × Rotation. Works on rectangular matrices (unlike eigendecomp). Singular values = stretch factors. Used for PCA, recommender systems, compression.

### Worked example — 2×2, fully computed

$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

**Step 1 — the singular values** are the square roots of the eigenvalues of $A^TA$:
- $A^TA = \begin{bmatrix} 10 & 6 \\ 6 & 10 \end{bmatrix}$; its eigenvalues are $16$ and $4$.
- Singular values: $\sigma_1 = \sqrt{16} = 4$, $\sigma_2 = \sqrt{4} = 2$.

$$\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}$$

**Step 2 — the singular vectors** (eigenvectors of $A^TA$): $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1,1)$, $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1,-1)$. So

$$V = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad U = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \quad \text{(for this symmetric example, } U = V \text{)}$$

**Step 3 — verify the product:** $U \Sigma V^T = \frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix} \begin{bmatrix}4&0\\0&2\end{bmatrix} \frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \begin{bmatrix}3&1\\1&3\end{bmatrix} = A$ ✓

**Why singular values matter — low-rank approximation:** to compress $A$, keep only the $k$ largest singular values (set the rest to 0). The reconstruction $\hat{A} = U_k \Sigma_k V_k^T$ is the *best rank-$k$ approximation* of $A$. This is the entire idea behind image compression, recommendation systems, and LoRA.

### Where, why, how in ML

- **Where:** recommendation systems (Netflix-style: factorize the user×movie rating matrix into latent "user preferences" × "movie traits"), image compression, LSA (topic modeling), PCA implementation.
- **Why:** SVD extracts *latent structure*: in a user×movie matrix, the top singular vectors reveal hidden factors like "action preference" without anyone labeling them.
- **How:** `numpy.linalg.svd(A)` returns $U$, $\Sigma$, $V^T$ directly.

### How SVD differs from eigendecomposition (quick recap of the table above)

Eigendecomposition needs square matrices with independent eigenvectors; SVD works on *anything*, is always stable, and is what libraries actually use under the hood for PCA and least squares.

---

# MODULE 5: TENSORS & ADVANCED TOPICS FOR MODERN ML

---

## 5.1 Tensors — vectors and matrices on steroids

### What is it?

A **tensor** is a multi-dimensional array of numbers. Vectors and matrices are just special cases:

| Rank (number of axes) | Name | Shape example | ML example |
| :--- | :--- | :--- | :--- |
| 0 | Scalar | $()$ | learning rate $0.01$ |
| 1 | Vector | $(n,)$ | one patient's features |
| 2 | Matrix | $(m, n)$ | a dataset / a weight layer |
| 3 | 3D tensor | $(w, h, 3)$ | one RGB image (width × height × color channels) |
| 4 | 4D tensor | $(b, w, h, 3)$ | a *batch* of images |

![Scalar, vector, matrix, 3D tensor](/maths-images/linalg-tensors.png)

**Worked example — an image is a tensor:** a 32×32 color photo = a $32 \times 32 \times 3$ tensor: 32 rows of pixels × 32 columns × 3 numbers per pixel (red, green, blue intensities). A batch of 64 such photos = $64 \times 32 \times 32 \times 3$.

**The reshape view:** tensors can be *flattened* — a $3 \times 3$ grayscale image becomes a 9-vector by reading pixels row by row. Dense neural network layers always receive flattened (1D) inputs.

### Where, why, how in ML

- **Where:** the fundamental data type of PyTorch and TensorFlow — every parameter, activation, and gradient is a tensor.
- **Why:** GPUs process entire tensors in parallel; tensor shapes encode the structure of the data.
- **How:** `torch.randn(64, 3, 32, 32)` creates the batch-of-images tensor above; the model's layers transform its shape step by step.

### How a tensor differs from a matrix

- A **matrix** is always 2D and usually conceptualized as a transformation.
- A **tensor** is any-dimensional and usually conceptualized as *data* (images, batches, sequences, vocabularies). Matrices are rank-2 tensors.

---

## 5.2 Matrix Calculus Basics — gradients through matrices (bridge to the Calculus doc)

### What is it?

Deep learning needs derivatives of *loss functions with respect to weight matrices*. The key objects:

**Gradient of a scalar w.r.t. a vector:** for $L(\mathbf{w})$ scalar, $\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} & \cdots & \frac{\partial L}{\partial w_n} \end{bmatrix}$ — the vector of all partial derivatives (see Calculus doc, Module 2).

**Jacobian of a vector function:** for a layer that maps vector $\mathbf{x}$ to vector $\mathbf{f}(\mathbf{x})$, the matrix of all partials $J_{ij} = \frac{\partial f_i}{\partial x_j}$.

**Worked example — the quadratic form (the loss in linear regression):**

$$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

Its gradient is $\nabla L = 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$ — every weight's sensitivity in one matrix-vector product. Setting this to zero and solving gives the normal equation $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ from section 2.5. **One concept, three names (gradient, normal equation, least squares) — all the same math.**

### Where, why, how in ML

- **Where:** every backward pass (backpropagation) in every neural network.
- **Why:** gradients through matrix layers are computed by chain-rule *matrix multiplications* — the linear algebra from this document and the calculus from the Calculus doc are literally the two halves of backprop.
- **How:** frameworks compute $\nabla L$ w.r.t. each weight matrix automatically; the optimizer then applies the update rule from the Calculus doc.

---

## 5.3 Low-Rank Adaptation (LoRA) — why rank is a memory saver for LLMs

### What is it?

Fine-tuning a large language model means updating a giant weight matrix $W$ (e.g. $4096 \times 4096$ — 16 million entries). LoRA's bet: *the update doesn't need full rank.* Instead of learning $\Delta W$ (16M numbers), learn two tiny matrices $B$ and $A$ whose product approximates it:

$$\Delta W \approx B A, \qquad B: 4096 \times r, \quad A: r \times 4096$$

with $r$ tiny (e.g. 8). The number of trainable parameters drops from 16M to $2 \times 4096 \times 8 = 65{,}536$ — a **99.6% reduction** — because the update is constrained to rank $r$.

**ASCII sketch:**

```
Full update (expensive):          LoRA update (cheap):
   W  ←  W + ΔW                      W  ←  W + B·A
   ΔW is 4096×4096                   B is 4096×8, A is 8×4096
   = 16,777,216 params               = 65,536 params (0.4%!)
```

### Where, why, how in ML

- **Where:** fine-tuning GPT-class models, QLoRA (even quantized), adapter research.
- **Why:** rank measures *true information content* — and research shows LLM updates genuinely live in a low-rank subspace.
- **How:** during training only $A$ and $B$ get gradients; at inference, $BA$ is folded back into $W$ (zero extra inference cost).

---

## 5.4 Embeddings & Attention — the linear algebra of modern NLP

### What is it?

**Embeddings:** words (or images, users) converted into vectors such that *similar things are close together*:

- "king" → $[0.12, -0.45, 0.87, \dots]$ (a vector in $\mathbb{R}^{512}$)
- The famous relation "king − man + woman ≈ queen" is literal vector arithmetic in embedding space.

**Attention (transformers):** each word's representation is updated by a *weighted sum* of other words' representations. The weights come from **dot products**:

$$\text{attention weight}(i, j) = \text{softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}\right)$$

where $\mathbf{q}_i$ (query) and $\mathbf{k}_j$ (key) are learned vectors for words $i$ and $j$. Big dot product = high similarity = word $i$ should pay attention to word $j$ (see the Probability doc for softmax).

**Where the linear algebra lives:** every step is matrix multiplication — queries, keys, values are computed as $Q = XW_Q$, $K = XW_K$, $V = XW_V$, and the whole attention matrix is one big batched matrix product.

### How embeddings differ from raw tokens

- **Raw tokens:** discrete IDs ("dog" = integer 314) — no notion of similarity; $314 + 1$ is meaningless.
- **Embeddings:** continuous vectors where *distance and direction carry meaning* — the linear algebra tools from this entire document apply directly (cosine similarity for retrieval, dot products for attention).

---

# Part 3: SUMMARY CHEAT-SHEET

| Concept | Definition in one line | Primary ML application | Key formula |
| :--- | :--- | :--- | :--- |
| **Scalar** | single number | learning rate, single value | $x$ |
| **Vector** | ordered list of numbers | one data sample / features | $\mathbf{x} = [x_1, \dots, x_n]$ |
| **Dot product** | sum of element-wise products = alignment | neuron computation, attention, similarity | $\mathbf{a}\cdot\mathbf{b} = \sum a_i b_i$ |
| **L1 norm** | sum of absolute values | Lasso regularization (feature selection) | $\|\mathbf{x}\|_1 = \sum\|x_i\|$ |
| **L2 norm** | Euclidean length | Ridge regularization, distance metrics | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ |
| **Cosine similarity** | angle between vectors, length-free | embeddings, search, RAG | $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ |
| **Matrix** | 2D grid of numbers | datasets, weight layers | $A_{ij}$ |
| **Matrix × vector** | weighted combination of columns | every neural layer | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ |
| **Matrix multiplication** | row × column dot products (not commutative) | batched forward/backward passes | $C_{ij} = \sum_k A_{ik}B_{kj}$ |
| **Transpose** | swap rows ↔ columns | aligning dimensions | $A^T$ |
| **Identity** | the "1" of matrices | $A A^{-1} = I$ | $I$ |
| **Determinant** | area/volume scaling factor; 0 = degenerate | invertibility checks, Gaussians | $ad - bc$ |
| **Inverse** | the matrix that undoes $A$ | normal equation, solving systems | $A^{-1} = \frac{1}{\det}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ |
| **Rank** | number of independent directions | LoRA, redundancy detection | — |
| **Span / Independence** | reachable points / redundant vectors | feature analysis | — |
| **Linear transformation** | matrix = move space without bending | neural layers, augmentations | $\mathbf{x} \to A\mathbf{x}$ |
| **Eigenvector / Eigenvalue** | fixed direction / stretch factor | PCA, covariance analysis | $A\mathbf{v} = \lambda\mathbf{v}$ |
| **Eigendecomposition** | $A = PDP^{-1}$ (square matrices) | matrix powers, PCA | $P D P^{-1}$ |
| **PCA** | directions of maximum variance | dimensionality reduction | eigenvectors of covariance |
| **SVD** | $A = U\Sigma V^T$ (any matrix) | recommendations, compression, PCA | $U \Sigma V^T$ |
| **Tensor** | multi-dimensional array | data in PyTorch/TensorFlow | shape $(b, c, h, w)$ |
| **LoRA** | low-rank update $W + BA$ | efficient LLM fine-tuning | $\Delta W = BA$ |

---

# Part 4: WHAT TO READ NEXT (inside this same math folder)

- **calculus.md** — the gradients and loss functions that matrices carry; backpropagation is this document + calculus combined.
- **probability.md** — softmax, expectation, and the probabilistic view of models (which are linear-algebra machines).
- **statistics.md** — covariance matrices, PCA from the statistical side, and the normal equation derived via MLE.
