---
title: Linear Algebra for Kids (10-year-old friendly)
description: Linear algebra explained like a story — arrows, toy grids, and photo-compressing machines. No scary algebra. Just pictures, tables, and one idea at a time.
tags: [math, kids, linear-algebra, easy, beginner]
---

# LINEAR ALGEBRA FOR KIDS

> This document is written for someone who is **10 years old** and has **never done math beyond basic numbers**. Every idea comes with a picture, a story, and real numbers you can check yourself. Read it top to bottom. If a symbol looks scary, the "say it" note tells you how to say it out loud — and that's all you need.

---

## The whole story in 10 lines (read this first!)

1. A **vector** is an arrow made of numbers (like "go 3 right, 2 up").
2. Arrows can be added, stretched, and measured.
3. The **dot product** is "how much two arrows agree".
4. A **matrix** is a grid of numbers that turns arrows into other arrows.
5. **Matrix multiplication** = doing one turn, then another.
6. The **determinant** = how much a matrix stretches the area.
7. The **inverse** = the "undo" button.
8. **Eigenvectors** = arrows that only get stretched, not turned.
9. **PCA and SVD** = squeezing big piles of data into small ones (photo compression!).
10. A **tensor** is just a box of numbers with more sides.

---

# PART 0: THINGS YOU ALREADY KNOW (with a quick warm-up)

## 0.1 Coordinates — the map game

Remember treasure maps? Start at the corner (0, 0). "Go 3 right, 2 up" lands you at a spot. We write that spot as $(3, 2)$:

```
y
3 │
2 │          ● (3,2)
1 │        ╱
0 │───────╱──────── x
  0  1  2  3
```

- First number = how far **right** (the $x$ direction)
- Second number = how far **up** (the $y$ direction)
- $(3, 2)$ is NOT the same as $(2, 3)$ — order matters! (3 right then 2 up ≠ 2 right then 3 up.)

**That's the entire coordinate system.** One number per direction, always in order.

## 0.2 Lists of numbers

A **list of numbers** is written like this:

$$(3, 2) \quad \text{or} \quad \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

Both mean the same list: 3, then 2. (Grown-ups sometimes stack the numbers in a column instead of a row — same list, two ways of writing it.)

**Why lists matter:** computers store everything as lists of numbers. A photo is a giant list. A song is a list. A house price prediction uses a list of features: (size, rooms, age).

## 0.3 The distance formula (you already know it!)

How far is it from (0,0) to (3,4)? Use the triangle trick (grown-ups call it **Pythagoras**):

$$\text{distance} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**STEP 1:** square each number: $3^2 = 9$, $4^2 = 16$.
**STEP 2:** add: $9 + 16 = 25$.
**STEP 3:** take the square root: $\sqrt{25} = 5$.
**CHECK:** the arrow (3,4) is exactly 5 units long. (You can check with a ruler on graph paper!)
**WHAT IT MEANS:** "length of an arrow" = square each step, add, square-root. This one rule appears everywhere in ML.

## 0.4 Notation table — the only 6 symbols you need

| Symbol | Say it | Means |
| :--- | :--- | :--- |
| $x$ (plain) | x | one single number |
| $\mathbf{x}$ (bold) | x | a list of numbers (a vector) |
| $\mathbb{R}^n$ | R to the n | "all lists of n numbers" — the whole space |
| $A$ (big letter) | A | a grid of numbers (a matrix) |
| $A^T$ | A transpose | the grid flipped: rows ↔ columns |
| $\|\mathbf{x}\|$ | norm of x | the length of the arrow |

**That's it.** Everything else in this document is built from these six.

> **ONE-LINE POINT of Part 0:** A vector is an ordered list of numbers = an arrow. Length = square, add, square-root.

---

# PART 1: VECTORS — arrows made of numbers

## 1.1 What is a vector? (say: **"vek-tor"**)

A **vector** is a list of numbers that points somewhere. $\mathbf{x} = (3, 2)$ means "3 right, 2 up" — an arrow from the start to the spot (3,2).

![Vector basics](/maths-images/linalg-vector-basics.png)

**Real-world vectors you already know:**
- A character in a game: position = (x, y) — that's a 2D vector.
- A player in Minecraft: (x, y, z) — a 3D vector.
- A robot's guess about a house: (size, rooms, age) — a 3D vector too!
- ChatGPT's idea of a word: a vector with **hundreds of numbers** — one number per "meaning dial".

That last one is the big deal: in ML, *everything* is a vector. Words, photos, sounds — all turned into lists of numbers.

## 1.2 Adding vectors — walking one arrow, then the other

**GIVEN:** $\mathbf{a} = (2, 1)$ and $\mathbf{b} = (1, 3)$.
**STEP 1:** add the first numbers: $2 + 1 = 3$.
**STEP 2:** add the second numbers: $1 + 3 = 4$.
**STEP 3:** result: $\mathbf{a} + \mathbf{b} = (3, 4)$.
**CHECK:** walk arrow a, then from where it ends walk arrow b — you end at (3,4). ✓
**WHAT IT MEANS:** adding arrows = add matching numbers = "go a, then go b".

![Vector addition](/maths-images/linalg-vector-add.png)

## 1.3 Stretching vectors (scaling)

Multiply an arrow by a number, and it gets longer (or shorter, or flipped):

| Arrow | Multiply by 2 | Multiply by 0.5 | Multiply by −1 |
| :--- | :--- | :--- | :--- |
| $(2, 1)$ | $(4, 2)$ | $(1, 0.5)$ | $(-2, -1)$ |

**The rule:** multiply **every number** in the list. The arrow keeps its direction, just changes length (multiplying by −1 flips it to point backwards).

![Vector scaling](/maths-images/linalg-vector-scale.png)

> **ONE-LINE POINT:** vectors add by matching numbers; scaling multiplies every number.

---

# PART 2: MEASURING ARROWS

## 2.1 Length (the norm, say: **"norm"**)

The **norm** $\|\mathbf{x}\|$ is the arrow's length — the distance formula from Part 0:

$$\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \dots}$$

**GIVEN:** $\mathbf{x} = (3, 4)$.
**STEP 1:** square each part: $9 + 16$.
**STEP 2:** add: $25$.
**STEP 3:** square root: $5$.
**ANSWER:** $\|\mathbf{x}\| = 5$.

**Why it matters in ML:** if a robot's guess vector is nearly the same length as the true answer, it's probably a good guess. Lengths measure "how big is this thing, direction ignored".

## 2.2 Distance between two arrows

Distance = length of the arrow connecting them = subtract, then take the norm:

**GIVEN:** $\mathbf{a} = (1, 2)$, $\mathbf{b} = (4, 6)$.
**STEP 1:** subtract: $(4-1, 6-2) = (3, 4)$.
**STEP 2:** norm of (3,4) = 5 (from above).
**ANSWER:** distance = 5.

**Why it matters in ML:** "find the photo most similar to this one" = "find the photo with the smallest distance to it". Robots do this all day.

## 2.3 The dot product — how much do the arrows agree?

The **dot product** (say: **"dot product"**) multiplies matching numbers and adds them up:

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots$$

**GIVEN:** $\mathbf{a} = (1, 2)$, $\mathbf{b} = (3, 4)$.
**STEP 1:** multiply first numbers: $1 \times 3 = 3$.
**STEP 2:** multiply second numbers: $2 \times 4 = 8$.
**STEP 3:** add: $3 + 8 = 11$.
**ANSWER:** $\mathbf{a} \cdot \mathbf{b} = 11$.

![Dot product](/maths-images/linalg-dot-product.png)

**The secret meaning:** the dot product is biggest when the arrows **point the same way**, zero when they're at right angles, and negative when they point opposite ways. It's an "agreement meter".

**GIVEN:** two arrows, $\mathbf{a} = (1, 0)$ (points right), $\mathbf{b} = (0, 1)$ (points up).
**STEP 1:** $1 \times 0 = 0$; $0 \times 1 = 0$.
**STEP 2:** sum $= 0$.
**WHAT IT MEANS:** right and up agree on nothing → dot product 0. Grown-ups say they are **orthogonal** (say: **"or-thog-uh-null"**) — at right angles.

**Why it matters in ML:** "how similar are these two words?" = "how big is the dot product of their word-vectors?" Big dot product → similar meaning. This single operation is the engine of search, recommendations, and ChatGPT itself.

## 2.4 A quick way to see similarity: cosine

The dot product grows with length too. To measure *direction agreement only* (ignoring length), grown-ups divide by the lengths. That's the **cosine similarity** — a number between −1 and 1:

- $1$ → arrows point the exact same way
- $0$ → arrows at right angles (no agreement)
- $-1$ → arrows point opposite ways

**You don't need to compute it — just know what it means.**

> **ONE-LINE POINT:** Norm = length. Distance = length of the difference. Dot product = agreement meter. Cosine = agreement ignoring length.

---

# PART 3: MATRICES — grids that turn arrows

## 3.1 What is a matrix? (say: **"may-tricks"**)

A **matrix** is a grid of numbers with rows and columns:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

- Row 1: $1, 2$. Row 2: $3, 4$.
- We say "A is a 2 by 2 matrix" (2 rows, 2 columns).
- The number in row 2, column 1 is written $A_{21} = 3$ (say: **"A two-one"**).

**Why it matters:** a matrix is a *turn-machine*. You feed it an arrow, it spits out a new arrow. Turn-machines can rotate, stretch, squish, and flip. When a robot learns, it's really just adjusting the numbers inside its matrices — billions of tiny adjustments.

![Matrix transforms arrows](/maths-images/linalg-matrix-vector.png)

## 3.2 How a matrix turns one arrow (matrix × vector)

**GIVEN:** matrix $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, arrow $\mathbf{x} = (2, 1)$.

The output is a new arrow. Here's the recipe:

**STEP 1:** take row 1 of the matrix, dot it with the arrow: $1 \times 2 + 2 \times 1 = 4$. That's the first number of the output.
**STEP 2:** take row 2, dot it with the arrow: $3 \times 2 + 4 \times 1 = 10$. That's the second number.

**ANSWER:** $A \mathbf{x} = (4, 10)$.

**In words:** *each row of the matrix is a pair of scales; each scale weighs the arrow and reports its own number.* The arrow (2,1) became the arrow (4,10).

**CHECK:** you can verify with the "walk" picture: the machine stretched and turned the arrow. Draw (2,1) and (4,10) on graph paper — the second is longer and points differently. ✓

## 3.3 Matrices × matrices — doing one turn, then another

Turning arrows twice (machine B, then machine A) can be done in one step using **matrix multiplication**. The rule:

**To get the number in row $i$, column $j$ of the answer: dot row $i$ of the first matrix with column $j$ of the second.**

**GIVEN:** $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$.

**STEP 1:** top-left of the answer: dot row 1 of A with column 1 of B: $1 \times 5 + 2 \times 7 = 19$.
**STEP 2:** top-right: row 1 of A with column 2 of B: $1 \times 6 + 2 \times 8 = 22$.
**STEP 3:** bottom-left: row 2 of A with column 1 of B: $3 \times 5 + 4 \times 7 = 43$.
**STEP 4:** bottom-right: row 2 of A with column 2 of B: $3 \times 6 + 4 \times 8 = 50$.

**ANSWER:** $AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$.

**WHAT IT MEANS:** this grid does "turn with B, then turn with A" in one step. Matrix multiplication = combining turn-machines.

![Matrix multiplication](/maths-images/linalg-matmul.png)

> **ONE-LINE POINT:** A matrix is a turn-machine. Matrix × arrow = each row weighs the arrow. Matrix × matrix = two turns in one.

---

# PART 4: DETERMINANT AND INVERSE — stretch factor and undo button

## 4.1 The determinant (say: **"dee-ter-min-ant"**) — how much area stretches

Feed the matrix any square shape. The square gets squished into a parallelogram. The **determinant**, written $\det(A)$, is the **stretch factor**: how many times bigger the area became.

![Determinant = area stretch factor](/maths-images/linalg-det-area.png)

**The recipe for a 2×2 matrix:**

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = a \cdot d - b \cdot c$$

Say it out loud: **"a d minus b c"** — multiply the two main-corner numbers, subtract the product of the other two corners.

**GIVEN:** $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.
**STEP 1:** main corners: $1 \times 4 = 4$.
**STEP 2:** other corners: $2 \times 3 = 6$.
**STEP 3:** $4 - 6 = -2$.
**ANSWER:** $\det(A) = -2$.

**WHAT IT MEANS:** the machine flips the shape over (that's the minus sign) and doubles the area (that's the 2).

**The three most important facts:**
- $\det > 0$ → stretches (and doesn't flip)
- $\det = 0$ → **squishes everything flat** — the machine destroyed information! (More below.)
- $\det < 0$ → stretches AND flips

> **ONE-LINE POINT:** determinant = "how much does the grid stretch?" If it's zero, the machine squished everything flat.

## 4.2 The inverse (say: **"in-verse"**) — the undo button

The **inverse** of a matrix, written $A^{-1}$ (say: **"A inverse"**), is the machine that **undoes** whatever $A$ did. Turn with A, then turn with A⁻¹ → you're back where you started.

**GIVEN:** the turn-machine $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and an arrow $\mathbf{x} = (2, 1)$.
**STEP 1:** turn with A: $A\mathbf{x} = (4, 10)$ (we did this before).
**STEP 2:** turn the result with $A^{-1}$: you get back $(2, 1)$. ✓
**WHAT IT MEANS:** inverse = undo. Like rewinding a video.

**The catch — when there's NO undo button:** if $\det(A) = 0$, the machine squished the whole plane flat, and there's no way to un-squish it. The inverse **does not exist**. Grown-ups say the matrix is **singular** (say: **"sing-gyu-lar"**). One arrow went in, one flat spot came out — two different arrows can't be separated anymore.

**Why it matters in ML:** solving "find $\mathbf{x}$ so that $A\mathbf{x} = \mathbf{b}$" = "turn $\mathbf{b}$ backwards": $\mathbf{x} = A^{-1}\mathbf{b}$. If the inverse doesn't exist, the problem has no unique answer — often a sign the data is broken or duplicated.

> **ONE-LINE POINT:** inverse = undo button. Determinant zero = no undo possible = information destroyed.

---

# PART 5: EIGENSTUFF — arrows that don't turn

## 5.1 Eigenvectors (say: **"eye-gen-vek-tors"**)

Most arrows get turned AND stretched by a matrix. But **some special arrows only get stretched** — their direction doesn't change. Those are the **eigenvectors**. The stretch amount is the **eigenvalue** (say: **"eye-gen-val-yoo"**) — how many times longer the arrow became.

![Eigenvectors only stretch](/maths-images/linalg-eigenvectors.png)

**The rubber-band story:** imagine the matrix is a rubber sheet you pull. A circle drawn on it becomes an oval. The *longest* direction of the oval and the *shortest* direction of the oval are the eigenvectors — the only directions that stay as lines instead of turning. The eigenvalues are "how much stretched" and "how much squished".

**GIVEN:** matrix $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$.
**STEP 1:** feed in the arrow (1, 0) (points right): $A(1,0) = (2, 0)$ — still points right, twice as long. So (1,0) is an eigenvector with eigenvalue 2.
**STEP 2:** feed in the arrow (0, 1) (points up): $A(0,1) = (0, 3)$ — still points up, 3× longer. Eigenvector with eigenvalue 3.
**CHECK:** any other arrow, like (1,1), becomes (2,3) — different direction! So it's NOT an eigenvector.
**WHAT IT MEANS:** eigenvectors = the "special directions" of a machine that only get stretched.

**Why it matters in ML:** eigenvectors find the *most important directions* in piles of data. That's exactly what PCA does (next section).

> **ONE-LINE POINT:** eigenvectors = arrows that only stretch, not turn. Eigenvalue = how much they stretch.

---

# PART 6: PCA AND SVD — squeezing data small

## 6.1 The problem: photos are huge

A small photo is a grid of pixels, each with 3 numbers (red, green, blue). A $1000 \times 1000$ photo = **3 million numbers**. Sending, storing, and learning from 3 million numbers is slow. Can we keep almost all the important information with far fewer numbers? **Yes — that's PCA and SVD.**

## 6.2 PCA (say: **"P-C-A"**) — find the most important direction

**PCA = Principal Component Analysis.** The idea: look at a cloud of data dots, find the direction where they spread out the most (the "longest direction"), then tilt your view so that direction becomes the new "left-right" axis.

![PCA finds the main directions of the data](/maths-images/linalg-pca.png)

**The silhouette story:** a shadow on a wall loses depth but keeps the outline. A smart shadow angle keeps the *most* information. PCA picks the best angle to view your data so that most of the variation survives.

**The steps (you recognize, not compute):**
**STEP 1:** find the data's most spread-out direction → that's the first **principal component** (an eigenvector!).
**STEP 2:** find the next most spread-out direction at right angles to it → second component.
**STEP 3:** keep the top few components (say 2 out of 1000), throw away the rest.
**WHAT IT MEANS:** you compressed the data — the top components carry most of the "shape" of the data.

**Real use:** face recognition robots compress each face to ~100 numbers instead of millions, then compare faces using only those numbers.

## 6.3 SVD (say: **"S-V-D"**) — the Swiss army knife

**SVD = Singular Value Decomposition.** It takes ANY grid and writes it as three grids multiplied together:

$$A = U \Sigma V^T$$

Say it out loud: **"A equals U, sigma, V transpose"**. (The $\Sigma$ here is the capital Greek letter **sigma** — in SVD it's a diagonal grid holding the "stretch amounts".)

**The meaning:** every matrix secretly does: **turn → stretch → turn back.**

**STEP 1:** $V^T$ turns the space so the special directions line up with the axes.
**STEP 2:** $\Sigma$ stretches along those directions (the stretch amounts!).
**STEP 3:** $U$ turns the space again to the final orientation.

**Why it's the Swiss army knife:** the stretch amounts in $\Sigma$ tell you which directions matter. Keep only the biggest ones, throw away the tiny ones → **perfect compression**, like PCA but for any shape of data. Recommender systems (Netflix "you might like...") and photo compression use it.

> **ONE-LINE POINT:** PCA = view data from the most informative angle. SVD = "turn, stretch, turn back" — with the stretch amounts telling you what to keep.

---

# PART 7: TENSORS — boxes of numbers (bonus)

A **tensor** (say: **"ten-sor"**) is just a box of numbers with any number of sides:

| Name | Shape | Example |
| :--- | :--- | :--- |
| Scalar | 1 number | $5$ |
| Vector | 1 list | $(3, 2)$ |
| Matrix | 1 grid | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| 3D tensor | 1 cube of numbers | a photo: (width, height, color) |
| 4D tensor | a stack of cubes | a batch of photos: (which photo, width, height, color) |

**The rule:** a tensor with $n$ sides has $n$ numbers in its "address". A photo's pixel at (row 40, column 70, red) is one number in a 3D tensor.

**Why it matters in ML:** PyTorch and TensorFlow store *everything* as tensors. When a robot trains on a batch of 32 photos, that's one big 4D tensor. Don't fear the name — it's just "a box of numbers".

![Tensor shapes](/maths-images/linalg-tensors.png)

> **ONE-LINE POINT:** scalar = 1 number, vector = 1 list, matrix = 1 grid, tensor = box with more sides. All just containers for numbers.

---

# THE ONE-PAGE GLOSSARY (print this!)

| Word | Say it | Means |
| :--- | :--- | :--- |
| vector $\mathbf{x}$ | vek-tor | an arrow made of a list of numbers |
| norm $\|\mathbf{x}\|$ | norm | length of the arrow |
| dot product $\mathbf{a} \cdot \mathbf{b}$ | dot product | agreement meter between arrows |
| orthogonal | or-thog-uh-null | at right angles (agreement = 0) |
| matrix $A$ | may-tricks | a grid of numbers = a turn-machine |
| matrix multiplication | — | combining two turn-machines |
| determinant $\det(A)$ | dee-ter-min-ant | how much the area stretches |
| inverse $A^{-1}$ | A inverse | the undo button |
| singular | sing-gyu-lar | no undo possible (squished flat) |
| eigenvector | eye-gen-vek-tor | arrow that only stretches, never turns |
| eigenvalue | eye-gen-val-yoo | how much the eigenvector stretches |
| PCA | P-C-A | find the most spread-out directions of data |
| SVD | S-V-D | turn → stretch → turn back; keeps the big stretches |
| tensor | ten-sor | a box of numbers with any number of sides |

**Final check — can you answer these?**
1. Add the vectors (2, 1) and (1, 3).
2. What's the length of (3, 4)?
3. Dot product of (1, 2) and (3, 4)?
4. What does $\det = 0$ mean?
5. What is an eigenvector?

(Answers: 1. (3, 4). 2. 5. 3. 11. 4. The machine squished everything flat — no undo. 5. An arrow that only gets stretched, not turned.)