---
title: Probability for Machine Learning
description: Complete beginner-friendly probability for ML — sample spaces, random variables, PMF/PDF/CDF, Bayes, distributions, entropy, Monte Carlo, with worked numeric examples and visualizations.
tags: [math, probability, random-variables, distributions, bayes, entropy, monte-carlo, ml]
---

# PROBABILITY FOR MACHINE LEARNING

> This document is **fully self-contained**. You do not need to search the internet, open another textbook, or guess anything. Every symbol is defined, every formula is derived step by step, every example shows the full arithmetic, and every concept has a picture. Read it top to bottom.

---

# Part 0: PREREQUISITES — read this first, nothing is skipped

Probability is the mathematics of *uncertainty* — and machine learning is entirely about making predictions under uncertainty. Before the probability itself, you need four small building blocks. They are reviewed below **in full**.

---

## 0.1 Sets and set operations (the language of events)

A **set** is a collection of things, written in curly braces: $\{1, 2, 3, 4, 5, 6\}$.

**The three operations you need (all have pictures in Module 1):**

- **Union** $A \cup B$ = everything in $A$ **or** $B$ (or both). Example: $A = \{1, 2\}$, $B = \{2, 3\}$ → $A \cup B = \{1, 2, 3\}$.
- **Intersection** $A \cap B$ = everything in $A$ **and** $B$. Example: $A \cap B = \{2\}$.
- **Complement** $A^c$ = everything *not* in $A$. Example: within $\{1,2,3\}$, if $A = \{1\}$ then $A^c = \{2, 3\}$.

**Disjoint (mutually exclusive):** two sets with no overlap: $A \cap B = \varnothing$ (empty set). Example: "roll a 3" and "roll a 5" are disjoint — one roll can't be both.

**Why this matters:** probability *is* set theory with numbers attached. "The probability of A or B" is literally "the size of $A \cup B$ divided by the size of the whole space."

---

## 0.2 Fractions and percentages (the arithmetic of probability)

- A probability is always a number between 0 and 1: $0 \le P \le 1$.
- Convert between forms freely: $\frac{1}{4} = 0.25 = 25\%$.
- **Multiplying probabilities:** the chance of two *independent* things both happening = product of their probabilities (see Module 3). Example: two coin flips both heads = $\frac{1}{2} \times \frac{1}{2} = \frac{1}{4} = 25\%$.
- **Adding probabilities (disjoint events):** chance of *either* of two mutually exclusive events = sum. Example: roll a die, chance of 1 *or* 6 = $\frac{1}{6} + \frac{1}{6} = \frac{2}{6} = \frac{1}{3}$.

**The two golden rules of probability arithmetic — memorize:**
1. "**AND**" (independent) → **multiply**
2. "**OR**" (disjoint) → **add**

---

## 0.3 Factorials and counting (needed for permutations and combinations)

**Factorial:** $n!$ = the product of all integers from $n$ down to 1.

- $3! = 3 \times 2 \times 1 = 6$
- $5! = 5 \times 4 \times 3 \times 2 \times 1 = 120$
- By definition: $0! = 1$ (the empty product).

**Why factorials matter:** they count *arrangements*. If you have 3 different books, the number of orders to arrange them on a shelf is $3! = 6$:

```
ABC  ACB  BAC  BCA  CAB  CBA
```

**Basic counting principle:** if a choice has $a$ options and the next choice has $b$ options, together they have $a \times b$ options. Example: 3 shirts × 2 pants = 6 outfits.

---

## 0.4 Summation notation $\sum$ (how to write "add everything")

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n$$

Read: *"sum from i equals 1 to n of x sub i"* = add up all the $x_i$'s.

**Worked example:** if $x_1 = 3, x_2 = 5, x_3 = 2$, then $\sum_{i=1}^{3} x_i = 3 + 5 + 2 = 10$.

**Where it appears:** expected value $E[X] = \sum x \cdot P(X=x)$ (Module 2) is "multiply each value by its probability, then add everything."

---

## 0.5 Notation table — every symbol used in this document

**Essential now (Modules 1–2):**

| Symbol | Name | Meaning | Example |
| :--- | :--- | :--- | :--- |
| $\Omega$ | sample space | all possible outcomes | $\{1,2,3,4,5,6\}$ for a die |
| $\omega$ | outcome | one specific result | rolling a 4 |
| $A$, $B$ | events | subsets of outcomes you care about | "even number" = $\{2,4,6\}$ |
| $P(A)$ | probability of A | chance A happens | $P(\text{even}) = 0.5$ |
| $P(A \mid B)$ | conditional | A given B happened | $P(\text{rain} \mid \text{clouds})$ |
| $X$ | random variable | outcome mapped to a number | $X$ = number of heads |
| $P(X = x)$ | PMF | exact probability (discrete) | $P(X=2) = 0.25$ |
| $f(x)$ | PDF | density (continuous) | — |
| $F(x)$ | CDF | $P(X \le x)$ | — |

**Reference later (Modules 3–5):**

| Symbol | Name | Meaning | First appears |
| :--- | :--- | :--- | :--- |
| $P(A \cup B)$ | union | A or B (or both) | Module 1.2 |
| $P(A \cap B)$ | intersection | A and B | Module 1.2 |
| $P(A^c)$ | complement | not A | Module 1.2 |
| $E[X]$ | expectation | average value | Module 2.4 |
| $\text{Var}(X)$ | variance | spread around the mean | Module 2.4 |
| $\sigma$ | std dev | $\sqrt{\text{Var}}$ | Module 2.4 |
| $\text{Cov}(X,Y)$ | covariance | how X and Y move together | Module 2.4 |
| $\mu$, $\sigma^2$ | mean, variance | parameters of a Gaussian | Module 4.7 |
| $\lambda$ | lambda | rate parameter | Module 4.4, 4.5 |
| $H(P)$ | entropy | uncertainty in bits | Module 5.2 |
| $D_{KL}$ | KL divergence | information lost using Q for P | Module 5.2 |
| $\log$ / $\ln$ | logarithms | see Calculus doc prerequisites | Module 5.2 |

---

# Part 1: The Roadmap — where this document is going

```
                            PROBABILITY FOR ML
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
[MODULE 1]                      [MODULE 2]                      [MODULE 3]
Probability Foundations         Random Variables                 Conditional Probability
  ├── Sample Space & Events       ├── Discrete vs Continuous       ├── Independence & Chain Rule
  ├── Axioms & Set Operations     ├── PMF, PDF, CDF                ├── Law of Total Probability
  └── Counting (Perm/Comb)        ├── Joint, Marginal, Conditional └── Bayes' Theorem
                                  └── Expectation, Variance,           (updating beliefs)
                                      Covariance
                                    │
    ┌───────────────────────────────┘
    ▼
[MODULE 4]                      [MODULE 5]
Key Distributions in ML         Probabilistic ML & Information Theory
  ├── Bernoulli & Binomial        ├── Entropy, Cross-Entropy, KL
  ├── Categorical & Poisson       ├── Softmax vs Sigmoid
  ├── Uniform & Gaussian          ├── Monte Carlo Sampling
  ├── Exponential                 ├── Law of Large Numbers
  └── Beta & Dirichlet            └── Variational Inference & GenAI
```

**How to use this roadmap:** Module 1 is the vocabulary of chance. Module 2 turns outcomes into *numbers* (random variables) — this is where ML connects. Module 3 is the *engine*: conditional probability and Bayes power Naive Bayes, generative models, and Bayesian deep learning. Module 4 gives you the actual shapes data follows. Module 5 is the modern toolkit: information theory (loss functions) and sampling (how machines compute the impossible).

---

# Part 2: COMPREHENSIVE EXPLANATION

---

# MODULE 1: PROBABILITY FOUNDATIONS

---

## 1.1 Sample Space, Outcomes, and Events

### What is it?

Three nested ideas:

- **Outcome ($\omega$):** a single possible result of a random experiment. Example: rolling a die gives outcomes $1, 2, 3, 4, 5, 6$.
- **Sample space ($\Omega$):** the *set of all* possible outcomes. For a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$.
- **Event ($A$):** any subset of outcomes you are interested in — "the event A happens" means *the outcome landed inside A*. Example: $A$ = "roll an even number" = $\{2, 4, 6\}$.

**The definition of probability (the one you use 90% of the time):** for equally likely outcomes,

$$P(A) = \frac{\text{number of outcomes in } A}{\text{number of outcomes in } \Omega}$$

### Worked example (every number shown)

Roll one fair die.
- $\Omega = \{1, 2, 3, 4, 5, 6\}$, so $|\Omega| = 6$ outcomes.
- Event $A$ = "even number" = $\{2, 4, 6\}$ → 3 outcomes.
- $P(A) = \frac{3}{6} = \frac{1}{2} = 0.5 = 50\%$.

**Check the rules:** $0 \le P(A) \le 1$ ✓. The probability that *something* happens: $P(\Omega) = \frac{6}{6} = 1 = 100\%$ ✓.

### Where, why, how in ML

- **Where:** the output layer of a classifier. A model predicting an image as Dog/Cat/Bird must output probabilities over the *complete* sample space that sum to 1 (e.g. 0.70, 0.20, 0.10).
- **Why:** the model's job is to estimate $P(\text{class} \mid \text{image})$ — an event probability.
- **How:** softmax (Module 5) converts raw scores into valid probabilities over the class sample space.

### How outcomes, sample space, and events differ (zero confusion table)

| Term | Scale | Example (die) |
| :--- | :--- | :--- |
| Outcome $\omega$ | one result | the die shows 4 |
| Sample space $\Omega$ | all results | $\{1,2,3,4,5,6\}$ |
| Event $A$ | a set of results you care about | $\{2,4,6\}$ = "even" |

*An event is never "one thing that happens" — it is a SET of outcomes. Saying "event A" is shorthand for "the outcome belongs to set A."*

---

## 1.2 The Axioms of Probability & Set Operations

### What is it?

The **three axioms of probability** (Kolmogorov) are the rules every probability system obeys — think of them as the laws of physics for chance:

1. **Non-negativity:** $P(A) \ge 0$ — no negative probabilities, ever.
2. **Normalization:** $P(\Omega) = 1$ — the chance that *something* in the sample space happens is 100%.
3. **Additivity:** for mutually exclusive events $A$ and $B$: $P(A \cup B) = P(A) + P(B)$ — disjoint events' chances add.

**Two useful facts derived from the axioms (you'll use these constantly):**
- Complement rule: $P(A^c) = 1 - P(A)$ (something either happens or it doesn't).
- General union rule (works even when NOT disjoint): $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ — subtract the overlap once, so it isn't double-counted.

### Worked examples (full arithmetic)

**Example 1 — complement:** if $P(\text{rain tomorrow}) = 0.3$, then $P(\text{no rain}) = 1 - 0.3 = 0.7$.

**Example 2 — union of disjoint events:** $P(\text{roll a 1}) = \frac{1}{6}$, $P(\text{roll a 6}) = \frac{1}{6}$. Disjoint → $P(1 \text{ or } 6) = \frac{1}{6} + \frac{1}{6} = \frac{2}{6} = \frac{1}{3}$.

**Example 3 — union of overlapping events:** in a class, 60% of students drink coffee ($A$), 50% drink tea ($B$), 30% drink both ($A \cap B$). Then $P(\text{coffee or tea}) = 0.6 + 0.5 - 0.3 = 0.8$. (Adding 0.6 + 0.5 = 1.1 double-counts the 0.3 who drink both — subtract it once.)

![Venn diagrams: intersection, union, complement, disjoint](/maths-images/prob-venn.png)

### Where, why, how in ML

- **Where:** multi-label classification, normalizing model outputs, debugging loss functions.
- **Why:** a neural network's probabilities must form a *valid distribution* (non-negative, sum to 1) — otherwise cross-entropy loss (Module 5) becomes meaningless.
- **How:** softmax guarantees the axioms; any hand-written probability estimate must be checked against them.

### How the union rule differs from the additivity axiom

- **Axiom 3** only works for *disjoint* events (no overlap).
- **General union rule** handles *any* events by subtracting the double-counted intersection. For disjoint events, $P(A \cap B) = 0$, and the general rule collapses into the axiom — one rule, two forms.

---

## 1.3 Counting: Permutations vs. Combinations

### What is it?

Sometimes the sample space is too big to list by hand, and we must *count* it. Two classic counting problems:

- **Permutation (order MATTERS):** how many ways to pick and ARRANGE $r$ items from $n$?

$$P(n, r) = \frac{n!}{(n - r)!}$$

- **Combination (order does NOT matter):** how many ways to pick a SET of $r$ items from $n$?

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!\,(n - r)!}$$

**The only difference:** the combination divides by $r!$ to remove the redundant orderings. If order matters → permutation. If order doesn't matter → combination.

### Worked examples (full arithmetic)

**Example 1 — permutations:** 5 students, pick 3 to be president, vice-president, treasurer (order matters!):

$$P(5, 3) = \frac{5!}{(5-3)!} = \frac{5!}{2!} = \frac{120}{2} = 60$$

**Check by logic:** 5 choices for president, then 4 for VP, then 3 for treasurer: $5 \times 4 \times 3 = 60$ ✓.

**Example 2 — combinations:** 5 students, pick 3 to form a committee (order doesn't matter):

$$C(5, 3) = \binom{5}{3} = \frac{5!}{3!\,2!} = \frac{120}{6 \times 2} = \frac{120}{12} = 10$$

**Why so much smaller (60 → 10):** each committee of 3 has $3! = 6$ internal orderings that the permutation counted separately; dividing by 6 removes them.

**Example 3 — probability via counting:** flip 3 coins. Sample space = $\{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}$ → 8 outcomes. The event "exactly 2 heads" = $\{HHT, HTH, THH\}$ → 3 outcomes. $P = \frac{3}{8} = 0.375$. (Equivalently: $C(3, 2) = 3$ ways to choose which 2 of 3 coins are heads.)

### Where, why, how in ML

- **Where:** the binomial distribution (Module 4) uses $\binom{n}{k}$ directly; sampling plans, experimental design, feature subset selection.
- **Why:** counting is how discrete probabilities are computed — "number of favorable outcomes divided by total."
- **How:** the binomial PMF $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$ = (ways to choose the k successes) × (chance of those k) × (chance of the rest).

### How permutations differ from combinations (decision table)

| | Permutation | Combination |
| :--- | :--- | :--- |
| Order matters? | YES | NO |
| Formula | $\frac{n!}{(n-r)!}$ | $\frac{n!}{r!(n-r)!}$ |
| Example question | "arrange a podium of 3" | "choose a committee of 3" |
| Relationship | — | combination = permutation ÷ $r!$ |

---

# MODULE 2: RANDOM VARIABLES & JOINT DISTRIBUTIONS

---

## 2.1 Random Variables — turning random outcomes into numbers

### What is it?

A **random variable** $X$ is a rule that assigns a number to every outcome of a random experiment. It's not "a variable that is random" — it's a *function from outcomes to numbers*.

**Discrete random variable:** takes countable, separate values. Example: $X$ = number of heads in 3 coin flips → values $\{0, 1, 2, 3\}$.

**Continuous random variable:** takes any value in an interval. Example: $X$ = height of a randomly chosen person → any value like $172.45$ cm.

| | Discrete $X$ | Continuous $X$ |
| :--- | :--- | :--- |
| Values | countable (gaps between) | uncountable (smooth) |
| Probability math | sums (PMF) | integrals (PDF) |
| ML examples | class labels, word counts | prices, heights, losses |

**Worked example — mapping outcomes to numbers:** roll two dice; let $X$ = the sum. Outcome $(1, 3) \to X = 4$, outcome $(6, 6) \to X = 12$. The random variable turned 36 outcomes into the values $2$ through $12$, each with its own probability.

### Where, why, how in ML

- **Where:** model inputs $X$, targets $Y$, predictions $\hat{Y}$ are all random variables. Classification targets are discrete ($Y \in \{0, 1\}$); regression targets are continuous ($Y \in \mathbb{R}$).
- **Why:** ML models *estimate the distribution* of $Y$ given $X$ — probabilities live on random variables.
- **How:** training data is a set of *samples* $(x_i, y_i)$ drawn from the joint distribution $P(X, Y)$ — the model learns the pattern between them.

---

## 2.2 PMF, PDF, and CDF — the three descriptions of a random variable

### What is it?

Three functions describe everything about a random variable. **They answer three different questions — this is the #1 confusion in all of probability, so read carefully:**

**PMF — Probability Mass Function (for DISCRETE $X$):**
$$P(X = x)$$
- Gives the *exact probability* of each value.
- Picture: a **bar chart**; bar heights ARE probabilities; all heights sum to 1.
- Valid because discrete values have "mass" at exact points.

**PDF — Probability Density Function (for CONTINUOUS $X$):**
$$f(x)$$
- Gives *density* (probability per unit of $x$), NOT probability.
- **The probability of any exact single value is 0** (a single point has zero width → zero area).
- Probabilities are *areas* over intervals: $P(a \le X \le b) = \int_a^b f(x)\,dx$ (see Calculus doc Module 5).
- Picture: a **smooth curve**; the total area under it = 1.

**CDF — Cumulative Distribution Function (BOTH types):**
$$F(x) = P(X \le x)$$
- Gives the probability of *everything up to and including* $x$.
- Picture: **staircase** for discrete, **smooth S-curve** for continuous.
- Always: $F(-\infty) = 0$, $F(+\infty) = 1$, and $F$ never decreases.

![PMF, PDF, CDF side by side for discrete and continuous variables](/maths-images/prob-pmf-pdf-cdf.png)

> **TL;DR:** PMF = bar heights = exact probabilities (discrete). PDF = curve height = density, NOT probability (continuous). CDF = area up to x = cumulative probability. **For continuous: P(exact value) = 0, only ranges have probability.**

**The Dartboard Analogy (why area = probability for continuous):**
```
Discrete: throw dart at numbered bins (1, 2, 3...)
          probability = hits on bin / total throws
Continuous: throw dart at a NUMBER LINE
            chance of hitting EXACTLY 2.00000... = 0 (line has no width)
            probability = area of region / total area
            The PDF's height = "how crowded is this region"
```

### Worked examples (full arithmetic)

**Example 1 — discrete PMF:** $X$ = number of heads in 2 coin flips:

| $x$ | $0$ | $1$ | $2$ |
| :--- | :--- | :--- | :--- |
| $P(X = x)$ | $0.25$ | $0.50$ | $0.25$ |

Check: $0.25 + 0.50 + 0.25 = 1$ ✓. CDF: $F(1) = P(X \le 1) = 0.25 + 0.50 = 0.75$.

**Example 2 — continuous PDF:** $X$ uniform on $[0, 4]$, so $f(x) = \frac{1}{4}$ (constant). Then $P(1 \le X \le 3) = \text{area} = \text{width} \times \text{height} = 2 \times \frac{1}{4} = 0.5$. And $P(X = 2)$ exactly = 0 (zero-width point).

### How PMF differs from PDF (the table to memorize)

| | PMF (discrete) | PDF (continuous) |
| :--- | :--- | :--- |
| Output at a point | a probability | a density |
| $P(X = x)$ at one point | can be > 0 | always 0 |
| Everything adds/integrates to | 1 (sum of bars) | 1 (area under curve) |
| Compute probabilities by | adding bars | integrating |
| Analogy | counting steps on a staircase | moving along a smooth ramp |

### Where, why, how in ML

- **Where:** classification models output a PMF over classes (softmax); anomaly detection fits a PDF to normal data; ROC curves use CDFs.
- **Why:** knowing which description applies decides whether you *sum* or *integrate* — using a PDF like a PMF (asking "P(X = 5)?") is a category error that leads to nonsense answers.
- **How:** anomaly detection example — fit a Gaussian PDF to normal transaction amounts; a new transaction landing where the PDF is near 0 (extreme tail) is flagged as fraud.

---

## 2.3 Joint, Marginal, and Conditional Distributions — the full family tree

### What is it?

When two random variables $X$ and $Y$ are related, we describe them with four related ideas:

- **Joint probability** $P(X, Y)$: the chance of $X = x$ AND $Y = y$ simultaneously (a whole table).
- **Marginal probability** $P(X)$: the chance of $X$ alone, found by *summing the joint table over all values of Y* ("marginalizing out" Y):

$$P(X = x) = \sum_{y} P(X = x, Y = y)$$

- **Conditional probability** $P(Y \mid X)$: the chance of $Y$ given that $X$ is already known:

$$P(Y \mid X) = \frac{P(X, Y)}{P(X)}$$

**The family rule (memorize):** *joint = marginal × conditional* — i.e. $P(X, Y) = P(X) \cdot P(Y \mid X)$.

### Worked example — one table, all three concepts

$X$ = weather {Rainy, Sunny}, $Y$ = traffic {Heavy, Light}. The joint table:

| $P(X, Y)$ | Heavy | Light | marginal $P(X)$ |
| :--- | :--- | :--- | :--- |
| **Rainy** | 0.30 | 0.21 | **0.51** |
| **Sunny** | 0.25 | 0.24 | **0.49** |
| marginal $P(Y)$ | **0.55** | **0.45** | **1.00** |

**Step 1 — marginals:** $P(\text{Rainy}) = 0.30 + 0.21 = 0.51$. $P(\text{Heavy}) = 0.30 + 0.25 = 0.55$. (Rows and columns each sum to 1.00 — the whole table sums to 1 ✓.)

**Step 2 — conditionals:** $P(\text{Heavy} \mid \text{Rainy}) = \frac{P(\text{Rainy, Heavy})}{P(\text{Rainy})} = \frac{0.30}{0.51} \approx 0.588$. $P(\text{Heavy} \mid \text{Sunny}) = \frac{0.25}{0.49} \approx 0.510$.

**Step 3 — interpretation:** if you know it's rainy, the chance of heavy traffic is 58.8%; if sunny, 51.0%. **Knowing the condition changed the probability** — weather and traffic are not independent (Module 3.1).

![Joint probability table with marginals](/maths-images/prob-joint-table.png)

### Where, why, how in ML

- **Where:** the fundamental split of ML model types.
  - **Discriminative models** learn the *conditional* $P(Y \mid X)$ directly — logistic regression, neural classifiers. They answer "given the image, what's the class?"
  - **Generative models** learn the *joint* $P(X, Y)$ or the data distribution $P(X)$ — diffusion models, GANs, VAEs. They can *create* new samples because they know how the whole data space is distributed.
- **Why:** the choice of which distribution to learn determines what the model can do — discriminative = classify, generative = generate.
- **How:** classifiers' softmax outputs ARE $P(Y \mid X)$ estimates; diffusion models estimate $P(X)$ (the distribution of all images).

### How joint, marginal, and conditional differ (one table)

| | Joint $P(X, Y)$ | Marginal $P(X)$ | Conditional $P(Y \mid X)$ |
| :--- | :--- | :--- | :--- |
| Question | X AND Y | X alone | Y GIVEN X |
| Computed by | counting pairs | summing joint over Y | dividing joint by marginal |
| Size | table (all pairs) | one row/column | table of ratios |

---

## 2.4 Expectation, Variance, and Covariance — summarizing random variables

### What is it?

Three summary numbers describe a random variable's *center*, *spread*, and *co-movement*:

**Expected value (the average):**
$$E[X] = \sum_x x \cdot P(X = x) \quad \text{(discrete)}, \qquad E[X] = \int x\, f(x)\, dx \quad \text{(continuous)}$$

*Weight every value by its probability and add. The "long-run average" — repeat the experiment forever and average the results.*

**Variance (the spread):**
$$\text{Var}(X) = E[(X - E[X])^2] = \sum_x (x - \mu)^2 \cdot P(X = x)$$

*Average squared distance from the mean. Standard deviation $\sigma = \sqrt{\text{Var}(X)}$ brings it back to original units.*

**Covariance (the co-movement):**
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

- Positive: above-average $X$ goes with above-average $Y$.
- Negative: above-average $X$ goes with below-average $Y$.
- Near zero: no *linear* relationship.

### Worked examples (full arithmetic)

**Example 1 — expectation of a die:** $X$ = die roll, all $P = \frac{1}{6}$:

$$E[X] = 1\cdot\frac{1}{6} + 2\cdot\frac{1}{6} + 3\cdot\frac{1}{6} + 4\cdot\frac{1}{6} + 5\cdot\frac{1}{6} + 6\cdot\frac{1}{6} = \frac{21}{6} = 3.5$$

**Example 2 — variance of a die:** deviations from $\mu = 3.5$ are $-2.5, -1.5, -0.5, 0.5, 1.5, 2.5$; squared: $6.25, 2.25, 0.25, 0.25, 2.25, 6.25$:

$$\text{Var} = \frac{6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25}{6} = \frac{17.5}{6} \approx 2.92$$

$$\sigma = \sqrt{2.92} \approx 1.71$$

**Example 3 — covariance sign:** study hours vs. test scores — students who study more tend to score higher → Cov > 0. (Same data, scatter plots below.)

![Covariance signs: positive, negative, zero](/maths-images/prob-covariance-signs.png)

### Where, why, how in ML

- **Where:** every loss function (minimizing expected loss = *empirical risk minimization*), reinforcement learning (maximize expected reward), feature analysis (covariance matrix → PCA).
- **Why:** ML training literally minimizes the *expected* loss over the data distribution: $\min_\theta E_{(x,y)\sim D}[\mathcal{L}(y, f_\theta(x))]$. In practice the expectation is approximated by the average over the training set — the empirical expectation.
- **How:** variance tells you how much predictions spread (related to model uncertainty); covariance matrices drive PCA and data whitening.

### How variance differs from covariance

| | Variance $\text{Var}(X)$ | Covariance $\text{Cov}(X, Y)$ |
| :--- | :--- | :--- |
| Variables | one | two |
| Question | how much does X spread? | do X and Y move together? |
| Relation | $\text{Var}(X) = \text{Cov}(X, X)$ | symmetric: Cov(X,Y) = Cov(Y,X) |

---

# MODULE 3: CONDITIONAL PROBABILITY & INFERENCE

---

## 3.1 Independence and the Chain Rule — when events don't talk to each other

### What is it?

**Independence:** events $A$ and $B$ are independent if knowing one gives *zero* information about the other:

$$P(A \cap B) = P(A) \cdot P(B) \quad \text{(equivalent form: } P(A \mid B) = P(A))$$

**Chain rule (the general way to decompose a joint distribution):**

$$P(X_1, X_2, \dots, X_n) = P(X_1) \cdot P(X_2 \mid X_1) \cdot P(X_3 \mid X_1, X_2) \cdots P(X_n \mid X_1, \dots, X_{n-1})$$

*Every joint probability can be sliced into a product of conditionals, one per variable.*

### Worked examples (full arithmetic)

**Example 1 — independent coin flips:** $P(\text{head}) = 0.5$ each flip. Two flips: $P(H, H) = 0.5 \times 0.5 = 0.25$. Knowing flip 1 was heads tells you nothing about flip 2 — independent. ✓

**Example 2 — dependent events:** $P(\text{cloudy}) = 0.4$, $P(\text{rain}) = 0.2$. If rain *only happens when cloudy*: $P(\text{rain} \mid \text{cloudy}) = 0.5 \ne P(\text{rain}) = 0.2$ → dependent. And $P(\text{cloudy, rain}) = P(\text{cloudy}) \cdot P(\text{rain} \mid \text{cloudy}) = 0.4 \times 0.5 = 0.2$ (chain rule in action).

**Example 3 — the chain rule IS language modeling:** an LLM predicts sentence probability one word at a time:

$$P(\text{"I love ML"}) = P(\text{"I"}) \cdot P(\text{"love"} \mid \text{"I"}) \cdot P(\text{"ML"} \mid \text{"I love"})$$

This is exactly how GPT generates text — next word given all previous words.

### Where, why, how in ML

- **Where:** autoregressive language models (GPT family), Naive Bayes, probability model factorization.
- **Why:** the chain rule turns one giant impossible joint probability (over 50,000 vocabulary words) into a sequence of small conditional predictions — one word at a time.
- **How:** each token prediction is a softmax over the vocabulary conditioned on all prior tokens — precisely the chain rule's factors.

### How independence differs from zero correlation (the subtle trap)

![Independent vs correlated vs dependent-but-uncorrelated](/maths-images/prob-independence-correlation.png)

- **Independent** → no relationship of *any* kind → correlation is definitely 0.
- **Zero correlation** → only no *linear* relationship — a curved relationship (like a parabola) can have correlation 0 yet be *strongly dependent*!
- **Rule:** independence is stronger. Correlation 0 does NOT imply independence; independence DOES imply correlation 0.

---

## 3.2 The Law of Total Probability — breaking a probability into pieces

### What is it?

If events $B_1, B_2, \dots, B_k$ are a **partition** (they cover the whole sample space and never overlap), then:

$$P(A) = P(A \mid B_1)\,P(B_1) + P(A \mid B_2)\,P(B_2) + \cdots + P(A \mid B_k)\,P(B_k)$$

**Plain words:** the total chance of $A$ = the chance of $A$ within each slice, weighted by the slice's size, all added up.

### Worked example (full arithmetic)

A factory has 2 machines: M1 makes 70% of products, M2 makes 30%. M1 produces 2% defective; M2 produces 5% defective. What's the overall defect rate?

- Partition: every product comes from M1 or M2: $P(M1) = 0.7$, $P(M2) = 0.3$.
- Conditionals: $P(\text{defect} \mid M1) = 0.02$, $P(\text{defect} \mid M2) = 0.05$.
- Total: $P(\text{defect}) = (0.02)(0.7) + (0.05)(0.3) = 0.014 + 0.015 = 0.029 = 2.9\%$.

**Tree diagram of the same calculation:**

```
                        start
                       /      \
              P=0.7  /        \  P=0.3
                    M1          M2
                  /    \      /    \
              good  defect  good  defect
              0.98   0.02   0.95   0.05
                    │             │
              0.7×0.02=0.014  0.3×0.05=0.015
                    └───── 0.014+0.015 = 0.029 ─────┘
```

### Where, why, how in ML

- **Where:** the *denominator* of Bayes' theorem (next section), ensemble model reasoning, mixture models.
- **Why:** when you can't measure $P(A)$ directly but can measure it within subgroups, total probability assembles it from the pieces.
- **How:** computing $P(\text{evidence})$ in Bayes — the normalizing constant — is always a total-probability sum over the hypotheses.

---

## 3.3 Bayes' Theorem — updating beliefs with evidence

### What is it?

**Bayes' theorem** reverses a conditional probability — it answers: *"given the evidence I observed, how likely is each cause?"*

$$P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$$

**The four pieces (memorize these names):**
- $P(Y)$ — **prior**: your belief *before* seeing evidence.
- $P(X \mid Y)$ — **likelihood**: how likely the evidence is *if* the hypothesis is true.
- $P(X)$ — **evidence** (marginal): how likely the evidence is *overall* (usually via the law of total probability).
- $P(Y \mid X)$ — **posterior**: the updated belief *after* seeing evidence.

**The story:** start with a prior belief, observe evidence, and Bayes tells you the new belief. Repeat as more evidence arrives — this is *Bayesian updating*, the mathematical heartbeat of Naive Bayes, Bayesian neural networks, and modern generative models.

### Worked example — medical testing (full arithmetic)

A disease affects 1% of the population: $P(\text{disease}) = 0.01$. A test is 90% accurate on sick people: $P(\text{positive} \mid \text{disease}) = 0.90$. The test has a 5% false-positive rate: $P(\text{positive} \mid \text{healthy}) = 0.05$.

**Question: you test positive. What is $P(\text{disease} \mid \text{positive})$?**

**Step 1 — the prior:** $P(\text{disease}) = 0.01$, so $P(\text{healthy}) = 0.99$.

**Step 2 — the evidence via total probability:**
$$P(\text{positive}) = P(\text{positive}\mid\text{disease})P(\text{disease}) + P(\text{positive}\mid\text{healthy})P(\text{healthy})$$
$$= (0.90)(0.01) + (0.05)(0.99) = 0.009 + 0.0495 = 0.0585$$

**Step 3 — Bayes:**
$$P(\text{disease} \mid \text{positive}) = \frac{0.009}{0.0585} \approx 0.154 = 15.4\%$$

**The famous insight:** even with a 90%-accurate test, a positive result means only ~15% chance of disease — because the disease is rare (1%), so most positives come from the 5% false-positive rate among the 99% healthy people. **The prior matters enormously.** This is why doctors don't order rare-disease tests without symptoms.

**Alternative intuition — count 10,000 people (exactly matches the formula):**
- 100 are sick (1% of 10,000). 90% test positive → **90 true positives**.
- 9,900 are healthy (99% of 10,000). 5% false positive → **495 false positives**.
- Total positive results = 90 + 495 = 585.
- $P(\text{disease} \mid \text{positive}) = \frac{90}{585} \approx 0.154 = 15.4\%$ ✓

*Both the formula and the counting method now give exactly 15.4% — no discrepancy!*

### Where, why, how in ML

- **Where:** Naive Bayes classifiers, Bayesian inference, Kalman filters, generative AI (diffusion's reverse process is literally Bayes' theorem).
- **Why:** ML models must constantly combine *prior knowledge* with *new evidence* — Bayes is the only principled formula for doing so.
- **How:** a spam filter: prior $P(\text{spam})$ from history, likelihood $P(\text{"winner"} \mid \text{spam})$ from word statistics, posterior $P(\text{spam} \mid \text{"winner"})$ = the filtered probability.

### How Bayes differs from the law of total probability

- **Total probability** builds the *evidence* from slices (bottom-up assembly).
- **Bayes** reverses the arrow: from evidence *back to* the hypothesis (top-down attribution). Bayes *uses* total probability for its denominator — one is the other's ingredient.

---

# MODULE 4: KEY PROBABILITY DISTRIBUTIONS IN ML

A **distribution** is the complete description of a random variable — every value and its probability (or density). Each distribution below gets: *what it models, its formula, its mean & variance, a numeric example, a picture, and where ML uses it.*

---

## 4.1 Bernoulli — one trial, two outcomes

**Models:** a single experiment with two outcomes (success = 1, failure = 0), with success probability $p$.

**Formula (PMF):** $P(X = 1) = p$, $P(X = 0) = 1 - p$.

**Mean and variance:** $E[X] = p$, $\text{Var}(X) = p(1-p)$.

**Worked example:** $p = 0.7$ (e.g. 70% chance an email is spam): $P(X = 1) = 0.7$, $P(X = 0) = 0.3$. Mean 0.7, variance $0.7 \times 0.3 = 0.21$.

![Bernoulli: two bars](/maths-images/prob-bernoulli.png)

**ML use:** binary classification — logistic regression's output is $p = P(\text{class 1})$, and its loss (binary cross-entropy) is derived *from* the Bernoulli distribution (see Statistics doc, MLE).

---

## 4.2 Binomial — counting successes in n trials

**Models:** the *number of successes* $k$ in $n$ independent Bernoulli trials.

**Formula (PMF):**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

*$\binom{n}{k}$ ways to choose which trials succeed × chance those succeed × chance the rest fail.*

**Mean and variance:** $E[X] = np$, $\text{Var}(X) = np(1-p)$.

**Worked example:** 10 coin flips, count heads ($n = 10$, $p = 0.5$). Probability of exactly 7 heads:

$$P(X = 7) = \binom{10}{7} (0.5)^7 (0.5)^3 = 120 \times 0.0078125 \times 0.125 = 0.117$$

($\binom{10}{7} = \frac{10!}{7!3!} = \frac{3628800}{5040 \times 6} = 120$.)

![Binomial(10, 0.5): bell-like bars](/maths-images/prob-binomial.png)

**ML use:** click-through rate modeling (ad impressions → clicks), A/B testing, ensemble voting analysis.

**How Bernoulli differs from Binomial:** Bernoulli = ONE trial; Binomial = the COUNT of successes in MANY Bernoulli trials. If $n = 1$, Binomial *is* Bernoulli.

---

## 4.3 Categorical — one trial, K categories

**Models:** a single trial with $K$ possible categories, each with its own probability.

**Formula (PMF):** $P(X = \text{category } i) = p_i$, with $\sum_i p_i = 1$.

**Mean and variance:** given by the probability vector itself.

**Worked example:** image classifier: $p = (0.70 \text{ dog}, 0.20 \text{ cat}, 0.10 \text{ bird})$ — sums to 1 ✓.

**ML use:** the target distribution for multi-class classification — the softmax layer's output IS a categorical distribution.

**How Categorical differs from Bernoulli:** Bernoulli = 2 categories; Categorical = K categories (Bernoulli is Categorical with $K = 2$).

---

## 4.4 Poisson — counting rare events in a fixed interval

**Models:** the number of *rare events* occurring in a fixed window of time/space, given an average rate $\lambda$ (e.g. calls per hour, defects per batch).

**Formula (PMF):**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Mean and variance (a neat signature):** $E[X] = \lambda$, $\text{Var}(X) = \lambda$.

**Worked example:** a website gets $\lambda = 3$ visits per minute on average. Probability of exactly 5 visits in a minute:

$$P(X = 5) = \frac{3^5 e^{-3}}{5!} = \frac{243 \times 0.0498}{120} = \frac{12.10}{120} \approx 0.101$$

![Poisson(3): skewed bars](/maths-images/prob-poisson.png)

**ML use:** count data modeling, anomaly detection on event rates, queueing/wait-time models.

**How Poisson differs from Binomial:** Binomial counts successes in a *fixed* number of trials; Poisson counts events in a *fixed window* with no trial limit. For large $n$ and small $p$, Binomial ≈ Poisson with $\lambda = np$.

---

## 4.5 Exponential — the waiting time until the next event

**Models:** the *time until the next event* when events occur at constant rate $\lambda$ (the continuous partner of Poisson).

**Formula (PDF):**
$$f(x) = \lambda e^{-\lambda x} \quad (x \ge 0)$$

**Mean and variance:** $E[X] = \frac{1}{\lambda}$, $\text{Var}(X) = \frac{1}{\lambda^2}$.

**Worked example:** calls arrive at rate $\lambda = 0.5$ per minute. Expected wait for the next call: $\frac{1}{0.5} = 2$ minutes. Probability of waiting more than 3 minutes: $P(X > 3) = e^{-0.5 \times 3} = e^{-1.5} \approx 0.223$.

![Exponential: decaying density](/maths-images/prob-exponential.png)

**ML use:** survival analysis, time-between-events modeling, exponential decay in learning rates.

**How Exponential differs from Poisson:** Poisson counts *how many* events; Exponential measures *how long until the next one*. Same $\lambda$, two questions.

---

## 4.6 Uniform — everything equally likely

**Models:** all values in an interval $[a, b]$ equally likely (continuous), or all $k$ outcomes equally likely (discrete).

**Formula (PDF):** $f(x) = \frac{1}{b - a}$ for $x \in [a, b]$, else 0.

**Mean and variance:** $E[X] = \frac{a+b}{2}$, $\text{Var}(X) = \frac{(b-a)^2}{12}$.

**Worked example:** $U[0, 4]$: density $\frac{1}{4}$ (area = $4 \times \frac{1}{4} = 1$ ✓), mean 2, $P(1 \le X \le 3) = 2 \times \frac{1}{4} = 0.5$.

![Uniform: flat density](/maths-images/prob-uniform.png)

**ML use:** random weight initialization, random hyperparameter search (each candidate value equally plausible), data augmentation randomness.

---

## 4.7 Gaussian (Normal) — the most important distribution in ML

**Models:** symmetric bell-shaped variation around a center $\mu$ with spread $\sigma$.

**Formula (PDF):**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Mean and variance:** $E[X] = \mu$, $\text{Var}(X) = \sigma^2$.

**Notation:** $X \sim \mathcal{N}(\mu, \sigma^2)$. The *standard normal* is $\mathcal{N}(0, 1)$.

**Worked example:** heights ~ $\mathcal{N}(170, 8^2)$ cm. The density is highest at 170 (most people near average) and tapers off symmetrically. Probability of being between 162 and 178 (within ±1σ): about 68% (see below).

![Gaussians with different σ](/maths-images/prob-gaussian.png)

**The 68–95–99.7 rule (memorize):**

![Empirical rule shaded](/maths-images/prob-gaussian-689599.png)

| Range | % of data | Meaning |
| :--- | :--- | :--- |
| $\mu \pm 1\sigma$ | ~68% | most values are here |
| $\mu \pm 2\sigma$ | ~95% | almost all |
| $\mu \pm 3\sigma$ | ~99.7% | practically everything; beyond = extreme outlier |

**ML use:** weight initialization, Gaussian Naive Bayes, VAE latent spaces, anomaly detection (flag anything beyond $\pm 3\sigma$), noise models in diffusion.

**How Gaussian differs from Uniform:** Uniform = flat (every value equally likely); Gaussian = peaked (values near $\mu$ much more likely). Gaussian is the *limit* of many natural processes (see CLT in the Statistics doc) — which is why it's everywhere.

---

## 4.8 Beta and Dirichlet — distributions OVER probabilities

**Models:** Beta describes a *probability itself* (values between 0 and 1); Dirichlet is its multi-class generalization (probability *vectors*).

**Beta formula (PDF):**
$$f(p) \propto p^{\alpha - 1} (1 - p)^{\beta - 1} \quad (0 \le p \le 1)$$

**Mean:** $E[p] = \frac{\alpha}{\alpha + \beta}$.

![Beta with different shapes](/maths-images/prob-beta.png)

**Worked example — the coin-flip prior:** with $\alpha = 2, \beta = 2$, the mean is $\frac{2}{4} = 0.5$ — a neutral "fair coin" belief. After seeing 7 heads in 10 flips, updating makes the distribution peak near 0.7 (this *is* Bayesian updating — see Statistics doc, MAP).

**Dirichlet:** a distribution over vectors $(p_1, \dots, p_K)$ with $\sum p_i = 1$. It's the "probability over probability vectors" — parameterized by $\alpha_1, \dots, \alpha_K$.

![Dirichlet samples on the simplex](/maths-images/prob-dirichlet.png)

**ML use:** Beta — A/B testing priors, Bayesian hyperparameter tuning; Dirichlet — Latent Dirichlet Allocation (LDA topic modeling), mixture model priors.

**How Beta differs from Dirichlet:** Beta = one probability (2 parameters); Dirichlet = a whole probability vector (K parameters). Dirichlet with K = 2 is a Beta.

---

# MODULE 5: PROBABILISTIC ML & INFORMATION THEORY

---

## 5.1 Softmax and Sigmoid — turning raw scores into probabilities

### What is it?

Models output *raw scores* (logits) — any real numbers. **Softmax** converts them into a valid probability distribution over classes:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

- Exponentiating makes everything positive.
- Dividing by the total makes everything sum to 1.
- The exponential *amplifies differences*: the biggest logit gets a disproportionately large probability.

**Sigmoid** is the special case for *two* classes:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

### Worked example (full arithmetic)

Logits for dog/cat/bird: $z = (2.0, 1.0, 0.1)$.

**Step 1 — exponentiate:** $e^{2.0} = 7.39$, $e^{1.0} = 2.72$, $e^{0.1} = 1.11$.

**Step 2 — sum:** $7.39 + 2.72 + 1.11 = 11.22$.

**Step 3 — divide:**
- $P(\text{dog}) = \frac{7.39}{11.22} = 0.659$
- $P(\text{cat}) = \frac{2.72}{11.22} = 0.242$
- $P(\text{bird}) = \frac{1.11}{11.22} = 0.099$
- Check: $0.659 + 0.242 + 0.099 = 1.000$ ✓

![Softmax: logits → probabilities](/maths-images/prob-softmax.png)

**Notice:** dog's logit (2.0) is only twice cat's (1.0), but dog's probability (0.66) is ~2.7× cat's — the exponential stretched the gap. This is by design: models learn to push the right class's logit up.

### How softmax differs from sigmoid (zero confusion)

| | Sigmoid | Softmax |
| :--- | :--- | :--- |
| Classes | exactly 2 | K (any number) |
| Output | one probability $p$ | a whole probability vector |
| Connection | — | softmax with K=2 reduces to sigmoid (up to a scaling) |
| ML use | binary classification | multi-class classification |

---

## 5.2 Entropy, Cross-Entropy, and KL-Divergence — the information toolkit

### What is it?

Three related measures of *information and surprise* — these literally ARE the loss functions of deep learning.

**Entropy — uncertainty of a distribution:**
$$H(P) = -\sum_x P(x) \log_2 P(x)$$

*The average number of bits needed to encode outcomes of $P$. Maximum when all outcomes are equally likely (most uncertain); 0 when one outcome is certain.*

**Cross-Entropy — cost of using Q when truth is P:**
$$H(P, Q) = -\sum_x P(x) \log_2 Q(x)$$

*The average surprise when your model (Q) believes one thing and reality (P) is another. This is THE classification loss.*

**KL-Divergence — extra cost of Q relative to P:**
$$D_{KL}(P \parallel Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}$$

*How many extra bits you pay by using Q instead of P. Zero only when $P = Q$; always $\ge 0$.*

**The golden relation (memorize):**
$$H(P, Q) = H(P) + D_{KL}(P \parallel Q)$$

*Cross-entropy = intrinsic entropy of truth + the extra cost of the model's mistake. Minimizing cross-entropy = minimizing KL divergence (the $H(P)$ part is fixed).*

> **TL;DR:** Entropy = uncertainty (bits). Cross-entropy = cost of wrong model (classification loss). KL = extra cost vs truth (VAE loss). **Golden rule:** $H(P,Q) = H(P) + D_{KL}(P||Q)$ — minimizing cross-entropy = minimizing KL.

> **Why log₂ here, ln in PyTorch?**
> | Base | Unit | Use case | Conversion |
> | :--- | :--- | :--- | :--- |
> | $\log_2$ | **bits** | human-readable "questions to identify outcome" | 1 bit = ln(2) nats |
> | $\ln$ (base $e$) | **nats** | math-friendly, derivatives are cleaner | 1 nat = log₂(e) bits |
> 
> In formulas: use $\log_2$ for human intuition (bits). In code (PyTorch/TensorFlow): `F.cross_entropy` uses natural log → outputs in **nats**. To convert: divide nats by $\ln(2) \approx 0.693$ to get bits.

### Worked examples (full arithmetic)

**Example 1 — entropy of a fair coin:** $P = (0.5, 0.5)$:

$$H = -(0.5 \log_2 0.5 + 0.5 \log_2 0.5) = -(0.5(-1) + 0.5(-1)) = 1 \text{ bit}$$

A biased coin, $P = (0.9, 0.1)$: $H = -(0.9 \log_2 0.9 + 0.1 \log_2 0.1) = -(0.9(-0.152) + 0.1(-3.322)) = 0.137 + 0.332 = 0.469$ bits — *less* uncertainty (we're usually right).

![Binary entropy curve: max at p = 0.5](/maths-images/prob-entropy.png)

**Example 2 — cross-entropy loss in action:** true label is dog, encoded one-hot: $P = (1, 0, 0)$. Model outputs $Q = (0.66, 0.24, 0.10)$:

$$H(P, Q) = -1 \cdot \log_2(0.66) - 0 \cdot \log_2(0.24) - 0 \cdot \log_2(0.10) = -\log_2(0.66) = 0.60 \text{ bits}$$

If the model had been wrong ($Q = (0.10, 0.66, 0.24)$): $H = -\log_2(0.10) = 3.32$ bits — much bigger loss, as it should be. **Only the probability the model assigned to the TRUE class matters in the one-hot case.**

**Example 3 — KL divergence:** $P = (0.6, 0.2, 0.15, 0.05)$, $Q = (0.4, 0.3, 0.25, 0.05)$:

$$D_{KL} = 0.6 \log_2\frac{0.6}{0.4} + 0.2 \log_2\frac{0.2}{0.3} + 0.15 \log_2\frac{0.15}{0.25} + 0.05 \log_2\frac{0.05}{0.05}$$
$$= 0.6(0.585) + 0.2(-0.585) + 0.15(-0.737) + 0.05(0) = 0.351 - 0.117 - 0.111 = 0.123 \text{ bits}$$

![Cross-entropy: P vs Q and the per-class terms](/maths-images/prob-cross-entropy.png)

### How entropy, cross-entropy, and KL differ (the definitive table)

| | Entropy $H(P)$ | Cross-entropy $H(P, Q)$ | KL $D_{KL}(P\parallel Q)$ |
| :--- | :--- | :--- | :--- |
| Question | how uncertain is P? | how surprised by Q given truth P? | how much extra do I pay using Q? |
| Involves | P only | P and Q | P and Q |
| Value when P = Q | — | $H(P)$ | $0$ |
| ML use | data complexity | classification loss | VAE loss, distribution matching |

*Relationship: $H(P,Q) = H(P) + D_{KL}(P\parallel Q)$ — cross-entropy is entropy plus the penalty for being wrong.*

### Where, why, how in ML

- **Where:** cross-entropy is the default loss for *every* classifier; KL divergence powers VAEs and knowledge distillation; entropy estimates data complexity.
- **Why:** Euclidean distance doesn't respect probability structure (probabilities must stay non-negative and sum to 1) — information measures do.
- **How:** training a classifier = minimizing average cross-entropy over the batch, which drives the predicted Q toward the true P (the Statistics doc derives exactly why from MLE).

---

## 5.3 Monte Carlo Sampling — computing the impossible with random draws

### What is it?

Many expectations/integrals in ML cannot be computed exactly (no closed form). **Monte Carlo** approximates them by *drawing random samples and averaging*:

$$E[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i) \quad \text{where } x_i \text{ are samples from } P$$

**The raindrop analogy:** to measure an irregular lake's area, drop 10,000 random raindrops over a square containing it and count the fraction landing in the water. No calculus needed — just counting.

### Worked example — estimating π (full arithmetic)

**Setup:** random points uniformly in the square $[-1, 1] \times [-1, 1]$. The inscribed circle has area $\pi$; the square has area 4. So:

$$P(\text{inside circle}) = \frac{\pi}{4} \quad \Rightarrow \quad \pi \approx 4 \times \frac{\text{points inside}}{\text{total points}}$$

**Simulation (first few draws):** suppose 4000 points, 3146 inside → estimate $\pi \approx 4 \times \frac{3146}{4000} = 3.146$ (true π ≈ 3.14159 — 0.1% error from random sampling!).

![Monte Carlo π estimation: dots in a square](/maths-images/prob-montecarlo-pi.png)

**The convergence guarantee:** the error shrinks like $\frac{1}{\sqrt{N}}$ — 100× more samples → 10× smaller error.

![Estimate converging to π as N grows](/maths-images/prob-montecarlo-convergence.png)

### Where, why, how in ML

- **Where:** reinforcement learning (AlphaGo's Monte Carlo tree search), Bayesian deep learning (sampling weights), diffusion sampling, evaluating intractable expectations.
- **Why:** exact integration over a 100-billion-dimension parameter space is impossible; sampling is the only way.
- **How:** every diffusion model generates images by *sampling* the reverse process step by step — Monte Carlo, under the hood.

---

## 5.4 The Law of Large Numbers — why averages stabilize

### What is it?

The **Law of Large Numbers (LLN)** states: as the number of trials grows, the sample average converges to the true expected value:

$$\frac{1}{n}\sum_{i=1}^{n} x_i \to E[X] \quad \text{as } n \to \infty$$

*With few trials, luck dominates; with many trials, the pattern emerges.*

### Worked example — die rolls

Roll a die once: the average is 1, 4, or whatever — far from 3.5. Roll 5,000 times and keep a running average: it wanders at first, then homes in on 3.5 (the true mean) and stays there.

![Running average of die rolls converging to 3.5](/maths-images/prob-lln.png)

### Where, why, how in ML

- **Where:** the foundation of Monte Carlo (previous section), of empirical risk minimization (training loss ≈ expected loss when the dataset is big), and of the Central Limit Theorem (Statistics doc).
- **Why:** it justifies training on a *sample* of data: with enough samples, sample statistics approach true statistics.
- **How:** model accuracy on a large test set ≈ true generalization performance — the LLN in action.

---

## 5.5 Variational Inference & Generative AI (VAEs, Diffusion)

### What is it?

Generative models need to sample *new* data from a learned distribution. The classic roadblock: the true posterior $P(z \mid x)$ (latent cause of an image) is **intractable** — it cannot be computed exactly. **Variational inference** replaces it with a simpler, tractable distribution $Q(z \mid x)$ (usually a Gaussian) chosen to be as close as possible:

$$Q^* = \arg\min_Q D_{KL}(Q(z \mid x) \parallel P(z \mid x))$$

**How a VAE works (the concrete flow):**

1. **Encoder:** image $x$ → parameters of a Gaussian $Q(z \mid x)$ (mean + variance of the latent space).
2. **KL loss:** pull $Q(z \mid x)$ toward the standard Gaussian $\mathcal{N}(0, 1)$ — this *organizes* the latent space so similar images sit near each other and empty regions are smooth.
3. **Decoder:** sample a latent vector $z$ from the Gaussian and decode it back into an image.
4. **Generation:** draw $z \sim \mathcal{N}(0, 1)$ (pure noise from a standard Gaussian) and decode — a brand-new image, because smooth latent space means every $z$ maps to something plausible.

```
   x ──▶ ENCODER ──▶ Q(z|x) = N(μ, σ²) ──▶ sample z ──▶ DECODER ──▶ x̂ (reconstruction)
                        │  ▲
                        └──┘ KL loss pulls Q toward N(0,1)
   Generation:  z ~ N(0,1) ──▶ DECODER ──▶ brand-new image
```

> **Why KL pulls Q toward N(0,1) (the mechanics):**
> $$D_{KL}(Q \parallel \mathcal{N}(0,1)) = \mathbb{E}_Q[\log Q(z) - \log \mathcal{N}(0,1)(z)]$$
> $$= \underbrace{-\mathcal{H}(Q)}_{\text{entropy (wants Q spread out)}} + \underbrace{\tfrac{1}{2}\mathbb{E}_Q[z^2]}_{\text{2nd moment (wants Q near 0)}} + \text{const}$$
> Minimizing this makes Q have **high entropy** (spread out) + **small 2nd moment** (concentrated near 0).
> The only distribution that does both: $\mathcal{N}(0,1)$. ✓

**Diffusion models** do the same dance in reverse: their reverse process uses Bayes' theorem step by step, and each step is a Gaussian whose parameters the network predicts (see Calculus doc, Module 5.3 for the noise picture).

### How variational inference differs from exact inference

| | Exact inference | Variational inference |
| :--- | :--- | :--- |
| Computes | the true posterior | the *closest* tractable approximation |
| Feasible for | tiny problems | large deep models |
| Tool | algebra / integration | KL-divergence minimization |
| Cost | exact but impossible in practice | approximate but always computable |

### Where, why, how in ML

- **Where:** VAEs, latent diffusion (Stable Diffusion), Bayesian neural networks, topic models.
- **Why:** sampling from a messy, uncomputable posterior is impossible; sampling from a nice Gaussian is one line of code. Variational inference buys tractability at the cost of approximation.
- **How:** the VAE's total loss = reconstruction error + KL term — the KL term *is* variational inference in action.

---

# Part 3: SUMMARY CHEAT-SHEET

| Concept | Definition in one line | Primary ML application | Key formula |
| :--- | :--- | :--- | :--- |
| **Sample space / Event** | all outcomes / the subset you care about | classification output space | $P(A) = \frac{|A|}{|\Omega|}$ |
| **Axioms** | non-negativity, normalization, additivity | valid probability outputs | $P(\Omega) = 1$ |
| **Permutation** | arrangements, order matters | counting outcomes | $\frac{n!}{(n-r)!}$ |
| **Combination** | subsets, order doesn't matter | binomial, sampling | $\binom{n}{r} = \frac{n!}{r!(n-r)!}$ |
| **Random variable** | outcomes → numbers | inputs/targets/predictions | $X$ |
| **PMF** | exact probabilities (discrete) | class predictions | $P(X = x)$ |
| **PDF** | density (continuous); P(point) = 0 | density estimation, anomaly | $\int_a^b f\,dx$ |
| **CDF** | $P(X \le x)$ | ROC curves | $F(x) = P(X \le x)$ |
| **Marginal** | sum joint over other variable | extracting single-variable info | $P(X) = \sum_y P(X,y)$ |
| **Expectation** | long-run average | expected loss, RL rewards | $E[X] = \sum x\,P(x)$ |
| **Variance** | spread around mean | model uncertainty | $E[(X-\mu)^2]$ |
| **Covariance** | co-movement of two variables | feature analysis, PCA | $E[(X-\mu_X)(Y-\mu_Y)]$ |
| **Independence** | $P(A,B) = P(A)P(B)$ | factorization, Naive Bayes | — |
| **Chain rule** | joint = product of conditionals | autoregressive LLMs (GPT) | $P(X_1,\dots,X_n) = \prod P(X_i \mid \text{previous})$ |
| **Total probability** | weighted sum over slices | evidence for Bayes | $\sum P(A \mid B_i)P(B_i)$ |
| **Bayes' theorem** | reverse conditionals, update beliefs | Naive Bayes, Bayesian ML | $P(Y\mid X) = \frac{P(X\mid Y)P(Y)}{P(X)}$ |
| **Bernoulli** | 1 trial, 2 outcomes | binary classification | $P(1) = p$ |
| **Binomial** | count of successes in n trials | CTR, A/B tests | $\binom{n}{k}p^k(1-p)^{n-k}$ |
| **Categorical** | 1 trial, K outcomes | multi-class softmax targets | $P(i) = p_i$ |
| **Poisson** | rare events in a window | count data | $\frac{\lambda^k e^{-\lambda}}{k!}$ |
| **Exponential** | waiting time at rate λ | survival, decay | $\lambda e^{-\lambda x}$ |
| **Uniform** | all values equal | init, random search | $\frac{1}{b-a}$ |
| **Gaussian** | bell curve (μ, σ²) | everywhere in ML | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| **Beta / Dirichlet** | distributions over probabilities | priors, topic models | $p^{\alpha-1}(1-p)^{\beta-1}$ |
| **Sigmoid** | 2-class probability | binary classification | $\frac{1}{1+e^{-z}}$ |
| **Softmax** | K-class probability vector | multi-class output layer | $\frac{e^{z_i}}{\sum e^{z_j}}$ |
| **Entropy** | uncertainty of P | data complexity | $-\sum P\log P$ |
| **Cross-entropy** | cost of Q given truth P | classification loss | $-\sum P\log Q$ |
| **KL divergence** | extra cost of Q vs P | VAEs, distribution matching | $\sum P\log\frac{P}{Q}$ |
| **Monte Carlo** | approximate integrals by sampling | RL, Bayes, diffusion | $E[f] \approx \frac{1}{N}\sum f(x_i)$ |
| **Law of large numbers** | averages converge to means | why big data works | $\bar{x}_n \to E[X]$ |

---

# Part 4: WHAT TO READ NEXT (inside this same math folder)

- **statistics.md** — how MLE derives the loss functions introduced here (cross-entropy and MSE), plus confidence intervals and hypothesis testing built on these distributions.
- **calculus.md** — the integrals behind PDFs and expectations (Module 5) and the gradient descent that minimizes cross-entropy loss.
- **linear-algebra.md** — the vectors/matrices that carry probability distributions through neural networks (softmax, embeddings, attention).