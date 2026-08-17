---
title: Probability — Quick Revision
description: 10-minute revision of probability for ML — PMF/PDF/CDF, Bayes, all key distributions, entropy, Monte Carlo in one page.
tags: [math, probability, quick-rev, distributions, bayes, entropy]
---

# PROBABILITY — QUICK REVISION

> Condensed from the full guide at `MLDL/maths/probability.md`. For revision: scan, recall, quiz yourself.

## THE BIG PICTURE (3 lines)

1. **Probability** = size of event ÷ size of sample space (equally likely case). "AND" → multiply, "OR" (disjoint) → add.
2. **Random variable** = outcome mapped to a number. All ML (inputs, targets, predictions) is random variables.
3. **Bayes** = how to update beliefs with evidence — the engine of Naive Bayes, generative models, Bayesian ML.

## NOTATION QUICK-REF

| Symbol | Meaning |
| :--- | :--- |
| $\Omega$, $\omega$, $A$ | sample space, outcome, event |
| $P(A \mid B)$ | probability of A given B |
| $X \sim \text{Bin}(n, p)$ | X follows binomial(n, p) |
| $P(X = x)$ | PMF (discrete, exact prob) |
| $f(x)$ | PDF (continuous, density) |
| $F(x) = P(X \le x)$ | CDF |
| $E[X]$, $\text{Var}(X)$ | expectation, variance |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian |
| $H(P)$, $D_{KL}$ | entropy, KL divergence |

## THE FOUNDATION FORMULAS

| Concept | Formula | Memory hook |
| :--- | :--- | :--- |
| Probability | $P(A) = \frac{|A|}{|\Omega|}$ | favorable / total |
| Complement | $P(A^c) = 1 - P(A)$ | — |
| Union | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | subtract double-count |
| Permutation | $\frac{n!}{(n-r)!}$ | order matters |
| Combination | $\binom{n}{r} = \frac{n!}{r!(n-r)!}$ | order doesn't (÷ r!) |
| Marginal | $P(X) = \sum_y P(X, y)$ | collapse a table |
| Conditional | $P(Y \mid X) = \frac{P(X, Y)}{P(X)}$ | joint ÷ marginal |
| Independence | $P(A, B) = P(A)P(B)$ | knowing A tells nothing |
| Chain rule | $P(X_1,\dots,X_n) = \prod P(X_i \mid \text{prev})$ | **this is GPT** |
| Total probability | $P(A) = \sum_i P(A \mid B_i)P(B_i)$ | weighted slices |
| **Bayes** | $P(Y \mid X) = \frac{P(X \mid Y)P(Y)}{P(X)}$ | prior × likelihood ÷ evidence |
| Expectation | $E[X] = \sum x\,P(x)$ (or $\int x f(x)dx$) | long-run average |
| Variance | $\text{Var}(X) = E[(X - E[X])^2]$ | squared spread |
| Covariance | $\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]$ | co-movement |

**Bayes worked example (30 sec):** 1% disease, test 90% sensitive, 5% false positive. Positive → $P(\text{disease}\mid+) = \frac{0.90 \times 0.01}{0.90(0.01) + 0.05(0.99)} = \frac{0.009}{0.0585} \approx 0.154$ — only 15.4%! Rare disease → most positives are false.

## PMF vs PDF vs CDF (the #1 confusion — nail it)

| | PMF (discrete) | PDF (continuous) | CDF (both) |
| :--- | :--- | :--- | :--- |
| Output at point $x$ | exact probability | density (can exceed 1) | $P(X \le x)$ |
| $P(X = x)$ at one point | may be > 0 | always 0 | — |
| Total | bars sum to 1 | area under curve = 1 | $F(\infty) = 1$ |
| Get probabilities by | adding bars | integrating | reading the curve |

**Example:** $X \sim \text{Uniform}[0,4]$ → $f = \frac{1}{4}$, $P(1 \le X \le 3) = \text{width} \times \text{height} = 2 \times \frac{1}{4} = 0.5$.

## THE DISTRIBUTION FAMILY (memorize mean & variance)

| Distribution | Models | $P(X=k)$ or $f(x)$ | Mean | Var | ML use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bernoulli** | 1 trial, 2 outcomes | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | binary classification |
| **Binomial** | #successes in n trials | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | CTR, A/B tests |
| **Categorical** | 1 trial, K outcomes | $p_i$ | — | — | softmax targets |
| **Poisson** | rare events in a window | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | count data |
| **Exponential** | wait time at rate $\lambda$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | survival, decay |
| **Uniform** | all equal | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | init, random search |
| **Gaussian** | bell curve | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | everywhere |
| **Beta** | a probability (0–1) | $p^{\alpha-1}(1-p)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | — | priors, A/B tests |
| **Dirichlet** | a probability vector | multi-dim Beta | — | — | LDA, mixture priors |

**68–95–99.7 rule (Gaussian):** $\mu \pm 1\sigma$ → 68%, $\mu \pm 2\sigma$ → 95%, $\mu \pm 3\sigma$ → 99.7%. Beyond 3σ = extreme outlier (anomaly detection).

**Poisson vs Exponential:** Poisson = *how many* events; Exponential = *how long until next*. Same $\lambda$, two questions.
**Beta vs Dirichlet:** Beta = one probability; Dirichlet = whole probability vector (Beta with K=2).
**Bernoulli vs Binomial:** Bernoulli = 1 trial; Binomial = count over n trials (n=1 ⟹ same).

## SIGMOID vs SOFTMAX

| | Sigmoid | Softmax |
| :--- | :--- | :--- |
| Classes | 2 | K |
| Output | one number in (0,1) | vector summing to 1 |
| Formula | $\frac{1}{1+e^{-z}}$ | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |

**Worked softmax (30 sec):** logits $[2.0, 1.0, 0.1]$ → $[7.39, 2.72, 1.11]$ → ÷ 11.22 → $[0.66, 0.24, 0.10]$ — sums to 1 ✓. Exponent amplifies gaps: logit 2.0 is 2× of 1.0 but probability is ~2.7×.

## ENTROPY FAMILY (the loss functions)

| Measure | Formula | Question it answers | ML use |
| :--- | :--- | :--- | :--- |
| **Entropy** | $H(P) = -\sum P \log P$ | how uncertain is P? | data complexity |
| **Cross-entropy** | $H(P,Q) = -\sum P \log Q$ | cost of model Q vs truth P | **classification loss** |
| **KL divergence** | $D_{KL}(P\parallel Q) = \sum P \log \frac{P}{Q}$ | extra bits paying for Q | VAEs, distillation |

**Golden relation:** $H(P,Q) = H(P) + D_{KL}(P \parallel Q)$ — minimizing cross-entropy = minimizing KL (the $H(P)$ part is fixed by data).

**Worked (30 sec):** fair coin → $H = 1$ bit. Biased 0.9/0.1 → 0.47 bits (less uncertain). One-hot true label + model output 0.66 → loss $= -\log_2 0.66 = 0.6$ bits.

## SAMPLING & INFERENCE

**Monte Carlo:** approximate intractable expectations by averaging samples: $E[f(X)] \approx \frac{1}{N}\sum f(x_i)$. Error shrinks like $1/\sqrt{N}$ (100× samples → 10× better). π example: points in square, $\pi \approx 4 \times \frac{\text{inside}}{\text{total}}$.

**Law of Large Numbers:** $\bar{x}_n \to E[X]$ as $n \to \infty$ — why big data + averaging works.

**Variational inference (1 line):** replace intractable posterior with closest Gaussian: $Q^* = \arg\min_Q D_{KL}(Q \parallel P)$ — this is what a VAE's KL loss does.

## WHERE PROBABILITY APPEARS IN ML (one-line map)

| Concept | ML location |
| :--- | :--- |
| Conditional $P(Y \mid X)$ | discriminative models (classifiers) |
| Joint $P(X, Y)$ | generative models (diffusion, GAN) |
| Chain rule | GPT / autoregressive language models |
| Bayes | Naive Bayes, Bayesian NN, diffusion reverse step |
| Bernoulli / Gaussian | binary / regression loss derivation |
| Categorical | softmax output layer |
| Entropy / cross-entropy / KL | every classification loss, VAEs |
| Monte Carlo | RL, sampling, uncertainty |
| LLN | empirical risk minimization (training ≈ true loss) |

## TOP 5 COMMON MISTAKES

1. Using a PDF like a PMF: asking "what is P(X = 5)?" for a continuous variable — answer is 0; integrate instead.
2. Assuming correlation 0 = independent — only rules out *linear* relationships (parabola: r ≈ 0 but dependent).
3. Forgetting the prior in Bayes — the medical test example: 90% accuracy ≠ 90% probability.
4. Adding probabilities of overlapping events without subtracting the intersection.
5. Multiplying probabilities of *dependent* events — use the chain rule ($P(A,B) = P(A)P(B\mid A)$).

> Full detail + worked examples: `MLDL/maths/probability.md` — then `statistics` (MLE derives the losses), `calculus` (integrals for PDFs), `linear-algebra` (softmax in matrix form).