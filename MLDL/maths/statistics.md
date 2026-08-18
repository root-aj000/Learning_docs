---
title: Statistics for Machine Learning
description: Complete beginner-friendly statistics for ML — descriptive statistics, standardization, correlation, CLT, confidence intervals, hypothesis testing, MLE, MAP, bootstrap, with worked numeric examples and visualizations.
tags: [math, statistics, descriptive, clt, hypothesis-testing, mle, bootstrap, ml]
---

# STATISTICS FOR MACHINE LEARNING

> This document is **fully self-contained**. You do not need to search the internet, open another textbook, or guess anything. Every symbol is defined, every formula is derived step by step, every example shows the full arithmetic, and every concept has a picture. Read it top to bottom.

---

# Part 0: PREREQUISITES — read this first, nothing is skipped

Statistics is how we *learn from data*: taking a pile of numbers and answering "what's typical?", "how much do they vary?", "is this difference real or luck?". Before the statistics itself, you need four small building blocks. They are reviewed below **in full**.

---

## 0.1 Data, observations, and variables (the raw material)

- **Data:** a collection of numbers (or categories) measured from the world.
- **Observation / sample point:** one row of data. Example: one customer's purchase.
- **Variable / feature:** one column. Example: purchase amount.
- **Dataset:** the whole table.

```
            ┌─────────────────────────────────────────────┐
            │  Customer │ Age │  Spend ($) │ City        │
            ├─────────────────────────────────────────────┤
 observation│   Alice   │ 34  │    120.50  │ Austin  ... │
  (row)     │   Bob     │ 22  │     45.00  │ Boston     │
            │   ...     │ ... │      ...   │ ...        │
            └─────────────────────────────────────────────┘
              ▲ column = variable / feature (one per column)
```

In ML, each row is one **training example** $x_i$ and each column is a **feature**.

**Two kinds of variables (you must know which you're holding):**
- **Numerical** (amounts, ages, prices) → can do arithmetic: mean, variance.
- **Categorical** (city, color, class) → can only count: frequencies, proportions.

---

## 0.2 Populations vs. samples — the whole truth vs. what we saw

- **Population:** *every* individual you care about. Example: all 8 billion people on Earth.
- **Sample:** the subset you actually measured. Example: 1,000 surveyed people.

**Symbols differ (this matters):**

| | Population (all) | Sample (measured) |
| :--- | :--- | :--- |
| Mean | $\mu$ | $\bar{x}$ |
| Variance | $\sigma^2$ | $s^2$ |
| Std deviation | $\sigma$ | $s$ |
| Size | $N$ | $n$ |

**The key insight of all statistics:** we can't measure the population, so we measure a sample and *estimate* the population values. Everything in Modules 3–5 is about making those estimates trustworthy.

---

## 0.3 Sorting numbers and the mean (the arithmetic foundation)

**The mean** of numbers $x_1, x_2, \dots, x_n$ is their sum divided by how many there are:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Worked example:** spend data $= 45, 60, 75, 90, 130$. Mean $= \frac{45+60+75+90+130}{5} = \frac{400}{5} = 80$.

**The median** = the middle value *after sorting*. Sorted: $45, 60, 75, 90, 130$ → median = 75. (If even count, average the two middle values.)

**Why both?** The mean is dragged around by extreme values; the median resists them. A single $10{,}000$ purchase in the data above would yank the mean to $\frac{400 + 9900}{6} \approx 1717$ while the median stays calm near 77.5.

**Squaring and square roots** (needed for variance): $5^2 = 25$, $\sqrt{25} = 5$. Variance is built from squared deviations, then the square root brings the units back.

---

## 0.4 Percentages and proportions

- **Proportion:** a fraction between 0 and 1. Example: 12 of 60 customers churned → $\hat{p} = \frac{12}{60} = 0.2$.
- **Percentage:** proportion × 100 = 20%.
- Proportions are *estimates* too: the sample churn proportion $\hat{p} = 0.2$ estimates the population churn rate (Module 4 covers how precisely).

---

## 0.5 Notation table — every symbol used in this document

**Essential now (Modules 1–2):**

| Symbol | Name | Meaning |
| :--- | :--- | :--- |
| $x_i$ | observation i | the i-th data point |
| $n$ / $N$ | sample / population size | how many items measured / exist |
| $\bar{x}$ | sample mean | average of the sample |
| $\mu$ | population mean | average of everything (unknown) |
| $s^2$ / $\sigma^2$ | sample / population variance | spread around the mean |
| $s$ / $\sigma$ | sample / population std dev | $\sqrt{\text{variance}}$ |
| $Q_1, Q_2, Q_3$ | quartiles | values at 25%, 50%, 75% |
| $IQR$ | interquartile range | $Q_3 - Q_1$ (robust spread) |
| $z$ | z-score | standardized value (units: std devs) |
| $r$ | Pearson correlation | strength of linear relationship, $[-1, 1]$ |

**Reference later (Modules 3–5):**

| Symbol | Name | Meaning | First appears |
| :--- | :--- | :--- | :--- |
| $\hat{p}$ | sample proportion | estimated success rate | Module 2.3 |
| $SE$ | standard error | std dev of a statistic (e.g. of the mean) | Module 3.2 |
| $CI$ | confidence interval | range that plausibly holds $\mu$ | Module 4.1 |
| $H_0$, $H_1$ | null / alternative hypothesis | claim vs. its rival | Module 4.2 |
| $\alpha$ | significance level | risk of false positive (usually 0.05) | Module 4.2 |
| $p$ | p-value | probability of data this extreme if $H_0$ true | Module 4.2 |
| $\mathcal{L}(\theta)$ | likelihood | probability of data given parameter $\theta$ | Module 5.1 |
| $\hat{\theta}_{MLE}$ | MLE estimate | parameter making data most likely | Module 5.1 |
| $D_{KL}$ | KL divergence | how far two distributions are apart (see Probability doc) | Module 5.2 |

---

# Part 1: The Roadmap — where this document is going

```
                              STATISTICS FOR ML
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
[MODULE 1]                      [MODULE 2]                      [MODULE 3]
Describing Data                 Transforming Data               Sampling & Distributions
  ├── Central tendency            ├── Standardization (z-scores)   ├── Law of Large Numbers
  │    (mean, median, mode)       ├── Normalization (min-max)      ├── Sampling distributions
  ├── Spread (variance, std)      ├── Standardization vs           ├── Central Limit Theorem
  ├── Percentiles & boxplots         normalization                 └── Standard Error
  └── Skewness                    └── Correlation (Pearson r)          (what n buys you)
                                      │
        ┌─────────────────────────────┘
        ▼
[MODULE 4]                      [MODULE 5]
Inference from Samples          Estimation & Fitting Models
  ├── Confidence Intervals         ├── Maximum Likelihood (MLE)
  ├── Hypothesis Testing           ├── MLE ⇄ Cross-Entropy ⇄ MSE
  ├── p-values and significance   ├── MAP: adding priors (Bayesian)
  └── Type I & Type II errors      └── Bootstrap: resampling magic
```

**How to use this roadmap:** Module 1 is describing what you see. Module 2 is rescaling and relating data (crucial preprocessing for every ML pipeline). Module 3 explains how samples behave — the gateway to trusting models. Module 4 is formal decision-making from data. Module 5 is the crown jewel: it shows that the *loss functions deep learning minimizes are exactly Maximum Likelihood Estimation* — statistics and ML are the same subject.

---

# Part 2: COMPREHENSIVE EXPLANATION

---

# MODULE 1: DESCRIBING DATA — central tendency and spread

---

## 1.1 Mean, Median, Mode — the three "centers" of data

### What is it?

Three different answers to "where is the middle of this data?":

- **Mean** $\bar{x}$: the arithmetic average — the balance point.
- **Median**: the middle value after sorting — the 50% mark.
- **Mode**: the most frequent value — the most common occurrence.

> **TL;DR:** Mean = average (sensitive to outliers). Median = middle after sorting (robust). Mode = most frequent. Right-skewed → mean > median. Left-skewed → mean < median.

### Worked examples (full arithmetic)

**Dataset A (symmetric):** spends $= 60, 70, 80, 90, 100$.
- Mean $= \frac{400}{5} = 80$. Median (sorted, middle) $= 80$. Mode: all values occur once — no mode.
- All three centers agree at 80. **Symmetric data → mean = median.**

**Dataset B (right-skewed, one big spender):** spends $= 10, 20, 30, 40, 500$.
- Mean $= \frac{600}{5} = 120$. Median $= 30$. **The mean got dragged to 120 by the single 500; the median stayed at the honest center 30.**
- Right-skewed data (long tail to the right) → **mean > median**.

**Dataset C (left-skewed):** $= 500, 400, 30, 20, 10$ → mean $= 120$ again? No — mean $= \frac{960}{5} = 192$, median $= 30$. The rule: **the mean is pulled toward the tail, whichever side it's on.**

**Mode example:** ages $= 25, 30, 30, 30, 35, 40$ → mode = 30 (appears 3 times).

![Histograms: symmetric, right-skewed, left-skewed with mean vs median](/maths-images/stat-mean-median-mode.png)

### Where, why, how in ML

- **Where:** every summary statistic you print while debugging a model; baseline predictions (predicting the mean minimizes MSE — see Module 5).
- **Why:** the mean and median answer different questions: "what's the average?" vs. "what's typical?" For salaries or house prices (heavy tails), the median is the honest answer and the mean is misleading.
- **How:** when normalizing features (Module 2) you subtract the *mean*; if data has extreme outliers, you often use the *median* instead (robust scaling).

### How mean, median, and mode differ (decision table)

| | Mean | Median | Mode |
| :--- | :--- | :--- | :--- |
| Definition | sum ÷ count | middle after sorting | most frequent |
| Sensitive to outliers? | YES | no | no |
| Uses all values? | yes | no (only order) | no |
| Best for | symmetric data | skewed data | categorical data |
| Skewed data behavior | pulled toward tail | stays central | stays at the peak |

---

## 1.2 Variance and Standard Deviation — how far from the center?

### What is it?

Two measures of *spread*: how spread out the data is around the mean.

**Sample variance:**
$$s^2 = \frac{1}{n - 1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Sample standard deviation:** $s = \sqrt{s^2}$ — same spread, but in the *original units* (dollars, not dollars-squared).

**The mystery of $n - 1$ (WHY, not just what):** we use the sample mean $\bar{x}$ (not the true $\mu$) to measure deviations, and $\bar{x}$ is itself computed *from the data* — it's always slightly closer to the data than the true mean is. This makes raw squared deviations *systematically too small*. Dividing by $n - 1$ (instead of $n$) compensates for that bias. The result: $s^2$ is an **unbiased estimate** of $\sigma^2$. This correction is called **Bessel's correction**. (For small $n$ it matters a lot: dividing by 5 vs 4 is a 25% difference; for $n = 10{,}000$ it's a rounding error.)

> **TL;DR:** Variance = average squared distance from mean. $n-1$ (not $n$) = Bessel's correction for unbiased estimate. Std dev = $\sqrt{\text{variance}}$ (back to original units).

### Worked example (full arithmetic)

Spends: $60, 70, 80, 90, 100$. Mean $= 80$.

**Step 1 — deviations from the mean:** $-20, -10, 0, 10, 20$.

**Step 2 — square them:** $400, 100, 0, 100, 400$.

**Step 3 — sum:** $400 + 100 + 0 + 100 + 400 = 1000$.

**Step 4 — divide by $n - 1 = 4$:** $s^2 = \frac{1000}{4} = 250$.

**Step 5 — square root:** $s = \sqrt{250} \approx 15.8$.

**Meaning:** "typical" spends sit about $15.8 away from the mean of 80.

![Variance visualized: deviations from the mean, squaring, averaging](/maths-images/stat-variance.png)

### The population vs. sample formulas (exact comparison)

| | Population variance | Sample variance |
| :--- | :--- | :--- |
| Formula | $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$ | $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$ |
| Divides by | $N$ (true count) | $n - 1$ (Bessel's correction) |
| Uses | true mean $\mu$ | sample mean $\bar{x}$ |
| Our example | $1000 / 5 = 200$, $\sigma = 14.1$ | $1000 / 4 = 250$, $s = 15.8$ |

*Note: numpy's `np.var(x)` uses N (population, ddof=0) by default; `np.var(x, ddof=1)` gives the unbiased sample version. pandas' `df.var()` uses $n-1$ by default. Know which you're getting!*

### Where, why, how in ML

- **Where:** every standard deviation-based normalization, anomaly detection (data beyond $3s$), evaluating model output stability, weight initialization.
- **Why:** spread tells you how reliable the center is — the same mean with 10× the variance is a very different situation.
- **How:** anomaly detection: if customer spend has $\bar{x} = 80$, $s = 16$, a new transaction of $400 = 20\sigma$ from the mean is screaming "outlier — investigate fraud."

### How variance differs from standard deviation

| | Variance $s^2$ | Std deviation $s$ |
| :--- | :--- | :--- |
| Units | squared (dollars²) | original (dollars) |
| Used for | math (it's additive across sums) | interpretation and z-scores |
| Relation | — | $s = \sqrt{s^2}$ |

---

## 1.3 Quartiles, IQR, and Boxplots — robust summaries with pictures

### What is it?

- **Quartiles** split sorted data into four equal parts: $Q_1$ (25%), $Q_2$ (median, 50%), $Q_3$ (75%).
- **Interquartile range:** $IQR = Q_3 - Q_1$ — the spread of the *middle half*, immune to outliers.
- **Boxplot:** the standard picture — a box from $Q_1$ to $Q_3$ with the median line inside, whiskers to the non-outlier extremes, and outlier dots beyond.

**Outlier rule (Tukey's):** any point below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$ is a flagged outlier.

### Worked example (full arithmetic)

Sorted spends: $45, 50, 60, 70, 75, 90, 130, 150, 900$.

**Step 1 — median ($Q_2$):** 9 values, middle = 5th = 75.

**Step 2 — $Q_1$:** median of the lower half $45, 50, 60, 70$ = $\frac{50+60}{2} = 55$.

**Step 3 — $Q_3$:** median of upper half $90, 130, 150, 900$ = $\frac{130+150}{2} = 140$.

**Step 4 — IQR:** $140 - 55 = 85$.

**Step 5 — outlier fences:** upper fence $= 140 + 1.5 \times 85 = 140 + 127.5 = 267.5$. The value 900 is *far* above the fence → **outlier**.

![Boxplot annotated: quartiles, IQR, whiskers, outlier](/maths-images/stat-boxplot.png)

### Where, why, how in ML

- **Where:** exploratory data analysis (EDA) — the first thing a data scientist does; outlier detection before training.
- **Why:** boxplots reveal skew, outliers, and spread in one glance across many features — a 50-feature dataset becomes one picture per feature.
- **How:** flag and inspect outliers before fitting models; consider winsorizing (clipping) extreme values.

### How IQR differs from standard deviation

| | Std deviation $s$ | IQR |
| :--- | :--- | :--- |
| Sensitive to outliers? | yes | no |
| Uses | all data (mean-centered) | middle 50% only |
| Best for | symmetric, no outliers | skewed data, outliers present |

---

## 1.4 Histograms and density — seeing the whole shape

### What is it?

- **Histogram:** bins of values on the x-axis, *counts* on the y-axis — a bar chart of where the data lives.
- **Density estimate:** the histogram smoothed into a curve (like a PDF — see Probability doc Module 2.2).

![Histogram with density curve overlaid](/maths-images/stat-histogram.png)

### Worked example

Spends sorted into bins of $20$: [0–20): 2, [20–40): 5, [40–60): 8, [60–80): 12, [80–100): 7, [100–120): 4, [120–140): 2. Total: 2+5+8+12+7+4+2 = 40 ✓. The histogram shows a peak near 60–80 and a long right tail → right-skewed.

### Where, why, how in ML

- **Where:** EDA before any modeling; understanding target distributions (e.g. log-transform skewed targets before regression).
- **Why:** the shape of the data dictates the model — skewed targets want log transforms, multi-modal data wants mixture models.
- **How:** plot every feature's histogram early; spot skew, clusters, and anomalies before a single model is trained.

---

# MODULE 2: TRANSFORMING AND RELATING DATA

---

## 2.1 Standardization (z-scores) — putting features on the same scale

### What is it?

**Standardization** converts each value to how many standard deviations it sits from the mean:

$$z_i = \frac{x_i - \bar{x}}{s}$$

After standardization, the feature has **mean 0 and std deviation 1** — units gone, scale-free.

### Worked example (full arithmetic)

Spends: $60, 70, 80, 90, 100$. We computed $\bar{x} = 80$, $s = 15.81$ (Module 1.2).

| $x_i$ | $x_i - \bar{x}$ | $z_i = (x_i - 80)/15.81$ |
| :--- | :--- | :--- |
| 60 | $-20$ | $-1.26$ |
| 70 | $-10$ | $-0.63$ |
| 80 | $0$ | $0.00$ |
| 90 | $10$ | $+0.63$ |
| 100 | $20$ | $+1.26$ |

**Check:** mean of z-scores $= \frac{-1.26 - 0.63 + 0 + 0.63 + 1.26}{5} = 0$ ✓; std dev = 1 ✓.

![Standardization: raw data becomes mean-0, std-1](/maths-images/stat-standardization.png)

**Interpretation:** a z-score of $+2$ means "2 standard deviations above average" — about the 97.7th percentile for normal data.

### Where, why, how in ML

- **Where:** preprocessing for gradient-descent models (neural nets, SVM, logistic regression, PCA, k-means).
- **Why:** features with huge scales (salary: millions) dominate features with tiny scales (age: ~30) in distance-based and gradient-based learning. Standardization makes all features contribute fairly and makes gradients well-conditioned (see Calculus doc on gradient descent).
- **How:** compute $(\bar{x}, s)$ on the *training* data only, then apply to train AND test with those same values (never re-fit on test — that's leakage).

---

## 2.2 Normalization (min–max) — squeezing into [0, 1]

### What is it?

**Min–max normalization** rescales every value into the range $[0, 1]$:

$$x'_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}$$

### Worked example (full arithmetic)

Spends: $60, 70, 80, 90, 100$. Min $= 60$, max $= 100$, range $= 40$.

| $x_i$ | $x'_i = (x_i - 60)/40$ |
| :--- | :--- |
| 60 | 0.00 |
| 70 | 0.25 |
| 80 | 0.50 |
| 90 | 0.75 |
| 100 | 1.00 |

Smallest → 0, largest → 1, everything else proportional.

### How standardization differs from normalization (the definitive comparison)

| | Standardization (z-score) | Normalization (min–max) |
| :--- | :--- | :--- |
| Formula | $\frac{x - \bar{x}}{s}$ | $\frac{x - \min}{\max - \min}$ |
| Output range | unbounded (usually $-3$ to $+3$) | exactly $[0, 1]$ |
| Uses | mean and std dev | min and max |
| Sensitive to outliers? | less (std dev absorbs them) | yes (min/max ARE outliers) |
| Preserves | relative distances | relative distances |
| Best for | gradient methods, Gaussian-ish data | bounded data, pixel values [0,255] |

*Both are linear rescalings (no shape change) — the only difference is which two anchor points you use.*

### Where, why, how in ML

- **Where:** image processing (pixels 0–255 → 0–1), neural nets, any bounded input.
- **Why:** some models need bounded inputs (e.g. certain activation functions); others just need comparable scales.
- **How:** for tree models (random forest, XGBoost) neither transform matters — trees only care about *order*, not scale.

---

## 2.3 Correlation — measuring how two variables move together

### What is it?

**Pearson correlation** $r$ measures the strength and direction of the *linear* relationship between two variables:

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}$$

**Properties (memorize):**
- $r \in [-1, 1]$: $+1$ perfect positive line, $-1$ perfect negative line, $0$ no linear relationship.
- **Scale-free:** unitless — $r$ between height-in-cm and weight-in-kg equals $r$ between height-in-inches and weight-in-pounds.

### Worked example (full arithmetic)

Study hours vs. test scores for 5 students:

| student | $x$ (hours) | $y$ (score) |
| :--- | :--- | :--- |
| 1 | 1 | 50 |
| 2 | 2 | 60 |
| 3 | 3 | 65 |
| 4 | 4 | 75 |
| 5 | 5 | 80 |

**Step 1 — means:** $\bar{x} = 3$, $\bar{y} = 66$.

**Step 2 — deviations:** $x$-deviations: $-2, -1, 0, 1, 2$. $y$-deviations: $-16, -6, -1, 9, 14$.

**Step 3 — products:** $32, 6, 0, 9, 28$; sum $= 75$.

**Step 4 — squared deviations:** $x$: $4, 1, 0, 1, 4$ → sum 10. $y$: $256, 36, 1, 81, 196$ → sum 570.

**Step 5 — correlation:**
$$r = \frac{75}{\sqrt{10 \times 570}} = \frac{75}{\sqrt{5700}} = \frac{75}{75.5} \approx 0.99$$

Nearly perfect positive correlation — more study, higher score, almost linearly.

![Scatter plots: positive, negative, zero, and curved correlations](/maths-images/stat-correlation-scatter.png)

### The critical warning (read twice)

**Correlation ≠ causation.** Ice cream sales and drowning deaths are strongly correlated — both rise in summer. Neither causes the other; heat drives both. A third factor (the "confounder") explains the link.

**Also: $r = 0$ doesn't mean "no relationship."** A perfect U-shape (parabola) has $r \approx 0$ yet a strong *non-linear* relationship (see Probability doc Module 3.1 for the same warning).

### Where, why, how in ML

- **Where:** feature selection (drop highly correlated redundant features), EDA, multicollinearity checks in regression.
- **Why:** two features with $|r| \approx 1$ carry the same information — keeping both wastes capacity and destabilizes linear models (see Linear Algebra doc, normal equations and rank).
- **How:** build a correlation matrix of all features; if $|r| > 0.9$ between two features, keep one.

### How correlation differs from covariance

| | Covariance $\text{Cov}(X, Y)$ | Correlation $r$ |
| :--- | :--- | :--- |
| Scale | depends on units (huge for big units) | unitless $[-1, 1]$ |
| Tells you | direction of co-movement | direction AND strength |
| Relation | — | $r = \frac{\text{Cov}(X,Y)}{s_X s_Y}$ |

---

# MODULE 3: SAMPLING AND SAMPLING DISTRIBUTIONS — why samples behave

---

## 3.1 The Law of Large Numbers — averages converge

### What is it?

As the sample size grows, the sample mean converges to the population mean:

$$\bar{x}_n \to \mu \quad \text{as } n \to \infty$$

**Plain words:** with few observations, chance rules; with many, the truth emerges. (Full treatment in Probability doc, Module 5.4.)

### Worked example

Roll a fair die. Roll 1: mean 4 (wrong, true mean 3.5). Roll 10: maybe 3.3. Roll 1,000: ~3.51. Roll 100,000: 3.499... — pinned to 3.5.

### Where, why, how in ML

- **Why:** it's the entire justification for *empirical risk minimization* — training loss averaged over the dataset approximates the true expected loss as the dataset grows.
- **How:** big data works *because* of this: more training samples → better estimates of the true patterns.

---

## 3.2 Sampling Distributions and the Standard Error — how trustworthy is the sample mean?

### What is it?

Imagine this experiment: take many different samples of size $n$ from the same population, and compute the mean of each. The *collection of those means* has its own distribution — the **sampling distribution of the mean**.

**Key facts (memorize):**
- The mean of all sample means = the population mean $\mu$ (sample means are correct *on average* — unbiased).
- The spread of the sample means is **much smaller** than the population spread — individual values scatter a lot; *averages* scatter much less.

**The standard error (SE)** — the standard deviation of the sample mean:

$$SE = \frac{\sigma}{\sqrt{n}}$$

*Larger samples → tighter sampling distribution → more trustworthy mean. This is the quantitative payoff of collecting more data.*

### Worked example (full arithmetic)

Population of spends with $\mu = 100$, $\sigma = 20$.

- Sample of $n = 25$: $SE = \frac{20}{\sqrt{25}} = \frac{20}{5} = 4$. Sample means typically land within ±4 of 100.
- Sample of $n = 100$: $SE = \frac{20}{\sqrt{100}} = \frac{20}{10} = 2$. Twice as tight — **4× the data → half the error**.

![Standard error shrinking with sqrt(n)](/maths-images/stat-se-vs-n.png)

### Where, why, how in ML

- **Where:** every reported metric (accuracy, loss) comes with uncertainty; A/B testing; comparing model A vs model B.
- **Why:** the SE tells you whether a difference between two models' accuracies is *signal or noise*: a 0.5% accuracy gap with SE 0.2% is meaningful; the same gap with SE 2% is nothing.
- **How:** report metric ± 2×SE (roughly a 95% interval — next module formalizes this).

---

## 3.3 The Central Limit Theorem — the magic of bell curves

### What is it?

**The Central Limit Theorem (CLT):** no matter what shape the population distribution has (skewed, uniform, weird), the *sampling distribution of the mean* becomes approximately **Gaussian** (bell-shaped) as the sample size grows. For $n \ge 30$, the approximation is usually excellent.

$$ \bar{x} \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{for large } n $$

**Why it's the most important theorem in statistics:** it's why the Gaussian distribution appears everywhere — it's not that the world is normal, it's that *averages* are normal. This one fact powers confidence intervals (Module 4), hypothesis tests, and the entire Gaussian toolkit in ML.

![CLT: skewed population → sample means become Gaussian](/maths-images/stat-clt.png)

> **TL;DR:** CLT = sample means become Gaussian no matter the population shape (n ≥ 30). This is why Gaussian tools work everywhere — averages are normal even when data isn't.

### Real Worked Example (with actual numbers)

**Population:** a fair die (values 1,2,3,4,5,6). Population mean $\mu = 3.5$, std $\sigma \approx 1.71$. This is **uniform** (not Gaussian!).

Take **4 samples of size $n=5$** each, compute their means:

| Sample | 5 rolls | Mean |
| :--- | :--- | :--- |
| 1 | 3, 6, 2, 5, 1 | 3.4 |
| 2 | 4, 4, 3, 2, 6 | 3.8 |
| 3 | 1, 5, 2, 3, 4 | 3.0 |
| 4 | 6, 1, 5, 2, 3 | 3.4 |

**Mean of means** = $(3.4 + 3.8 + 3.0 + 3.4) / 4 = 3.4$ ≈ population mean 3.5 ✓
**Spread of means (SE)** ≈ 0.3 (tiny compared to population spread 1.71)

Now take 100 samples of $n=30$ → the 100 means form a **bell-shaped histogram** centered at 3.5, spread = $1.71/\sqrt{30} \approx 0.31$.

![CLT: skewed population → sample means become Gaussian](/maths-images/stat-clt.png)

### Where, why, how in ML

- **Where:** justifying Gaussian assumptions, confidence intervals, ensemble models (bagging averages → Gaussian-ish).
- **Why:** anytime a model's output is an average of many terms (mini-batch losses, bagged trees), CLT explains why the aggregate behaves normally.
- **How:** your mini-batch loss, averaged over a few hundred samples, is already approximately Gaussian — which justifies using mean ± 2×SE as a monitoring band in training curves.

---

# MODULE 4: INFERENCE — decisions from data

---

## 4.1 Confidence Intervals — a range that plausibly contains the truth

### What is it?

A **95% confidence interval** for the population mean is a range constructed from the sample:

$$\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}$$

**The correct interpretation (95% of people get this wrong):** if we repeated the whole experiment many times and computed this interval each time, *about 95% of those intervals would contain the true $\mu$*. It is NOT "the probability that $\mu$ is in this interval is 95%" — $\mu$ is a fixed number, not random; the interval is what's random.

**Where does 1.96 come from?** By CLT, $\bar{x}$ is Gaussian around $\mu$ with std $SE$. For a standard Gaussian, 95% of the probability mass lies within $\pm 1.96$ standard deviations (the 68–95–99.7 rule from the Probability doc). So $\bar{x} \pm 1.96 \times SE$ captures $\mu$ 95% of the time.

### Worked example (full arithmetic)

Measure 100 customers' spends: $\bar{x} = 80$, $s = 20$.

**Step 1 — SE:** $SE = \frac{20}{\sqrt{100}} = 2$.

**Step 2 — margin of error:** $1.96 \times 2 = 3.92$.

**Step 3 — interval:** $[80 - 3.92, \ 80 + 3.92] = [76.08, 83.92]$.

**Interpretation:** the plausible range for the true mean spend is \$76.08 to \$83.92. With only $n = 25$: $SE = 4$, margin $= 7.84$, interval $[72.16, 87.84]$ — *wider*, correctly reflecting the less trustworthy estimate.

![95% confidence intervals from repeated samples](/maths-images/stat-confidence-interval.png)

### Where, why, how in ML

- **Where:** reporting model performance, A/B testing decisions, Bayesian-ish uncertainty estimates in production.
- **Why:** a point estimate (accuracy 91.3%) with no uncertainty is misleading; the interval tells you whether to trust it.
- **How:** compare two models by their intervals: if they don't overlap, the difference is real; if they overlap heavily, more data is needed before choosing.

---

## 4.2 Hypothesis Testing and p-values — is this difference real or luck?

### What is it?

**Hypothesis testing** formalizes the question "is this effect real?":

- **Null hypothesis $H_0$:** nothing is happening. Example: new model accuracy = old model accuracy.
- **Alternative hypothesis $H_1$:** something is happening. Example: new model is better.

**The p-value:** the probability of seeing data *at least this extreme* **assuming $H_0$ is true**. Small p-value → your data is surprising *if nothing is happening* → evidence against $H_0$.

**The decision rule (significance level $\alpha$, usually 0.05):**
- $p < \alpha$ → "statistically significant" → reject $H_0$ (accept the effect).
- $p \ge \alpha$ → fail to reject $H_0$ (not enough evidence).

> **TL;DR:** p-value = probability of seeing data this extreme IF $H_0$ is true. $p < 0.05$ → reject $H_0$ (effect is real). $p \ge 0.05$ → not enough evidence. Never "prove $H_0$" — absence of evidence ≠ evidence of absence.

### Worked example (full arithmetic)

A coin is claimed fair ($H_0$: $p = 0.5$). You flip it 20 times and get 16 heads. Is that suspicious?

**Step 1 — what's "at least this extreme"?** 16, 17, 18, 19, or 20 heads (both tails: 4 or fewer too, for a two-sided test).

**Step 2 — compute probabilities under $H_0$ (Binomial, Probability doc Module 4.2):**
$$P(X = 16) = \binom{20}{16}(0.5)^{16}(0.5)^4 = 4845 \times 0.00001526 \times 0.0625 = 0.0046$$
$$P(X = 17) = \binom{20}{17}(0.5)^{20} = 1140 \times 0.000000954 = 0.0011$$
$$P(X = 18) = 190 \times 0.000000954 = 0.00018$$
$$P(X = 19) = 20 \times 0.000000954 = 0.000019$$
$$P(X = 20) = 1 \times 0.000000954 = 0.000001$$

**Step 3 — sum the extreme tail (two-sided):**
$$p \approx 2 \times (0.0046 + 0.0011 + 0.00018 + 0.000019 + 0.000001) = 2 \times 0.0059 \approx 0.0118$$

> **Why Multiply by 2? (Two-sided test)**
> - Observed: 16 heads (extreme HIGH). 
> - Two-sided = "16+ heads OR 4- heads (equally extreme LOW)".
> - $P(16+) = 0.0059$, $P(4-) = 0.0059$ (symmetric under $H_0$).
> - Total = $2 \times 0.0059 = 0.0118$.

**Step 4 — decide:** $p = 0.0118 < 0.05$ → **reject $H_0$**. 16 heads in 20 flips is real evidence the coin is biased — such a result would happen by chance only ~1.2% of the time with a fair coin.

![p-value: tail area beyond the observed statistic](/maths-images/stat-pvalue.png)

### Where, why, how in ML

- **Where:** A/B testing (does the new UI lift conversion?), model comparison (is the accuracy gap real?), feature significance in regression.
- **Why:** with enough data, *every* tiny effect becomes "significant" — p-values answer "real or luck?" but NOT "how big?" (see next section's warning).
- **How:** before rolling out a model, test the hypothesis "new model ≥ old model" on a holdout set; only ship if the gap clears significance.

---

## 4.3 Type I and Type II Errors — the two ways to be wrong

### What is it?

When testing, there are exactly two ways to err:

- **Type I error (false positive):** reject $H_0$ when it's actually true — crying wolf. Probability = $\alpha$ (you set this: 0.05).
- **Type II error (false negative):** fail to reject $H_0$ when it's actually false — missing the real effect. Probability = $\beta$; **power** $= 1 - \beta$ is the chance of catching a real effect.

| Decision ↓ / Truth → | $H_0$ true | $H_0$ false |
| :--- | :--- | :--- |
| Reject $H_0$ | **Type I error** (α) | Correct! |
| Fail to reject $H_0$ | Correct! | **Type II error** (β) |

### Worked example (full arithmetic)

Testing whether a new spam filter is better.
- **Type I:** declaring the filter better when it's actually identical. Cost: wasted rollout.
- **Type II:** declaring "no improvement" when the filter actually IS better. Cost: missed business benefit.

Bigger samples shrink **both** errors (SE shrinks → distributions separate more clearly):

![Type I and Type II errors under two overlapping distributions](/maths-images/stat-type1-type2.png)

**The trade-off:** lowering $\alpha$ (0.05 → 0.01) makes Type I errors rarer but Type II errors more likely — stricter evidence requirements mean you miss more real effects.

### Where, why, how in ML

- **Where:** every "significant improvement" claim in ML papers; fraud detection thresholds; medical AI validation.
- **Why:** the error you care about depends on the stakes: false fraud alarms annoy customers (Type I); missed fraud costs money (Type II). Setting the threshold picks your trade-off.
- **How:** in fraud detection, tune the classification threshold to balance false positives (Type I) and false negatives (Type II) — the ROC curve exists to visualize exactly this trade-off.

### How Type I differs from Type II (one-line memory)

*Type I = believing a lie (false positive). Type II = missing the truth (false negative).*

---

# MODULE 5: ESTIMATION — fitting models to data

---

## 5.1 Maximum Likelihood Estimation (MLE) — pick the parameters that make the data most likely

### What is it?

Given data and a chosen distribution family (with unknown parameters $\theta$), **MLE** picks the $\theta$ that maximizes the **likelihood** — the probability of observing exactly this data:

$$\hat{\theta}_{MLE} = \arg\max_\theta \mathcal{L}(\theta) = \arg\max_\theta \prod_{i=1}^{n} P(x_i \mid \theta)$$

**Why a product?** If the data points are independent, the probability of *all of them* = product of their individual probabilities (Probability doc, Module 3.1 — AND means multiply).

**Why log?** Products of many tiny probabilities underflow to 0 in computers ($0.9^{10{,}000}$ is astronomically small). The log turns products into sums (log of product = sum of logs) and — since log is increasing — maximizing the log maximizes the likelihood:

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i \mid \theta)$$

### Worked example — the coin (full arithmetic, every step)

Flip a coin 10 times: 7 heads, 3 tails. Model: $X \sim \text{Bernoulli}(p)$. Likelihood:

$$\mathcal{L}(p) = p^7 (1 - p)^3$$

**Step 1 — try $p = 0.5$:** $\mathcal{L}(0.5) = (0.5)^7 (0.5)^3 = 0.5^{10} = 0.000977$.

**Step 2 — try $p = 0.7$:** $\mathcal{L}(0.7) = (0.7)^7 (0.3)^3 = 0.08235 \times 0.027 = 0.002224$ — bigger ✓.

**Step 3 — try $p = 0.8$:** $\mathcal{L}(0.8) = (0.8)^7 (0.2)^3 = 0.2097 \times 0.008 = 0.001678$ — smaller ✗.

**Step 4 — the winner:** between 0.7 and 0.8, the maximum lands at $\hat{p} = \frac{7}{10} = 0.7$. **The MLE of a proportion is just the observed proportion** — $7$ successes out of $10$ flips. This is the "obvious" answer, and MLE is why it's provably the best one.

![Likelihood curve: peak at p = 0.7](/maths-images/stat-mle-bernoulli.png)

**Proof via calculus:** $\frac{d}{dp}[\log \mathcal{L}] = \frac{d}{dp}[7\ln p + 3\ln(1-p)] = \frac{7}{p} - \frac{3}{1-p} = 0$ → $7(1-p) = 3p$ → $7 = 10p$ → $p = 0.7$ ✓.

### Where, why, how in ML — THE connection to loss functions

**MLE is not a side topic — it IS deep learning's loss functions:**

- **Classification (cross-entropy):** the model outputs $p_\theta(y_i \mid x_i)$ for true class $y_i$. MLE maximizes $\prod p_\theta(y_i \mid x_i)$. The **negative log-likelihood**:

$$-\sum_i \log p_\theta(y_i \mid x_i)$$

...is exactly **cross-entropy loss** (Probability doc, Module 5.2). Minimizing cross-entropy = maximizing likelihood = MLE. Same thing, two names.

- **Regression (MSE):** assume residuals are Gaussian: $y_i \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$. The log-likelihood is:

$$\log \mathcal{L} = \sum_i \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} \right]$$

> **Constants don't matter for optimization:** the term $-\frac{1}{2}\log(2\pi\sigma^2)$ is constant w.r.t. the model parameters $\theta$. **Constants don't affect the argmax**. This is why maximizing the log-likelihood = minimizing $\sum_i (y_i - f_\theta(x_i))^2$ = MSE. The Gaussian assumption *derives* MSE. (If you instead assume Laplace-distributed residuals, you get MAE / L1 loss — the choice of loss is a choice of noise model!)

**Summary table:**

| Task | Noise model | Loss MLE produces |
| :--- | :--- | :--- |
| Binary classification | Bernoulli | binary cross-entropy |
| Multi-class classification | Categorical | cross-entropy |
| Regression | Gaussian | MSE |
| Regression (robust) | Laplace | MAE |

### How MLE differs from just "fitting a curve"

- **Fitting** = tweaking parameters until predictions look good (vague).
- **MLE** = the parameters that *maximize the probability of the observed data* — a precise, provably optimal criterion. Every gradient-descent step in deep learning is doing MLE under the hood.

---

## 5.2 MAP — Maximum A Posteriori (adding priors)

### What is it?

**MAP** adds a **prior** $P(\theta)$ — your belief about the parameter *before* seeing data — and maximizes the **posterior** (Bayes' theorem, Probability doc Module 3.3):

$$\hat{\theta}_{MAP} = \arg\max_\theta \underbrace{P(\theta \mid \text{data})}_{\text{posterior}} = \arg\max_\theta \underbrace{\mathcal{L}(\theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}}$$

(denominator $P(\text{data})$ is constant in $\theta$ — drop it)

**In log form:** $\arg\max_\theta \left[ \sum_i \log P(x_i \mid \theta) + \log P(\theta) \right]$ — *likelihood + prior*. MLE is MAP with a **flat (uninformative) prior**.

### Worked example — the coin with a prior (full arithmetic)

Same data: 7 heads in 10 flips. Before seeing data, you believe coins are usually fair-ish: prior Beta(2, 2) peaked at 0.5 (Probability doc Module 4.8). The posterior is Beta with updated parameters:

$$\text{posterior} = \text{Beta}(7 + 2, 3 + 2) = \text{Beta}(9, 5)$$

**MAP estimate (mode of the posterior):** $\frac{\alpha - 1}{\alpha + \beta - 2} = \frac{9 - 1}{9 + 5 - 2} = \frac{8}{12} \approx 0.667$.

> **Where does the Beta mode formula come from?**
> Beta($\alpha, \beta$) PDF: $f(p) \propto p^{\alpha-1}(1-p)^{\beta-1}$.
> Take log: $\log f = (\alpha-1)\log p + (\beta-1)\log(1-p) + \text{const}$.
> Derivative: $\frac{d}{dp} = \frac{\alpha-1}{p} - \frac{\beta-1}{1-p} = 0$.
> Solve: $(\alpha-1)(1-p) = (\beta-1)p$ → $\alpha - 1 = p(\alpha + \beta - 2)$ → $p = \frac{\alpha-1}{\alpha+\beta-2}$.
> For $\alpha=9, \beta=5$: $\frac{8}{12} \approx 0.667$. ✓

**Compare:**
- MLE: $\hat{p} = 0.700$ (data alone).
- MAP: $\hat{p} \approx 0.667$ (data pulled toward the prior's 0.5).

**The prior acted as extra pseudo-data:** 2 extra "heads" and 2 extra "tails" from belief. With more data (1,000 flips, 700 heads), MAP → $\frac{700 + 2}{1000 + 4} \approx 0.701$ — the prior's influence evaporates as evidence accumulates. **MAP = MLE + regularization.**

![MAP: prior, likelihood, posterior all on one plot](/maths-images/stat-map.png)

### Where, why, how in ML — THE connection to regularization

- **L2 regularization** (weight decay, ridge regression) = MAP with a **Gaussian prior** on the weights $\theta \sim \mathcal{N}(0, \sigma^2)$. The log-prior term $\propto -\|\theta\|^2$ *is* the L2 penalty.
- **L1 regularization** (lasso) = MAP with a **Laplace prior** — same logic, different shape → sparse weights.
- So "regularization" and "prior beliefs" are literally the same mathematics. Every $\lambda \|\theta\|^2$ in a loss function is a Bayesian prior wearing a costume.

### How MLE differs from MAP (decision table)

| | MLE | MAP |
| :--- | :--- | :--- |
| Maximizes | likelihood $\mathcal{L}(\theta)$ | likelihood × prior |
| Uses prior? | no | yes |
| Small-data behavior | wild (overfits) | tamed by prior |
| Large-data behavior | converges to truth | converges to truth (prior fades) |
| ML identity | plain loss | loss + regularization |

---

## 5.3 The Bootstrap — resampling to get uncertainty for free

### What is it?

The **bootstrap** estimates the sampling distribution of *any* statistic (mean, median, accuracy, even a neural net's score) **without any formulas** — just resampling:

1. Take your sample of size $n$.
2. **Resample:** draw a new sample of size $n$ **with replacement** (same data point can appear multiple times).
3. Compute the statistic on the resample. Store it.
4. Repeat B times (e.g. 10,000).
5. The B stored values form an empirical sampling distribution → use its percentiles as a confidence interval.

**Why "with replacement"?** Without replacement you'd just get the original sample back. Replacement is what makes each resample a fresh "imitation experiment."

> **Bootstrap Pseudocode (runnable Python):**
> ```python
> import numpy as np
> 
> def bootstrap_ci(data, statistic, B=10000, alpha=0.05):
>     """Return (lower, upper) bootstrap percentile CI."""
>     stats = []
>     n = len(data)
>     for _ in range(B):
>         resample = np.random.choice(data, size=n, replace=True)
>         stats.append(statistic(resample))
>     lower = np.percentile(stats, 100 * alpha / 2)
>     upper = np.percentile(stats, 100 * (1 - alpha / 2))
>     return lower, upper
> 
> # Example:
> data = np.array([60, 70, 80, 90, 100])
> ci = bootstrap_ci(data, np.mean)
> print(f"95% CI for mean: {ci}")  # e.g. (70.4, 90.6)
> ```

### Worked example (full arithmetic)

Spends: $60, 70, 80, 90, 100$; $\bar{x} = 80$.

**Resample 1:** $60, 80, 90, 80, 100$ → mean $= 82$.
**Resample 2:** $100, 60, 70, 90, 100$ → mean $= 84$.
**Resample 3:** $60, 60, 80, 90, 70$ → mean $= 72$.
... do this 10,000 times.

**Bootstrap 95% CI for the mean:** sort the 10,000 resample means; the 2.5th percentile and 97.5th percentile bound the interval (e.g. $[70.4, 90.6]$). Notice it's *wider* than the CLT formula's interval here — the bootstrap honestly reflects this tiny sample's uncertainty.

![Bootstrap: original sample → many resamples → distribution of the statistic](/maths-images/stat-bootstrap.png)

### Where, why, how in ML

- **Where:** confidence intervals on model metrics (accuracy, AUC) without normality assumptions; **bagging** (Bootstrap AGGregatING) — random forests literally train each tree on a bootstrap resample!
- **Why:** some statistics (median, accuracy) have no clean formula for uncertainty — resampling sidesteps all theory.
- **How:** train 100 models on 100 bootstrap resamples, compute metric variance → model uncertainty estimate; or just note that random forests ARE bootstrap machines.

### How bootstrap differs from the CLT approach

| | CLT formula | Bootstrap |
| :--- | :--- | :--- |
| Requires | large n, known formula | only a sample |
| Assumptions | Gaussian approximation | almost none (resamples the data) |
| Works for | means and similar | ANY statistic |
| Cost | instant | compute-heavy (B resamples) |

---

# Part 3: SUMMARY CHEAT-SHEET

| Concept | One-line definition | Formula | ML application |
| :--- | :--- | :--- | :--- |
| **Mean** | arithmetic center | $\bar{x} = \frac{1}{n}\sum x_i$ | baseline predictions (minimizes MSE) |
| **Median** | middle after sorting | sorted, middle value | robust summary for skewed data |
| **Mode** | most frequent | — | categorical data summary |
| **Variance (sample)** | spread, Bessel-corrected | $s^2 = \frac{1}{n-1}\sum(x_i-\bar{x})^2$ | normalization, feature scaling |
| **Std deviation** | spread in original units | $s = \sqrt{s^2}$ | z-scores, anomaly detection |
| **IQR** | middle-50% spread | $Q_3 - Q_1$ | boxplots, robust outliers |
| **Boxplot** | picture of quartiles + outliers | — | EDA |
| **Standardization** | mean 0, std 1 | $z = \frac{x-\bar{x}}{s}$ | gradient models need it |
| **Normalization** | scale into [0,1] | $\frac{x-\min}{\max-\min}$ | pixels, bounded inputs |
| **Correlation $r$** | linear co-movement in $[-1,1]$ | $\frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2\sum(y-\bar{y})^2}}$ | feature selection |
| **LLN** | averages converge to truth | $\bar{x}_n \to \mu$ | why big data works |
| **CLT** | sample means → Gaussian | $\bar{x} \approx \mathcal{N}(\mu, \sigma^2/n)$ | justifies all Gaussian tools |
| **Standard error** | uncertainty of the mean | $SE = \sigma/\sqrt{n}$ | model comparison |
| **Confidence interval** | plausible range for $\mu$ | $\bar{x} \pm 1.96\,SE$ | reporting metrics |
| **p-value** | surprise of data if $H_0$ true | tail probability | A/B testing |
| **Type I error** | false positive | prob $\alpha$ | threshold tuning |
| **Type II error** | false negative | prob $\beta$ | power analysis |
| **MLE** | parameters that make data most likely | $\max_\theta \sum \log P(x_i\mid\theta)$ | cross-entropy, MSE losses |
| **MAP** | MLE + prior | $\max_\theta [\log\mathcal{L} + \log P(\theta)]$ | L1/L2 regularization |
| **Bootstrap** | uncertainty by resampling | resample with replacement, B times | random forests (bagging), CIs |

---

# Part 4: WHAT TO READ NEXT (inside this same math folder)

- **probability.md** — the distributions (Bernoulli, Gaussian, Beta) that Modules 4–5 rely on, plus the Bayes' theorem and entropy used by MAP and cross-entropy.
- **calculus.md** — the derivatives that optimize MLE/MAP (gradient descent IS how we maximize likelihood in deep learning) and the integrals behind expectations.
- **linear-algebra.md** — the vectors and matrices that let a model output probabilities over thousands of classes, and PCA (variance-based dimensionality reduction).