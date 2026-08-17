---
title: Statistics — Quick Revision
description: 10-minute revision of statistics for ML — descriptive stats, CLT, confidence intervals, p-values, MLE, MAP, bootstrap in one page.
tags: [math, statistics, quick-rev, clt, p-value, mle, bootstrap]
---

# STATISTICS — QUICK REVISION

> Condensed from the full guide at `MLDL/maths/statistics.md`. For revision: scan, recall, quiz yourself.

## THE BIG PICTURE (3 lines)

1. **Descriptive statistics** = summarizing data: center (mean/median), spread (variance/IQR), shape (histogram/skew).
2. **Inference** = deciding what the *population* is like from a *sample*: confidence intervals, p-values.
3. **Estimation** = fitting models: MLE (= the loss functions of deep learning!), MAP (= regularization), bootstrap.

## NOTATION QUICK-REF

| Symbol | Meaning |
| :--- | :--- |
| $\mu$, $\bar{x}$ | population / sample mean |
| $\sigma^2$, $s^2$ | population / sample variance |
| $\sigma$, $s$ | std deviations |
| $Q_1$, $Q_3$, IQR | quartiles, interquartile range |
| $r$ | Pearson correlation |
| $SE$ | standard error of the mean |
| $H_0$, $H_1$ | null / alternative hypothesis |
| $\alpha$ | significance level (default 0.05) |
| $p$ | p-value |
| $\hat{\theta}_{MLE}$ | maximum likelihood estimate |
| $B$ | bootstrap resamples |

## DESCRIBING DATA — ALL FORMULAS

| Measure | Formula | When to use |
| :--- | :--- | :--- |
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | symmetric data |
| Median | middle of sorted data | skewed data (income, prices) |
| **Sample variance** | $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ | estimating population spread |
| Std deviation | $s = \sqrt{s^2}$ | spread in original units |
| IQR | $Q_3 - Q_1$ | robust spread, boxplots |
| Outlier (Tukey) | beyond $Q_1 - 1.5\text{IQR}$ or $Q_3 + 1.5\text{IQR}$ | anomaly flagging |

**Why $n-1$?** The sample mean is *computed from the data*, so deviations measured from it are systematically too small. Dividing by $n-1$ (Bessel's correction) makes $s^2$ an unbiased estimate of $\sigma^2$. (Note: `np.var(x)` = population version, `pd.DataFrame.var()` = sample version!)

**Skew rules:** right-skew (long tail right) → mean > median; left-skew → mean < median; symmetric → mean = median.

**Worked variance (30 sec):** $60, 70, 80, 90, 100$; mean 80; deviations $-20,-10,0,10,20$; squares sum 1000; $s^2 = 1000/4 = 250$; $s = 15.8$.

## TRANSFORMING & RELATING DATA

| | Standardization (z-score) | Normalization (min–max) |
| :--- | :--- | :--- |
| Formula | $z = \frac{x - \bar{x}}{s}$ | $x' = \frac{x - \min}{\max - \min}$ |
| Output | mean 0, std 1 (unbounded) | exactly $[0, 1]$ |
| Sensitive to outliers | less | yes (min/max) |
| Best for | gradient models, PCA, k-means | pixels, bounded inputs |

**Pearson correlation:**
$$r = \frac{\sum (x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2 \sum(y-\bar{y})^2}}$$
- $r \in [-1, 1]$; unitless; direction + strength of *linear* relationship.
- ⚠️ Correlation ≠ causation; $r = 0$ ≠ no relationship (parabola has $r \approx 0$).
- ML: drop features with $|r| > 0.9$ (redundant).

## SAMPLING — THE THREE PILLARS

| Concept | Statement | Formula |
| :--- | :--- | :--- |
| **LLN** | sample mean → population mean as $n$ grows | $\bar{x}_n \to \mu$ |
| **CLT** | sample means become Gaussian (any population shape, $n \ge 30$) | $\bar{x} \approx \mathcal{N}(\mu, \sigma^2/n)$ |
| **Standard error** | uncertainty of the mean | $SE = \frac{\sigma}{\sqrt{n}}$ |

**The payoff of data:** 4× more data → 2× smaller SE ($\sqrt{n}$ law). $n=25$: $SE = \sigma/5$; $n=100$: $SE = \sigma/10$.

## INFERENCE — CONFIDENCE INTERVALS & HYPOTHESIS TESTS

**95% confidence interval for the mean:**
$$\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}$$
*Correct meaning: ~95% of such intervals (over repeated experiments) contain the true $\mu$.* NOT "95% chance μ is inside this one."

**Worked (30 sec):** $n = 100$, $\bar{x} = 80$, $s = 20$ → $SE = 2$ → margin $1.96 \times 2 = 3.92$ → CI $[76.08, 83.92]$.

**Hypothesis testing recipe (5 steps):**
1. $H_0$: nothing happening; $H_1$: effect exists.
2. Compute the statistic under $H_0$.
3. p-value = probability of data *this extreme* if $H_0$ true.
4. If $p < \alpha$ (0.05) → reject $H_0$ → effect is real.
5. Otherwise → not enough evidence (never "prove $H_0$").

**The two errors (memorize):**

| | $H_0$ true | $H_0$ false |
| :--- | :--- | :--- |
| Reject $H_0$ | **Type I** (false positive, prob $\alpha$) | ✓ correct |
| Accept $H_0$ | ✓ correct | **Type II** (false negative, prob $\beta$) |

*Type I = believing a lie; Type II = missing the truth. Lower $\alpha$ → more Type II; more data → fewer of both.*

## ESTIMATION — MLE, MAP, BOOTSTRAP

**MLE:** pick parameters that make the observed data most likely:
$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_i \log P(x_i \mid \theta)$$

**Worked (30 sec):** 7 heads in 10 flips → $\hat{p} = 0.7$ (the observed proportion — MLE makes the "obvious" answer provably optimal). Verify: $\mathcal{L}(0.7) = 0.7^7 0.3^3 = 0.00222 > \mathcal{L}(0.5) = 0.00098$ ✓.

**THE KEY TABLE — MLE IS EVERY LOSS FUNCTION:**

| Task | Noise model | Loss MLE produces |
| :--- | :--- | :--- |
| Binary classification | Bernoulli | binary cross-entropy |
| Multi-class classification | Categorical | cross-entropy |
| Regression | Gaussian | **MSE** |
| Regression (robust) | Laplace | MAE |

**MAP:** MLE + prior: $\hat{\theta}_{MAP} = \arg\max_\theta [\log \mathcal{L}(\theta) + \log P(\theta)]$
- **MAP = MLE + regularization.** L2 = Gaussian prior; L1 = Laplace prior. Every $\lambda\|\theta\|^2$ term is a Bayesian prior in disguise.
- Worked: coin with Beta(2,2) prior, 7 heads/10 → posterior Beta(9,5), $\hat{p} = \frac{8}{12} \approx 0.667$ (pulled from 0.7 toward 0.5).

**Bootstrap:** uncertainty for ANY statistic without formulas:
1. Resample with replacement (size n) → 2. compute statistic → 3. repeat B=10,000× → 4. percentiles = CI.
- Bonus: **random forests ARE bootstrap** (bagging = Bootstrap AGGregating).

## WHERE STATISTICS APPEARS IN ML (one-line map)

| Concept | ML location |
| :--- | :--- |
| Mean / variance | feature standardization, batch norm |
| IQR / boxplot | EDA, outlier detection |
| Standardization | preprocessing for gradient models |
| Correlation | feature selection, collinearity checks |
| CLT / SE | model comparison, monitoring bands |
| Confidence interval | reporting metrics honestly |
| p-values | A/B testing, "significant improvement" claims |
| MLE | cross-entropy & MSE losses (all training!) |
| MAP | L1/L2 regularization, Bayesian NN |
| Bootstrap | random forests, uncertainty estimates |

## TOP 5 COMMON MISTAKES

1. Using $n$ instead of $n-1$ for sample variance (and mixing up `np.var` vs `pd.var` defaults).
2. Misreading confidence intervals ("95% chance μ is inside") — it's about the *procedure*, not the single interval.
3. Saying a non-significant result "proves" $H_0$ — absence of evidence ≠ evidence of absence (Type II risk).
4. Forgetting that MLE ⟹ loss: cross-entropy and MSE aren't arbitrary — they're maximum likelihood in disguise.
5. Standardizing with test-set statistics — fit $(\bar{x}, s)$ on training only (leakage!).

> Full detail + worked examples: `MLDL/maths/statistics.md` — then `probability` (the distributions MLE uses), `calculus` (derivatives that maximize likelihood).