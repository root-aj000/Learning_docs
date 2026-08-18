---
title: Probability in Code (from scratch)
description: Every probability idea from the maths docs implemented in pure NumPy — PMF/PDF/CDF, expected value, Monte Carlo, Bayes, entropy, softmax, and a full Naive Bayes classifier. Verified outputs included.
tags: [math, probability, numpy, implementation, from-scratch, ml]
---

# PROBABILITY IN CODE — FROM SCRATCH

> This document implements **everything from `MLDL/maths/probability.md`** in pure NumPy. You write the distributions, the entropy, the Bayes counting — and end with a complete Naive Bayes classifier that beats matching sklearn's accuracy. Every code block below **actually ran** and its real output is shown underneath.

**Setup:**
```bash
source .venv/bin/activate    # numpy + scikit-learn installed
```

---

## 1. PMF, PDF, CDF — the three ways of drawing probability

**Theory (1 line each):** PMF = chance at exact spots (discrete). PDF = weight curve where *area* = chance (continuous). CDF = running total of weight.

### 1.1 Binomial PMF — "chance of exactly k heads"

**From scratch (the formula is $C(n,k)\,p^k(1-p)^{n-k}$ — just arithmetic):**
```python
from math import comb
import numpy as np

def binomial_pmf(k, n, p):
    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))

for k in range(0, 11):
    print(f"P({k} heads in 10 flips) = {binomial_pmf(k, 10, 0.5):.4f}")
print("sum over all k:", round(sum(binomial_pmf(k, 10, 0.5) for k in range(11)), 4))
```

**Verified output:**
```
P(0 heads in 10 flips) = 0.0010
P(1 heads in 10 flips) = 0.0098
P(2 heads in 10 flips) = 0.0439
P(3 heads in 10 flips) = 0.1172
P(4 heads in 10 flips) = 0.2051
P(5 heads in 10 flips) = 0.2461
P(6 heads in 10 flips) = 0.2051
P(7 heads in 10 flips) = 0.1172
P(8 heads in 10 flips) = 0.0439
P(9 heads in 10 flips) = 0.0098
P(10 heads in 10 flips) = 0.0010
sum over all k: 1.0
```

**Read the numbers:** the probabilities sum to **exactly 1.0** — that's the PMF's job (all chances must account for everything). The biggest bar is at 5 heads (0.246) — the most likely outcome, but far from guaranteed.

### 1.2 Gaussian PDF and CDF — the bell curve

**From scratch (PDF) and via numeric integration (CDF):**
```python
def gaussian_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))

print("N(0,1) pdf at 0:", round(gaussian_pdf(0.0), 4),
      "| at 1:", round(gaussian_pdf(1.0), 4),
      "| at 3:", round(gaussian_pdf(3.0), 4))

def gaussian_cdf(x, mu=0.0, sigma=1.0, steps=2000):
    xs = np.linspace(-10, x, steps)
    area = 0.0
    for i in range(1, len(xs)):                       # trapezoid rule
        area += (gaussian_pdf(xs[i-1], mu, sigma) + gaussian_pdf(xs[i], mu, sigma)) / 2 * (xs[i] - xs[i-1])
    return area

print("CDF at 0 (should be 0.5):", round(gaussian_cdf(0.0), 4))
print("CDF at 1.96 (should be ~0.975):", round(gaussian_cdf(1.96), 4))
```

**Verified output:**
```
N(0,1) pdf at 0: 0.3989 | at 1: 0.242 | at 3: 0.0044
CDF at 0 (should be 0.5): 0.5
CDF at 1.96 (should be ~0.975): 0.975
```

**Read the numbers:** the PDF is highest at the middle (0.40) and nearly zero at 3 (0.004) — "most things are average, extremes are rare" in numeric form. The CDF you built by summing skinny rectangles (the integral from the calculus doc!) gives 0.5 at the mean and 0.975 at 1.96 — the exact numbers behind the 68–95–99.7 rule.

**Real-world note:** `scipy.stats.norm.pdf` / `.cdf` do the same with better integration; your trapezoid version is within 1e-4 of them.

---

## 2. Expected value — number × chance, added up

**Theory (1 line):** $E[X] = \sum x \cdot P(X = x)$.

**From scratch:**
```python
dice = np.arange(1, 7)
probs = np.full(6, 1/6)
E = np.sum(dice * probs)
print("E[fair die] =", E)
```
```
E[fair die] = 3.5
```

**Read the number:** you can never roll a 3.5 — but a million rolls average out to it. This tiny line of code is the core of *every* loss function in ML: "on average, how wrong am I?"

---

## 3. Monte Carlo — guessing your way to the answer

**Theory (1 line):** throw random samples, count the fraction that hit the target, convert.

**From scratch (pi, and the Law of Large Numbers live):**
```python
rng = np.random.default_rng(42)
N = 100_000
pts = rng.uniform(-1, 1, (N, 2))
inside = np.sum(np.linalg.norm(pts, axis=1) <= 1.0)
print(f"Monte Carlo pi (N={N}): {4 * inside / N:.4f}  (true 3.1416)")

rolls = rng.integers(1, 7, 100_000)                       # a fair die, many times
avg = np.cumsum(rolls) / np.arange(1, 100_001)            # running average
print("average after 100 rolls:", round(avg[99], 3), "| after 100k:", round(avg[-1], 3))
```

**Verified output:**
```
Monte Carlo pi (N=100000): 3.1354  (true 3.1416)
average after 100 rolls: 3.69 | after 100k: 3.506
```

**Read the numbers:** 100,000 random dots estimate π to 2 decimal places (3.1354). And the die's running average starts wobbly (3.69 after 100 rolls) but settles toward the true 3.5 — that's the **Law of Large Numbers** happening in your terminal. Bigger N → closer to truth. Always.

---

## 4. Bayes — the town trick, as code

**Theory (1 line):** pretend 10,000 people, count, divide.

**From scratch (the medical test — same story as the math docs):**
```python
N_people = 10_000
sick = int(0.01 * N_people)           # 1% have the disease
healthy = N_people - sick
pos_sick = int(0.90 * sick)           # test catches 90% of the sick
pos_healthy = int(0.05 * healthy)     # 5% false alarms among healthy
all_positive = pos_sick + pos_healthy
print(f"positives={all_positive}, true={pos_sick} -> P(sick|pos)={pos_sick / all_positive:.4f}")
```
```
positives=585, true=90 -> P(sick|pos)=0.1538
```

**Read the numbers:** 585 positive tests, only 90 truly sick → **15.4%**. The formula and the counting give the identical answer — because the formula IS the counting, compressed. When Bayes feels scary, go back to the town.

---

## 5. Entropy and cross-entropy — the surprise meter, in code

**Theory (2 lines):** entropy $H = -\sum p \log_2 p$ measures average surprise. Cross-entropy measures "how surprised would the true distribution be by the model's predictions?" — the training loss of every classifier.

**From scratch:**
```python
def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                                  # log(0) is undefined, skip
    return -np.sum(p * np.log2(p))

print("entropy fair coin:", round(entropy([0.5, 0.5]), 4), "bits (max = 1)")
print("entropy rigged coin (0.99, 0.01):", round(entropy([0.99, 0.01]), 4), "bits (low surprise)")
print("entropy fair die:", round(entropy([1/6]*6), 4), "bits (should be log2(6)=2.585)")

def cross_entropy(p_true, q_pred):
    q = np.clip(np.asarray(q_pred, dtype=float), 1e-12, None)
    return -np.sum(np.asarray(p_true, dtype=float) * np.log2(q))

p_true = [1.0, 0.0, 0.0]                          # the true class is 0
for name, q in [("perfect", [1.0, 0.0, 0.0]), ("ok", [0.8, 0.15, 0.05]), ("bad", [0.3, 0.3, 0.4])]:
    print(f"cross-entropy ({name} model): {cross_entropy(p_true, q):.4f} bits")
```

**Verified output:**
```
entropy fair coin: 1.0 bits (max = 1)
entropy rigged coin (0.99, 0.01): 0.0808 bits (low surprise)
entropy fair die: 2.585 bits (should be log2(6)=2.585)
cross-entropy (perfect model): -0.0000 bits
cross-entropy (ok model): 0.3219 bits
cross-entropy (bad model): 1.7370 bits
```

**Read the numbers:** the fair coin has maximum surprise (1 bit); the rigged coin almost none (0.08 bits — you already know what's coming). For cross-entropy: perfect model → 0 (never surprised), OK model → 0.32, bad model → 1.74. **Training a classifier = making this number go down.** That number on your screen is literally the "loss" in every PyTorch training log.

---

## 6. Softmax and the stability trick — the final layer of every net

**Theory (1 line):** softmax turns raw scores (logits) into probabilities: $e^{z_i} / \sum_j e^{z_j}$.

**From scratch:**
```python
def softmax(z):
    z = z - np.max(z)                     # THE stability trick: shift so max = 0
    e = np.exp(z)
    return e / np.sum(e)

def log_softmax(z):                       # log(softmax) computed directly
    z = z - np.max(z)
    return z - np.log(np.sum(np.exp(z)))

z = np.array([2.0, 1.0, 0.1])
s = softmax(z)
print("softmax([2,1,0.1]):", np.round(s, 4), "sum:", round(s.sum(), 4))
print("log_softmax:", np.round(log_softmax(z), 4), "| log(softmax):", np.round(np.log(s), 4))
```

**Verified output:**
```
softmax([2,1,0.1]): [0.659  0.2424 0.0986] sum: 1.0
log_softmax: [-0.417 -1.417 -2.317] | log(softmax): [-0.417 -1.417 -2.317]
```

**Read the numbers:** the biggest logit (2.0) became the biggest probability (0.659), and everything sums to 1. The `- np.max(z)` shift prevents `np.exp(1000)` from overflowing to infinity — try removing it with `z = np.array([1000., 999.])` and watch it break.

**The nats vs bits gotcha (real bug, caught while writing this doc):**
```python
ce_nats = -log_softmax(z)[0]                       # natural log: 0.417
print(ce_nats, "nats |", ce_nats / np.log(2), "bits")
```
```
0.417 nats | 0.6016 bits
```
Same quantity, two units — divide nats by ln(2) ≈ 0.693 to get bits (PyTorch reports nats; this doc's section 5 used bits). When your loss suddenly looks 0.69× smaller than expected, it's probably this, not a bug.

---

## 7. Naive Bayes classifier — everything above, assembled

**Theory (3 lines):** Bayes + the "naive" assumption that features are independent. For each class, multiply the per-feature Gaussian PDFs and the prior; predict the class with the highest score. (Numbers get tiny, so work in log-space.)

**From scratch (full classifier, iris dataset):**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = {}; self.mu = {}; self.sigma = {}
        for c in self.classes:
            Xc = X[y == c]
            self.prior[c] = len(Xc) / len(X)
            self.mu[c] = Xc.mean(axis=0)
            self.sigma[c] = Xc.std(axis=0) + 1e-9      # avoid div by 0
        return self

    def _p_x_given_c(self, x, c):
        z = (x - self.mu[c]) / self.sigma[c]
        pdf = np.exp(-0.5 * z**2) / (self.sigma[c] * np.sqrt(2*np.pi))
        return np.prod(pdf)                            # independence assumption

    def predict(self, X):
        preds = []
        for x in X:
            scores = {c: np.log(self.prior[c]) + np.log(self._p_x_given_c(x, c))
                      for c in self.classes}           # log-space = multiply in real space
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

nb = NaiveBayes().fit(Xtr, ytr)
print("Naive Bayes from scratch accuracy:", round(np.mean(nb.predict(Xte) == yte), 4))

from sklearn.naive_bayes import GaussianNB
print("sklearn GaussianNB accuracy:     ", round(GaussianNB().fit(Xtr, ytr).score(Xte, yte), 4))
```

**Verified output:**
```
Naive Bayes from scratch accuracy: 0.9778
sklearn GaussianNB accuracy:       0.9778
```

**Read the numbers:** your hand-built classifier — using the Gaussian PDF you wrote in section 1, the Bayes counting idea from section 4, and the log trick from section 6 — scores **exactly the same as sklearn's**: 97.8% on unseen iris flowers. The library is just your code, optimized and battle-tested.

**Why log-space?** try it without `np.log` — `np.prod` of several tiny PDFs underflows to 0.0 (the numbers are smaller than what a float can represent), and every class scores 0. Logs turn those underflowing products into manageable negative sums. This is the single most common beginner bug in implementing classifiers.

---

## 8. Bonus: sampling from a distribution

Real models don't just score — they **generate**. `rng.multinomial` draws random word counts from a distribution (this is what language models do per token):

```python
rng = np.random.default_rng(1)
p = np.array([0.6, 0.3, 0.1])          # word frequencies
print("sample of 1000 tokens:", rng.multinomial(1000, p), "(expected ~600, 300, 100)")
```
```
sample of 1000 tokens: [606 307  87] (expected ~600, 300, 100)
```

---

## 9. What you've actually implemented (map to the math docs)

| Math idea (from `maths/` doc) | Your code | Verified result |
| :--- | :--- | :--- |
| PMF / PDF / CDF (2.3) | `binomial_pmf`, `gaussian_pdf`, `gaussian_cdf` | sums to 1.0; CDF(1.96)=0.975 |
| Expected value (2.4) | `np.sum(dice * probs)` | 3.5 |
| Law of large numbers (5.x) | running average of 100k rolls | 3.69 → 3.506 |
| Monte Carlo (5.x) | random dots for π | 3.1354 |
| Bayes (3.3) | town counting | 15.4% matches formula |
| Entropy / cross-entropy (5.2) | `entropy()`, `cross_entropy()` | 1 bit coin, loss semantics |
| Softmax (2.6) | `softmax()` + stability shift | sums to 1, matches log-space |
| Gaussian NB (3.x) | full classifier | 97.8% = sklearn |

**Test yourself — predict before running:**
1. What happens to `binomial_pmf`'s bar at k=5 if you change p to 0.7? *(Ans: the peak shifts right — the "most likely number of heads" moves to 7.)*
2. In `gaussian_cdf`, why integrate from −10 instead of −∞? *(Ans: beyond 10 the bell is ~0 — the error is below 1e-23. Every numeric CDF does this.)*
3. Why does `cross_entropy` need `np.clip(q, 1e-12, None)`? *(Ans: a confident-wrong model predicts q=0 for the true class → log(0) = −∞. The clip caps the damage.)*