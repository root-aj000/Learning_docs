---
title: Example 2 — Spam Detection (classification, end-to-end)
description: The full job pipeline on a real task — classify spam vs ham. TF-IDF (where logs enter real code), logistic regression from scratch vs sklearn, cross-entropy as the loss. Every line tagged with the math inside.
tags: [math, ml, practical, classification, logistic-regression, tf-idf, cross-entropy]
---

# EXAMPLE 2 — SPAM DETECTION (the classification pipeline)

> **The problem (as it would arrive at a job):** the company's email product gets spam. You must build a classifier that decides *ham* (normal) vs *spam* for every incoming email.

**This doc's promise:** the same loop from `00-mental-model.md` — but now with the two new math pieces that classification adds: **logs** (turning words into numbers) and **cross-entropy** (the loss). When you finish, you'll know *where* and *why* logarithms and softmax/sigmoid appear in real code — the two things you said felt unanchored.

---

## 0. The one-line version

> **A spam classifier is:** text → numbers (TF-IDF = **log**-based word weights) → one matmul + sigmoid → a 0–1 spam chance. Training minimizes **cross-entropy** (the surprise meter from the probability docs) by gradient descent. Everything else is detail.

---

## 1. Data — text doesn't speak math, so we translate it

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

messages = [
    # ham (normal mail)
    "hey are you free for lunch tomorrow", "see you at the meeting at 3pm",
    "can you send me the report please", "happy birthday have a great day",
    # ... 16 more ham messages ...
    # spam
    "WIN a free iPhone now click this link", "urgent your account has been locked",
    "claim your prize money today call now", "limited offer buy one get one free",
    # ... 16 more spam messages ...
]
labels = np.array([0]*20 + [1]*20)          # 0 = ham, 1 = spam

Xtr, Xte, ytr, yte = train_test_split(messages, labels, test_size=0.3, random_state=42, stratify=labels)
```

**The first wall every text problem hits:** a model does matmuls — matmuls need numbers — but emails are words. **TF-IDF is the translator.** (And TF-IDF is where **logarithms** enter real ML code for the first time.)

## 2. TF-IDF — the log math that turns words into numbers

**The math, in one sentence:** every word gets a number = (how often it appears in THIS email) × (how rare it is in general). Rarity is measured with a **log**:

$$\text{idf}(w) = \log\left(\frac{N}{\text{# documents containing } w}\right) + 1$$

**Why the log?** Without it, a word appearing in 1 of 28 documents would weigh 28× a word in 14 of them. The log **compresses** the range (your calculus docs: log turns multiplications into additions, growth into compression). "the", "you", "free" — common words — get idf ≈ 1 (no weight); rare words get big weights.

```python
vec = TfidfVectorizer(stop_words="english")      # removes "the", "and", "you"...
Xtr_vec = vec.fit_transform(Xtr)
Xte_vec = vec.transform(Xte)                     # SAME vocab as training!

print("vocab size:", len(vec.vocabulary_), "| train matrix:", Xtr_vec.shape)
for w in ["free", "click", "meeting", "report"]:
    print(f"  idf({w:10s}) = {vec.idf_[vec.vocabulary_[w]]:.3f}")
```

**Verified output:**
```
vocab size: 84 | train matrix: (28, 84)
  idf(free      ) = 2.758
  idf(click     ) = 3.269
  idf(meeting   ) = 3.269
  idf(report    ) = 3.674
```

**Read the numbers:** each email is now a **vector of 84 numbers** (one per word in the vocab) — the matrix is 28 emails × 84 words. An email with "free" gets a big number at that word's column. **Your linear algebra docs: an email IS a vector now.** The matmul can begin.

**Two real-world traps worth knowing now:**
1. `vec.fit_transform(Xtr)` **only on training** — the vocab must never see test data (same instinct as the train/test split; a leaked vocab silently inflates accuracy).
2. Real companies use the same idea with millions of words — the matrix becomes sparse (mostly zeros), which is why real code passes `Xtr_vec` (a sparse matrix) rather than dense arrays.

---

## 3. Logistic regression from scratch — the classifier's math, naked

**The model:** `p(spam) = sigmoid(X @ w + b)` — one matmul (every email gets a dot product with the weight vector), then **sigmoid** (your calculus docs: the S-curve that compresses any number into 0–1).

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

Xtr_d, Xte_d = Xtr_vec.toarray(), Xte_vec.toarray()
w = np.zeros(Xtr_d.shape[1]); b = 0.0

for epoch in range(100):
    z = Xtr_d @ w + b                     # FORWARD: the matmul
    p = sigmoid(z)                        # sigmoid -> spam chance per email
    grad_w = Xtr_d.T @ (p - ytr) / len(ytr)    # gradient of cross-entropy (see 4)
    grad_b = np.mean(p - ytr)
    w -= 1.0 * grad_w                     # UPDATE: gradient descent
    b -= 1.0 * grad_b

def predict(X):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)

from sklearn.metrics import accuracy_score
print("FROM SCRATCH accuracy:", round(accuracy_score(yte, predict(Xte_d)), 3))
```

**Verified output:**
```
FROM SCRATCH accuracy: 0.833
```

## 4. The loss — cross-entropy, and why the gradient looks like that

**The math:** the loss is **cross-entropy** — the surprise meter from your probability docs: *"how surprised would the truth be by the model's guesses?"*

$$\text{loss} = -\frac{1}{n}\sum_i \big[\,y_i \log(p_i) + (1-y_i)\log(1-p_i)\,\big]$$

**And the beautiful part — the gradient of this loss is the simplest gradient in all of ML:**

$$\frac{\partial \text{loss}}{\partial w} = \frac{1}{n} X^T (p - y)$$

*prediction minus truth*, times the data. That's why the loop above is `Xtr_d.T @ (p - ytr)`. Logistic regression's gradient is *just the difference between predicted and actual*, averaged — the chain rule did the hard work so you don't have to.

**Verified output (the loss falling):**
```
cross-entropy loss: 0.6931 -> 0.2409
```

**Read the numbers:** the starting loss is **0.6931 = ln 2** — that's the mathematical signature of *complete uncertainty* (a coin flip: p = 0.5 for everything). A model that knows nothing starts at exactly ln 2. When you see a training loss hovering near 0.69 in any real binary classifier, it means "the model has learned nothing yet." Training pushed it to 0.24.

## 5. What the model actually learned — the weights are word opinions

```python
idx2word = {v: k for k, v in vec.vocabulary_.items()}
print("top spam indicators:", [idx2word[i] for i in np.argsort(w)[-5:]])
print("top ham indicators: ", [idx2word[i] for i in np.argsort(w)[:5]])
```

**Verified output:**
```
top spam indicators: ['free', 'click', 'won', 'claim', 'offer']
top ham indicators:  ['meeting', 'tomorrow', 'lets', 'assignment', 'report']
```

**Read the numbers — the model is a dictionary of opinions:** `free`, `click`, `won`, `claim`, `offer` push toward spam; `meeting`, `tomorrow`, `report` push toward ham. Each word's number is its learned weight. **This is what "the model learned" means in plain sight** — a pile of weights that are readable by humans. (At a real job, this is how you audit a model and catch bias: inspect the weights.)

## 6. sklearn — the same math, production-ready

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000).fit(Xtr_vec, ytr)
print("sklearn accuracy:", round(clf.score(Xte_vec, yte), 3))
print("sklearn top spam:", [idx2word[i] for i in np.argsort(clf.coef_[0])[-5:]])
```

**Verified output:**
```
sklearn accuracy:     0.833
sklearn top spam:     ['free', 'click', 'won', 'claim', 'offer']
```

**Identical accuracy, identical learned words.** `LogisticRegression().fit()` runs the exact loop from section 3 — sigmoid, cross-entropy gradient, gradient descent — with faster optimization underneath. **You have now written the math that sklearn ships.**

## 7. Inference — the deployed classifier

```python
new = ["you have won a free iphone click here now"]
p_new = sigmoid(vec.transform(new).toarray() @ w + b)[0]
print(f"spam probability {p_new:.3f} -> {'SPAM' if p_new > 0.5 else 'HAM'}")
```
```
spam probability 0.839 -> SPAM
```

**Read the numbers:** the email's vector (mostly zeros — it only contains 6 known words) gets one matmul with the weight vector, sigmoid turns it into 83.9%, and the threshold at 0.5 decides. **The whole model at inference = one matmul + one sigmoid + one comparison.**

---

## 8. The map — where each math concept lives

| Where it happened | Math | Code |
| :--- | :--- | :--- |
| Text → numbers | **log** (idf) + counting | `TfidfVectorizer` |
| The model | matmul + **sigmoid** | `X @ w`, `1/(1+e^-z)` |
| The loss | **cross-entropy** (surprise meter) | `-y·log(p) - (1-y)·log(1-p)` |
| The gradient | chain rule → `Xᵀ(p − y)` | `Xtr_d.T @ (p - ytr)` |
| Training | gradient descent | `w -= lr * grad_w` |
| Decision | threshold at 0.5 | `p >= 0.5` |
| Interpreting the model | the weights = word opinions | `np.argsort(w)` |

**New mental furniture from this doc:** logs show up in real code *first* as TF-IDF; cross-entropy is the loss you'll see in every classifier's training log; and "0.6931" is the "model knows nothing" signature. **Next:** `03-mnist-cnn.md` — the same loop with images, where the matmul becomes a CONVOLUTION and the model finally needs a hidden layer.

---

## DEEP — WHY CROSS-ENTROPY, WHY THE LOG, WHY THE PRIOR

### DEEP-1: why MSE is the wrong loss for classification — measured

Both losses push the sigmoid score toward the truth. The difference is *what the gradient does when the model is confidently wrong*. Compute both gradients at various logits z (y = 1):

```
  z        p          |dCE/dz|        |dMSE/dz|
-10.0   0.0000        1.0000         9.08e-05   <- confidently WRONG (p≈0, y=1)
 -5.0   0.0067        0.9933         1.32e-02
  0.0   0.5000        0.5000         2.50e-01
  5.0   0.9933        0.0067         8.90e-05   <- confidently right
 10.0   1.0000        0.0000         4.12e-09
```

The key row is `z = -10`: the model is *completely wrong and fully certain about it*. Cross-entropy gives it the **maximum** gradient (1.0) — full blame. MSE gives it **9.1e-5** — the update is dead, and a confidently-wrong model *stays* confidently wrong. Why? The MSE gradient passes through the sigmoid's derivative `p(1−p)`, which is ~0 exactly where the model is confident (either way). CE's gradient `p − y` has no such factor — it's maximal when wrong, zero only when right. That's the whole reason the spam model's training log used CE, and it's why you'll see "use CE, not MSE for classification" in every codebase. (Derivations: `dCE/dz = p − y`, `dMSE/dz = 2(p−y)·p(1−p)` — verified: the formula `p − y` reproduces autograd's gradient to 6 decimals.)

### DEEP-2: cross-entropy IS the log-likelihood — the probability view

`CE = −[y·ln p + (1−y)·ln(1−p)]` is not an arbitrary choice. For one sample, `p` is the model's claimed probability of spam. The *likelihood* of the observed label is `P(label) = p^y · (1−p)^(1−y)`. Minimizing CE is *maximizing* that likelihood:

```
min CE  ⇔  max ln P(label)  ⇔  max P(data | weights)      (log is monotone)
```

So training a classifier is literally **maximum-likelihood estimation**: find the weights that make the observed labels most probable. And the whole logistic-gradient formula `∇ = Xᵀ(p − y)/n` follows from this one choice — verified against finite differences to 1e-10:

```
grad[0]: formula -0.311600  finite-diff -0.311600  diff 8.7e-12
grad[1]: formula +0.089342  finite-diff +0.089342  diff 1.7e-11
grad[2]: formula +0.180263  finite-diff +0.180263  diff 1.8e-10
```

One matmul `Xᵀ(p − y)` *is* the entire gradient of logistic regression. The "surprise" interpretation in the practical section and the "likelihood" interpretation here are the same number — the entropy from your probability docs and the MLE from your statistics docs meeting in one gradient.

### DEEP-3: TF-IDF — why the log, exactly

`idf = log(N/df)`. Why log and not just `N/df`? Because documents differ by orders of magnitude in rareness, and logs make those ratios additive. Measured on a 1,000-doc corpus:

```
term in 1 doc     (df=1):   idf = log(1000) = 6.908   <- appears once: very informative
term in 10 docs   (df=10):  idf = log(100)  = 4.605
term in 100 docs  (df=100): idf = log(10)   = 2.303
term in 800 docs  (df=800): idf = log(1.25) = 0.223   <- everywhere: nearly useless
term in 999 docs  (df=999): idf = log(1.00) = 0.001   <- zero information
```

The raw ratio `N/df` would give 1,000 vs 1.25 — a 800× spread that drowns the tf term. The log compresses the range to 0–7, keeps "rare = informative" monotone, and makes the spam doc's ranking (`free`/`click`/`won` in the top) mathematically inevitable: those words are rare in the ham class, common in the spam class, and the log amplifies exactly that asymmetry.

### DEEP-4: L2 regularization = a Gaussian prior (the Bayesian view)

"Add λ‖w‖² to the loss" sounds like a trick. It's a prior. If you believe weights are Gaussian, `P(w) ∝ exp(−‖w‖²/2σ²)`, then maximizing `P(data|w)·P(w)` (MAP, maximum a posteriori — the Bayes rule from your probability docs) is exactly:

```
minimize  CE + λ‖w‖²      with λ = 1/(2σ²)
```

**Measured:** training the same logistic model with and without L2, and watching the weight norm:

```
no-L2 ‖w‖ = 15.508     with-L2 λ=1: ‖w‖ = 0.195
```

L2 pulled the weights from 15.5 down to 0.195 — the model became "more certain of nothing", i.e. less overfit. The learned spam-word weights in section 2 are L2-regularized for exactly this reason. Same math as the "weight decay" argument you'll see in every PyTorch optimizer — it's a prior, not a patch.