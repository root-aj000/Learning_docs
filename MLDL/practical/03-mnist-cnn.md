---
title: Example 3 — Handwritten Digit Recognition (CNN, end-to-end)
description: The full job pipeline on a real vision task — recognize handwritten digits. From a 1-line softmax classifier to a CNN, each step in NumPy and PyTorch, every line tagged with the math inside. Where the chain rule, softmax, and convolution actually live.
tags: [math, ml, practical, cnn, vision, pytorch, backprop, softmax]
---

# EXAMPLE 3 — HANDWRITTEN DIGIT RECOGNITION (the CNN pipeline)

> **The problem (as it would arrive at a job):** a bank needs to read handwritten digits from check images. You must build a classifier that maps a 8×8 (or 28×28) grayscale image to one of 10 digits. We use sklearn's built-in digits dataset (a small MNIST — same shapes, no download).

**This doc's promise:** you'll watch accuracy climb **93% → 96% → 98%** as we add exactly two things to the loop from `00-mental-model.md` — a hidden layer (which forces the chain rule to work for real) and convolutions (which make the matmul image-aware). Every line tagged with the math inside.

---

## 0. The one-line version

> **A digit classifier is:** image (a vector of 64 numbers) → matmuls → softmax over 10 classes → predicted digit. Training runs the same five-line loop; the only new math is **cross-entropy over 10 classes** and (for the CNN) **convolutions = dot products at every image patch**.

---

## 1. Data — images are just vectors

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()                          # 1797 images, 8x8 pixels, 10 classes
X = digits.images.reshape(1797, -1) / 16.0      # flatten 8x8 -> 64 numbers, scale 0..1
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
print("train:", Xtr.shape, "test:", Xte.shape, "classes:", np.unique(y))
```
```
train: (1347, 64) test: (450, 64) classes: [0 1 2 3 4 5 6 7 8 9]
```

**Math tagged here:** an image is **a vector in 64-dimensional space** (your linear algebra docs). Dividing by 16 is normalization (statistics docs) — pixels are 0–16, scaled to 0–1 so gradients behave. This is the same scaling lesson as house prices, one level deeper.

---

## 2. Level 1 — a linear model: `softmax(X @ W)`, from scratch

**The math:** for each image, compute 10 scores — one per digit — as 10 dot products (`X @ W` where W is 64×10). Then **softmax** turns the 10 scores into 10 probabilities that sum to 1. The prediction is the class with the highest probability. Loss = **cross-entropy** (the surprise meter over 10 classes).

```python
W = np.zeros((64, 10)); b = np.zeros(10)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
def ce_loss(logits, y):
    return -np.mean(log_softmax(logits)[np.arange(len(y)), y])

for epoch in range(100):
    logits = Xtr @ W + b                      # FORWARD: matmul (1347 x 64 @ 64 x 10)
    loss = ce_loss(logits, ytr)               # LOSS: cross-entropy
    P = softmax(logits)
    grad_W = Xtr.T @ (P - np.eye(10)[ytr]) / len(Xtr)   # dCE/dW = X^T (P - onehot_y)
    grad_b = (P - np.eye(10)[ytr]).mean(axis=0)
    W -= 0.5 * grad_W; b -= 0.5 * grad_b      # UPDATE: gradient descent

def predict(X):
    return softmax(X @ W + b).argmax(axis=1)
print("A. linear softmax (from scratch): acc =", round(accuracy_score(yte, predict(Xte)), 4))
```
```
A. linear softmax (from scratch):  acc=0.9311  loss=0.4147
```

**Read the numbers — 93.1% with ZERO hidden layers.** Notice the gradient: `Xᵀ(P − onehot)` — *prediction minus truth*, exactly like logistic regression, now over 10 classes. This is the single most important gradient in all of deep learning. A linear model reads every image as 10 dot products — and that alone gets 93% on digits.

---

## 3. Level 2 — a hidden layer: the chain rule goes to work

**Why it's needed:** some classes can't be separated by 10 straight lines in pixel space (e.g., 3 vs 8 in certain handwritings). A hidden layer lets the model learn *features* first, then classify the features. **This is the moment backprop (chain rule) becomes non-trivial** — the gradient must flow through two layers:

```python
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(a): return a * (1 - a)

n_hid = 64
W1 = np.random.default_rng(0).normal(0, 0.5, (64, n_hid)); b1 = np.zeros(n_hid)
W2 = np.random.default_rng(0).normal(0, 0.5, (n_hid, 10)); b2 = np.zeros(10)
onehot = np.eye(10)[ytr]

for epoch in range(300):
    z1 = Xtr @ W1 + b1; a1 = sigmoid(z1)          # hidden layer: matmul + non-linearity
    logits = a1 @ W2 + b2                         # output layer: matmul
    P = softmax(logits)
    dL_dlogits = (P - onehot) / len(Xtr)
    dW2 = a1.T @ dL_dlogits; db2 = dL_dlogits.sum(axis=0)
    dL_da1 = dL_dlogits @ W2.T                    # chain rule: flow BACK through W2
    dL_dz1 = dL_da1 * sigmoid_deriv(a1)           # chain rule: through sigmoid
    dW1 = Xtr.T @ dL_dz1; db1 = dL_dz1.sum(axis=0)
    W1 -= 0.5 * dW1; b1 -= 0.5 * db1
    W2 -= 0.5 * dW2; b2 -= 0.5 * db2
print("B. numpy MLP (1 hidden, 64 units): acc =", round(accuracy_score(yte, predict_mlp(Xte)), 4))
```
```
B. numpy MLP (1 hidden, 64 units):   acc=0.9600
```

**Read the numbers: +2.9 points from one hidden layer.** The two new lines `dL_da1 = dL_dlogits @ W2.T` and `dL_dz1 = dL_da1 * sigmoid_deriv(a1)` **are the chain rule** — multiplying steepnesses backwards through the stack, exactly as in the calculus docs. Everything PyTorch's `loss.backward()` does for a 1000-layer net is these two lines, repeated.

---

## 4. Level 3 — a CNN in PyTorch: where the matmul becomes a convolution

**Why convolutions:** the linear model treats pixel 37 as unrelated to pixel 38. But images have *local structure* — a stroke is nearby pixels. A **convolution** slides a small filter (3×3) over the image, computing **a dot product between the filter and every 3×3 patch**. That's it: convolution = dot product, repeated at every position.

```python
import torch, torch.nn as nn, torch.nn.functional as F

Xtr_img = torch.tensor(Xtr.reshape(-1, 1, 8, 8), dtype=torch.float32)
Xte_img = torch.tensor(Xte.reshape(-1, 1, 8, 8), dtype=torch.float32)
ytr_t, yte_t = torch.tensor(ytr), torch.tensor(yte)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 16 filters of 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 2 * 2, 10)           # final matmul -> 10 logits

    def forward(self, x):
        x = F.relu(self.conv1(x))      # conv + non-linearity (ReLU: max(0, z))
        x = F.max_pool2d(x, 2)         # shrink 8x8 -> 4x4, keep the max of each 2x2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)         # 4x4 -> 2x2
        x = x.view(x.size(0), -1)      # flatten to a vector
        return self.fc(x)              # the final matmul: 128 numbers -> 10 scores

model = TinyCNN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(50):
    opt.zero_grad()
    logits = model(Xtr_img)                       # FORWARD
    loss = F.cross_entropy(logits, ytr_t)         # LOSS: cross-entropy
    loss.backward()                               # BACKWARD: chain rule through convs
    opt.step()                                    # UPDATE: Adam
with torch.no_grad():
    preds = model(Xte_img).argmax(dim=1).numpy()
print("C. torch CNN: acc =", round(accuracy_score(yte, preds), 4))
```
```
C. torch CNN:                    acc=0.9800  loss 2.3046 -> 0.0438
```

**Read the numbers: 98% — the same five-line loop, better vision.** The loss starts at **2.3046 = ln(10)** — the "model knows nothing" signature for 10 classes (remember ln 2 = 0.6931 for binary; now there are 10 equally-likely classes, so ln 10). It falls to 0.044.

**Every line of the CNN is math you know:**
- `nn.Conv2d` → **dot products at every patch** (see below)
- `F.relu` → the non-linearity (without it, many matmuls = one matmul; from the mental model)
- `F.max_pool2d` → **downsampling** (keep the strongest response — a crude "attention")
- `F.cross_entropy` → **log-softmax + cross-entropy in one call** (the surprise meter)
- `loss.backward()` → **the chain rule**, now through convolutions
- `opt.step()` with Adam → gradient descent with per-weight learning rates + momentum

---

## 5. What a convolution actually is (the one thing to see)

```python
patch = Xtr[0, :9].reshape(3, 3)              # one 3x3 patch of image 0
filt = rng.normal(size=(3, 3))                # one learned 3x3 filter
conv_out = np.sum(patch * filt)               # THE CONVOLUTION: one dot product
```
```
D. 3x3 patch dot 3x3 filter = 0.846  <- that's what nn.Conv2d does, at every patch, with every filter
```

**That's it.** `nn.Conv2d` with 16 filters = 16 dot products at every patch, at every image — the matmul from your linear algebra docs, made local. If the filter is "edge detector" weights, its dot products light up where edges are. **The model learns the filters; you just give it the dot product.**

---

## 6. Where regularization (L2) enters — the norm from linear algebra

Real training adds a term to the loss: penalize big weights. That's the **L2 norm** squared — from your linear algebra docs:

```python
def l2_penalty(W1_, W2_):
    return np.sum(W1_ ** 2) + np.sum(W2_ ** 2)
print(f"E. L2 penalty: {l2_penalty(W1, W2):.1f}  <- add this to the loss to keep weights small")
```
```
E. L2 penalty of current MLP weights: 1369.9
```

**Where it lives in real code:** `optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)` — the `weight_decay` argument *is* this L2 term. It prevents overfitting by refusing to let any weight grow huge (the norm, from your math docs, used as a tax).

---

## 7. The map — where each math concept lives

| Where it happened | Math | Code |
| :--- | :--- | :--- |
| Image → numbers | vector space (64-dim) + normalization | `reshape`, `/ 16.0` |
| Level 1 model | matmul + **softmax** over 10 classes | `X @ W`, `softmax()` |
| Level 1 loss | cross-entropy, gradient = `Xᵀ(P−y)` | `ce_loss`, `Xtr.T @ (P - onehot)` |
| Level 2 | **chain rule** through a hidden layer | `dL_da1 @ W2.T`, `* sigmoid_deriv` |
| Level 3 | **convolution = local dot products** | `nn.Conv2d` |
| Non-linearity | ReLU (the reason layers stack) | `F.relu` |
| Pooling | downsampling | `F.max_pool2d` |
| Regularization | **L2 norm** as a weight tax | `weight_decay` |
| Training | gradient descent (Adam = +momentum) | `opt.step()` |

**The progression to remember: 93.1% → 96.0% → 98.0%.** One hidden layer bought 2.9 points; convolutions bought another 2. In a real job, you'd keep going: more data, more layers, data augmentation (rotating images = **linear algebra transformations**!) — but the loop and the math never change.

**Next:** `04-sentiment-rnn.md` — the same loop on text, where the matmul learns to *remember* (LSTM gates = sigmoid + element-wise multiply).