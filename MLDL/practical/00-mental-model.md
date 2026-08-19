---
title: The Mental Model — How a Model Learns (practical map)
description: The one mental model that anchors ALL the math you learned. Every concept pinned to its exact spot in a real training loop. Read this before anything else in this folder.
tags: [math, ml, practical, mental-model, fundamentals]
---

# THE MENTAL MODEL — HOW A MODEL LEARNS FROM DATA

> Everything you learned in `MLDL/maths/` hangs off **one loop**. This doc shows you the loop and pins every math concept to its exact spot. If you feel like the math is "correct but ungrounded", this is the page that grounds it. There is no code here — just the map. Then the example docs (`01-house-price`, `02-spam`, `03-mnist-cnn`, `04-sentiment-rnn`) show the same map in real frameworks.

---

## 1. The one loop — everything is this

```
              ┌──────────────────────────────────────────────┐
              │                                              │
              ▼                                              │
   ┌───────────────────┐   ┌──────────┐   ┌──────────────┐   │
   │  FORWARD PASS     │──▶│   LOSS   │──▶│   BACKWARD   │──▶│
   │  (the model       │   │ (how bad │   │ (who's at    │   │
   │   makes a guess)  │   │  was it?)│   │  fault?)     │   │
   └───────────────────┘   └──────────┘   └──────────────┘   │
                                                             │
        ┌──────────────────┐                                 │
        │   UPDATE         │◀────────────────────────────────┘
        │  (fix the fault) │
        └──────────────────┘
              │
              ▼
        (repeat, thousands of times)
```

Every model in the world — linear regression, logistic, CNN, LSTM, GPT — is this loop. When you read "training" anywhere, you are reading "this loop, repeated."

**Your one-sentence mental model:**
> *A model is a pile of numbers (weights). It guesses (forward pass = math with those numbers). It measures how wrong (loss = one number). It finds who's to blame (backward pass = chain rule, one gradient per number). It adjusts (update = move each number opposite its gradient). Repeat until the loss stops dropping.*

---

## 2. Where each piece of your math lives in the loop

### FORWARD PASS — the model guessing
| Math you learned | What it actually does here |
| :--- | :--- |
| **Matrix multiply / dot product** | The model's brain: `X @ W` computes *all* guesses at once. Every neuron is one dot product of (its weights · the input). `nn.Linear(3, 1)` *is* a matmul. |
| **Softmax** | Turns raw scores into "probabilities" that sum to 1 — the final layer of every classifier. |
| **Sigmoid** | Compresses one score into a 0–1 chance — the final layer of binary classifiers, and the gate inside RNNs/LSTMs. |
| **ReLU/tanh (activations)** | The "non-linearity": without it, many matmuls in a row = one matmul, and the model can't learn curves. |
| **Vector norm** | Regularization (penalizing big weights) and gradient clipping. |

### LOSS — measuring the mistake
| Math you learned | What it actually does here |
| :--- | :--- |
| **MSE (mean squared error)** | Regression loss: "average squared distance between guess and truth" — exactly the variance formula from statistics, applied to errors. |
| **Cross-entropy** | Classification loss: "how surprised would the truth be by the model's guesses?" — the entropy from probability docs, as a loss. |
| **Expected value** | Every loss is an expected value: "on average, over all samples, how wrong am I?" |

### BACKWARD — assigning blame (the chain rule)
| Math you learned | What it actually does here |
| :--- | :--- |
| **Chain rule** | `loss.backward()` — literally the chain rule, executed layer by layer from the output backwards, multiplying steepnesses. One gradient number per weight. |
| **Derivative** | Each gradient is a derivative: "how much would the loss change if THIS weight moved a tiny bit?" |
| **Partial derivative** | Each gradient is a partial: computed holding every other weight still. |
| **Jacobian** | The shape of the backward pass: a matrix of all partials (your framework builds it implicitly). |

### UPDATE — fixing the fault
| Math you learned | What it actually does here |
| :--- | :--- |
| **Gradient descent** | `optimizer.step()` — `w = w - lr * gradient`. The rolling-ball rule. |
| **Learning rate** | `lr` — one number, often the difference between training and diverging. |
| **Momentum / Adam** | Smarter rolling balls: Adam = per-weight learning rates + momentum. |

### DATA — before the loop even starts
| Math you learned | What it actually does here |
| :--- | :--- |
| **Mean / std / z-score** | Normalization — the single most common bug fix in real training. (See house-price doc: raw scale makes one learning rate impossible.) |
| **CLT / variance** | Why bigger datasets and bigger batches give steadier loss curves (noise shrinks like σ/√n). |
| **Bootstrap / confidence intervals** | Measuring how *trustworthy* the model's performance number is. |

---

## 3. The two modes: training vs inference

- **Training** = the loop (forward → loss → backward → update), with gradients computed.
- **Inference** = forward pass only. The math is just matmuls + activations + softmax. This is what happens when you use ChatGPT or a deployed model — you never see a gradient; the gradient is only for *learning*.

> **The "aha" that grounds everything:** you already know every operation in this loop. Matmul, softmax, MSE, cross-entropy, chain rule, gradient descent — all from your math docs. The loop is the *only* new thing, and it's the same everywhere.

---

## 4. The loop with real library calls (memorize this mapping)

```python
model(X)              # FORWARD  -> X @ W + b (matmul), softmax at the end
loss_fn(pred, y)      # LOSS     -> MSE or cross-entropy (one number)
loss.backward()       # BACKWARD -> chain rule: fills model's .grad with derivatives
optimizer.step()      # UPDATE   -> gradient descent: w -= lr * grad
optimizer.zero_grad() # RESET    -> wipe last step's blame before recomputing
```

Five lines. Every framework (PyTorch, TensorFlow, JAX) is these five lines in a loop. Everything you learned in `maths/`, `math-for-kids/`, and `implementations/` explains what these five lines do internally.

---

## 5. The two questions you must be able to answer about ANY model

1. **"What is the forward pass doing mathematically?"** → name the matmuls and activations.
2. **"What is the loss measuring?"** → name the loss (MSE? cross-entropy?) and why it fits the problem.

If you can answer these two for a model, you can read any training code. If you can't, re-read this page — the gap you felt is exactly the loop.

**Next:** `01-house-price.md` — the same loop in real code (sklearn + PyTorch + NumPy side by side), with every line tagged to the math.