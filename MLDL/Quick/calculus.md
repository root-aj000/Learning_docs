---
title: Calculus — Quick Revision
description: 10-minute revision of calculus for ML — all key rules, formulas, differentiation tables, and ML applications in one page.
tags: [math, calculus, quick-rev, derivative, gradient, integrals]
---

# CALCULUS — QUICK REVISION

> Condensed from the full guide at `MLDL/maths/calculus.md` (read that for step-by-step derivations and examples). This page is for **revision**: scan, recall, quiz yourself.

## THE BIG PICTURE (3 lines)

1. **Derivative** = instantaneous rate of change = slope at a point. ML uses it to find "which direction should the weights move?"
2. **Gradient** = derivative of multi-variable functions = direction of steepest increase. Gradient descent goes the *opposite* way.
3. **Integral** = accumulation = area under a curve. ML uses it for probabilities (PDFs) and expectations.

## NOTATION QUICK-REF

| Symbol | Meaning |
| :--- | :--- |
| $f'(x)$, $\frac{dy}{dx}$ | derivative of y w.r.t. x |
| $\frac{\partial f}{\partial x}$ | partial derivative (one variable, others frozen) |
| $\nabla f$ | gradient vector of all partials |
| $J$ | Jacobian matrix (first derivatives) |
| $H$ | Hessian matrix (second derivatives) |
| $\int_a^b f(x)\,dx$ | integral from a to b |
| $\eta$ (eta) | learning rate |

## ALL DIFFERENTIATION RULES (memorize)

| Rule | Formula | Example |
| :--- | :--- | :--- |
| Power | $\frac{d}{dx}x^n = n x^{n-1}$ | $\frac{d}{dx}x^3 = 3x^2$ |
| Constant | $\frac{d}{dx}c = 0$ | — |
| Constant multiple | $\frac{d}{dx}[c f] = c f'$ | $\frac{d}{dx}5x^2 = 10x$ |
| Sum | $(f + g)' = f' + g'$ | — |
| Product | $(fg)' = f'g + fg'$ | $(x^2 e^x)' = 2xe^x + x^2 e^x$ |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ | — |
| **Chain** | $\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$ | $(e^{x^2})' = e^{x^2} \cdot 2x$ |

## SPECIAL FUNCTIONS YOU MUST KNOW

| Function | Derivative | Why it matters in ML |
| :--- | :--- | :--- |
| $e^x$ | $e^x$ (itself!) | softmax, exponentials everywhere |
| $\ln x$ | $\frac{1}{x}$ | log-likelihood, cross-entropy |
| $\sin x$ / $\cos x$ | $\cos x$ / $-\sin x$ | positional encodings |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | logistic regression, output layer |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN gates |

**Sigmoid derivative derivation (1 min):** $\sigma' = \sigma - \sigma^2 = \sigma(1-\sigma)$. Max at 0 (value 0.25), → 0 at extremes: explains vanishing gradients.

## MULTIVARIABLE CALCULUS (the ML engine)

| Object | What | Shape | Example (n-dim f) |
| :--- | :--- | :--- | :--- |
| **Partial** $\frac{\partial f}{\partial x_i}$ | derivative in one direction | scalar | freeze others, differentiate |
| **Gradient** $\nabla f$ | all partials, steepest ascent | $n \times 1$ | $\nabla f = \left[\frac{\partial f}{\partial x_1}, \dots\right]^T$ |
| **Jacobian** $J$ | all partials of a *vector* function | $m \times n$ | rows = outputs, cols = inputs |
| **Hessian** $H$ | all *second* partials | $n \times n$ | curvature / convexity |

**Worked gradient (10 sec):** $f(x, y) = 3x^2 + y^3$ → $\nabla f = (6x, 3y^2)$. At $(2, 1)$: $(12, 3)$.

**Jacobian (10 sec):** $f = (x^2 + 3y, xy)$ → $J = \begin{bmatrix} 2x & 3 \\ y & x \end{bmatrix}$.

**Hessian (10 sec):** $f = x^3 + xy + y^2$ → $H = \begin{bmatrix} 6x & 1 \\ 1 & 2 \end{bmatrix}$.

**Chain rule for a 2-layer net (backprop, 30 sec):** loss $L$, hidden $h$, input $x$:
- Forward: $h = \sigma(W_1 x)$, $\hat{y} = W_2 h$, $L = \frac{1}{2}(\hat{y} - y)^2$.
- Backward: $\frac{\partial L}{\partial W_2} = (\hat{y} - y)\, h^T$; $\frac{\partial L}{\partial h} = W_2^T(\hat{y}-y)$; $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h} \circ \sigma'(W_1 x)\, x^T$.

## OPTIMIZATION (how learning happens)

**Gradient descent update:** $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$

**Critical points:** $\nabla f = 0$ → minimum / maximum / saddle. **Convex** function: any critical point = global minimum (that's why linear regression is easy).

**Learning rate $\eta$:** too large → diverge (loss explodes); too small → crawl. Good starting range: $10^{-3}$ to $10^{-1}$.

**The three optimizers (difference table):**

| | Vanilla GD | Momentum | Adam |
| :--- | :--- | :--- | :--- |
| Extra state | none | velocity $v$ | $m$ (mean) + $v$ (variance) of gradients |
| Idea | $\theta \mathrel{-}= \eta g$ | $\theta \mathrel{-}= \eta v$, $v = \beta v + g$ | per-parameter adaptive step |
| Solves | — | oscillations / local valleys | sparse + noisy gradients (default choice) |

## INTEGRALS (for probability & expectations)

**Fundamental theorem:** $\int_a^b f(x)\,dx = F(b) - F(a)$ where $F' = f$.

**Key integrals:** $\int x^n dx = \frac{x^{n+1}}{n+1}$; $\int e^x dx = e^x$; $\int \frac{1}{x} dx = \ln|x|$; $\int \cos x\, dx = \sin x$.

**ML uses:** $P(a \le X \le b) = \int_a^b f(x)\,dx$; $E[X] = \int x f(x)\,dx$; area under ROC curve.

## THE DIFFERENTIATION TABLES (zero-confusion)

**Derivative vs. Integral:** derivative = rate/slope (local, "how fast"); integral = accumulation/area (global, "how much"). They undo each other.

**Gradient vs. Jacobian:** gradient = one vector for a scalar function; Jacobian = matrix for a vector function (its rows are gradients).

**Jacobian vs. Hessian:** Jacobian = first derivatives; Hessian = second derivatives. Jacobian shapes $m \times n$; Hessian is always $n \times n$ (and symmetric for smooth $f$).

**Local vs. global minimum:** local = lowest in a neighborhood; global = lowest everywhere. Non-convex (neural nets) → only local guaranteed; convex → local = global.

**Momentum vs. Adam:** momentum adds one smoothed velocity; Adam tracks both mean AND variance per parameter → adapts step size per weight. Use Adam unless you have a reason not to.

## WHERE CALCULUS APPEARS IN ML (one-line map)

| Concept | ML location |
| :--- | :--- |
| Derivative | gradient of any loss w.r.t. weights |
| Chain rule | **backpropagation** — the entire training loop |
| Gradient | gradient descent, SGD, Adam |
| Jacobian | neural net layers, normalizing flows, pose estimation |
| Hessian | second-order optimizers, curvature, batch-norm theory |
| Convexity | why linear/logistic regression converge to global optimum |
| Integrals / PDFs | likelihood, expected value, KL divergence, diffusion math |

## TOP 5 COMMON MISTAKES

1. Product rule: forgetting the second term — $(fg)' = f'g + fg'$ (NOT $f'g'$).
2. Chain rule: forgetting the inner derivative — $(e^{x^2})' \ne e^{x^2}$.
3. Gradient descent sign: subtract (go downhill), not add.
4. Learning rate: setting it from intuition — always log-scale search.
5. Confusing gradient (vector, scalar f) with Jacobian (matrix, vector f).

> Full detail + worked examples: `MLDL/maths/calculus.md` — then `linear-algebra` (vectors), `statistics` (MLE), `probability` (distributions).