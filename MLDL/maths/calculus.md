---
title: Calculus for Machine Learning
description: Complete beginner-friendly calculus for ML — prerequisites, limits, derivatives, gradients, optimization, integrals, with worked numeric examples and visualizations.
tags: [math, calculus, ml, derivatives, gradient, optimization, integrals]
---

# CALCULUS FOR MACHINE LEARNING

> This document is **fully self-contained**. You do not need to search the internet, open another textbook, or guess anything. Every symbol is defined, every formula is derived step by step, every example shows the full arithmetic, and every concept has a picture. Read it top to bottom.

---

# Part 0: PREREQUISITES — read this first, nothing is skipped

Before we can talk about derivatives, gradients, and loss curves, you need four tiny building blocks. They are reviewed below **in full** — if you already know them, skim quickly; if you do not, the refresher is complete enough to stand on.

---

## 0.1 What is a function? (You must know this before everything)

**Definition (plain words):** A **function** is a machine with one input slot and one output slot. You feed it a number $x$, it does a fixed rule, and it produces exactly one output number $y$. We write $y = f(x)$, read as *"y equals f of x"*, where:

- $x$ = the **input** (called the *independent variable* or the *argument*)
- $f$ = the **rule** (the fixed recipe the machine follows)
- $y$ = the **output** (called the *dependent variable*, because its value *depends* on $x$)

**Example 1 (concrete):** Let $f(x) = x^2$ (read: "f of x equals x squared").

| Input $x$ | The rule: square it | Output $y = f(x)$ |
| :--- | :--- | :--- |
| $2$ | $2 \times 2$ | $4$ |
| $-3$ | $(-3) \times (-3)$ | $9$ |
| $0.5$ | $0.5 \times 0.5$ | $0.25$ |

**Key rule that must never be broken:** one input → **exactly one** output. If a machine sometimes gives two outputs for the same input, it is *not* a function.

**Example 2 (why functions matter in ML):** A house-price predictor is a function. Input: features $x$ = (size in sq ft, number of bedrooms, age in years). Rule: the model's equation. Output: predicted price $y$. In ML the "rule" is not given to us — the model **learns** it from data, and calculus is the tool that lets it learn.

---

## 0.2 How to draw a function (plotting)

Every function can be drawn as a picture. The picture lets your eyes do the math.

**Procedure (memorize these 3 steps):**
1. Pick some input values $x$ (e.g. $x = -2, -1, 0, 1, 2$).
2. Compute the output $y = f(x)$ for each one, making an **ordered pair** $(x, y)$.
3. Put a dot at each pair on graph paper, where the horizontal axis (called the $x$-axis) measures inputs and the vertical axis (the $y$-axis) measures outputs. Connect the dots smoothly.

**Example:** Plot $f(x) = x^2$.

| $x$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $y = x^2$ | $4$ | $1$ | $0$ | $1$ | $4$ |
| point | $(-2, 4)$ | $(-1, 1)$ | $(0, 0)$ | $(1, 1)$ | $(2, 4)$ |

If you plot these five points and connect them, you get the famous **U-shape** (a *parabola*). You will see this exact picture repeatedly in this document, because the parabola is the simplest possible curve with a minimum — and finding minima is the whole point of ML.

---

## 0.3 The equation of a straight line (the single most important refresher)

A straight line has the equation

$$y = mx + b$$

- $m$ = **slope** = how steeply the line rises (or falls) when you walk one unit to the right
- $b$ = **y-intercept** = the height where the line crosses the $y$-axis (i.e. the value of $y$ when $x = 0$)

**How to compute the slope between two points:** If you know two points on the line, $(x_1, y_1)$ and $(x_2, y_2)$, then

$$\text{slope } m = \frac{\text{rise}}{\text{run}} = \frac{y_2 - y_1}{x_2 - x_1}$$

The word *rise* = vertical change (up is positive, down is negative). The word *run* = horizontal change (always measured left → right).

**Worked numeric example (every step shown):**

Take the line $y = 2x + 1$. Its slope is $m = 2$ and its intercept is $b = 1$.

1. Pick point 1: let $x_1 = 0.5$. Then $y_1 = 2(0.5) + 1 = 1 + 1 = 2$. So point 1 = $(0.5, 2)$.
2. Pick point 2: let $x_2 = 3.0$. Then $y_2 = 2(3) + 1 = 6 + 1 = 7$. So point 2 = $(3, 7)$.
3. Rise $= y_2 - y_1 = 7 - 2 = 5$.
4. Run $= x_2 - x_1 = 3.0 - 0.5 = 2.5$.
5. Slope $= \frac{5}{2.5} = 2$. ✓ (Matches $m=2$, exactly as the equation promised.)

![Slope = rise over run](/maths-images/calc-slope-line.png)

**Interpretation to carry with you forever:** a slope of $2$ means *"for every 1 unit you move right, the line moves up 2 units."* A slope of $-3$ means *"for every 1 unit right, the line moves down 3 units."* A slope of $0$ means *"perfectly flat."*

**Why this matters in ML:** a loss curve (error vs. parameter value) is *almost* a straight line if you zoom in close enough. The slope of that near-straight piece tells the model whether it should increase or decrease the parameter. That single idea — *"the slope tells you which way to go"* — is the engine of all machine learning.

---

## 0.4 Exponent rules (needed for the power rule)

Powers (also called *exponents*) appear everywhere in ML math. You must know these three rules cold:

1. **Product of powers:** $x^a \cdot x^b = x^{a+b}$ — *when multiplying, add the exponents.* (Check: $x^2 \cdot x^3 = (x \cdot x)(x \cdot x \cdot x) = x^5$.)
2. **Power of a power:** $(x^a)^b = x^{a \cdot b}$ — *when raising a power to a power, multiply the exponents.* (Check: $(x^2)^3 = x^2 \cdot x^2 \cdot x^2 = x^6$.)
3. **Negative exponents:** $x^{-a} = \frac{1}{x^a}$ — *a negative exponent means "put it in the denominator."* (Check: $x^{-2} = \frac{1}{x^2}$.)
4. **Zero exponent:** $x^0 = 1$ for any $x \neq 0$ — *anything to the zero power is 1.*
5. **Fractional exponents:** $x^{1/2} = \sqrt{x}$ — *the exponent 1/2 is the square root* (useful for L2 norms later).

**Worked example combining the rules:** simplify $\frac{x^3 \cdot x^4}{x^2}$:

- Step 1 (multiply, rule 1): $x^3 \cdot x^4 = x^{3+4} = x^7$.
- Step 2 (divide, rule 1 backwards): $\frac{x^7}{x^2} = x^{7-2} = x^5$.
- Answer: $x^5$.

---

## 0.5 Logarithms (needed for loss functions and information theory)

The **logarithm** answers the question *"what exponent do I need?"*

$$\log_b(a) = c \quad \iff \quad b^c = a$$

Read as: *"log base b of a equals the exponent c that turns b into a."*

**The two logarithms you will actually meet in ML:**

1. **Natural log** $\ln(x)$ = $\log_e(x)$ where $e \approx 2.71828$ (Euler's number). Used in loss functions, sigmoid, softmax.
2. **Log base 2** $\log_2(x)$ — used in entropy/cross-entropy (bits).

**The two rules you must know:**

1. $\ln(a \cdot b) = \ln(a) + \ln(b)$ — *log of a product = sum of logs.* (This is why logs turn multiplication into addition, which is why MLE turns products of probabilities into sums — see the Statistics document.)
2. $\ln(a^b) = b \cdot \ln(a)$ — *exponents come out front.*

**Worked example:** simplify $\ln(x^2 \cdot y^3)$:
- Step 1 (rule 1): $\ln(x^2) + \ln(y^3)$.
- Step 2 (rule 2): $2\ln(x) + 3\ln(y)$. Done.

**Easy anchor values:** $\ln(1) = 0$ (because $e^0 = 1$), $\ln(e) = 1$ (because $e^1 = e$).

---

## 0.6 Notation table — every symbol used in this document

| Symbol | Name | Meaning | Example |
| :--- | :--- | :--- | :--- |
| $f(x)$ | function | rule that maps input $x$ to output | $f(x) = x^2$ |
| $\Delta x$ | delta x | a *finite* (non-zero) change in $x$ | from $x=2$ to $x=3$, $\Delta x = 1$ |
| $dx$ | dee x | an *infinitesimally tiny* change in $x$ | the "limit" version of $\Delta x$ |
| $\lim_{h \to 0}$ | limit as h goes to 0 | what the expression approaches as $h$ gets closer to 0 | $\lim_{h \to 0} (x + h) = x$ |
| $f'(x)$ | f prime | first derivative of $f$ at $x$ (slope) | if $f(x)=x^2$, $f'(x)=2x$ |
| $f''(x)$ | f double-prime | second derivative (curvature) | if $f(x)=x^2$, $f''(x)=2$ |
| $\frac{dy}{dx}$ | derivative | same as $f'(x)$, Leibniz notation | $\frac{d}{dx}x^2 = 2x$ |
| $\frac{\partial f}{\partial x}$ | partial derivative | derivative w.r.t. $x$ only, others frozen | $\frac{\partial}{\partial x}(3x^2y) = 6xy$ |
| $\nabla f$ | nabla f / gradient | vector of all partial derivatives | $\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$ |
| $\alpha$ | alpha / learning rate | step size in gradient descent | $\alpha = 0.1$ |
| $\mathbf{w}$ | bold w | weight vector (the model's dials) | $\mathbf{w} = [0.5, -1.2]$ |
| $L(\mathbf{w})$ | loss | total error of the model | $L = 3.7$ |
| $\int_a^b f(x)\,dx$ | integral from a to b | total area under $f$ between $a$ and $b$ | $\int_0^2 x^2\,dx = \frac{8}{3}$ |
| $E[X]$ | expected value | average outcome | $E[\text{die}] = 3.5$ |
| $\sigma$ | sigma | standard deviation | $\sigma = 0.8$ |

**How to read $\frac{dy}{dx}$ (so you never panic):** it is *not* a fraction you can cancel; it is a *symbol* that means *"the derivative of y with respect to x"* = *"how fast y changes when x changes a tiny bit."* When you see $\frac{d}{dx}(x^2)$, read it as *"take the derivative of $x^2$ with respect to $x$."*

---

# Part 1: The Roadmap — where this document is going

```
                            CALCULUS FOR ML
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    ▼                              ▼                              ▼
[MODULE 1]                     [MODULE 2]                     [MODULE 3]
Single-Variable Basics         Multivariable Calculus         Higher-Order Derivatives
  ├── Limits                     ├── Multivariable Functions    ├── Second Derivative
  ├── The Derivative             ├── Partial Derivatives        ├── The Jacobian Matrix
  ├── Rules (power, product,     ├── The Gradient Vector        └── The Hessian Matrix
  │    quotient, chain)          └── Chain Rule / Backprop
  └── Common derivatives (eˣ,
       ln x, sigmoid)
                                   │
    ┌──────────────────────────────┘
    ▼
[MODULE 4]                     [MODULE 5]
Optimization                    Integrals & Probability
  ├── Minima / Maxima / Saddle    ├── Definite & Indefinite Integrals
  ├── Convex vs Non-Convex        ├── Area, Riemann Sums
  ├── Gradient Descent            ├── PDFs & Expected Value
  └── Learning Rate, Momentum,    └── Diffusion / Generative AI
       Adam
```

**How to use this roadmap:** Modules 1 and 2 are the *vocabulary* (derivatives, gradients). Module 3 is *extra vocabulary for advanced models*. Module 4 is where calculus actually *does* the ML (training = optimization). Module 5 connects calculus to probability and generative AI. Read in order — each module uses only what came before.

---

# Part 2: COMPREHENSIVE EXPLANATION

---

# MODULE 1: SINGLE-VARIABLE CALCULUS BASICS

---

## 1.1 Limits — the idea calculus is built on

### What is it?

A **limit** answers the question: *"as the input gets closer and closer to some value, what does the output get closer and closer to?"* We write

$$\lim_{x \to a} f(x) = L$$

and read it as *"the limit of f of x, as x approaches a, equals L."* This means: by choosing $x$ close enough to $a$ (but not necessarily equal to $a$), we can make $f(x)$ as close to $L$ as we want.

**Why do we need limits at all?** Because the derivative is defined as a limit, and the area under a curve is defined as a limit. If you do not understand limits, derivatives and integrals look like magic. With limits, they become obvious.

**Crucial fact:** the limit cares about *nearby* values, **not** about the value *at* the point. The function may be undefined exactly at $x = a$, and the limit can still exist.

### Step-by-step example with real numbers

Look at the function

$$f(x) = \frac{x^2 - 1}{x - 1}$$

**First observation:** at $x = 1$ this function is *undefined* — the denominator becomes $x - 1 = 1 - 1 = 0$, and division by zero is forbidden. So there is a **hole** at $x = 1$.

**Second observation:** let's see what happens near $x = 1$, approaching from both sides.

Approaching from the left (values smaller than 1):

| $x$ | $0.5$ | $0.9$ | $0.99$ | $0.999$ |
| :--- | :--- | :--- | :--- | :--- |
| $\frac{x^2-1}{x-1}$ | $1.5$ | $1.9$ | $1.99$ | $1.999$ |

Approaching from the right (values bigger than 1):

| $x$ | $1.5$ | $1.1$ | $1.01$ | $1.001$ |
| :--- | :--- | :--- | :--- | :--- |
| $\frac{x^2-1}{x-1}$ | $2.5$ | $2.1$ | $2.01$ | $2.001$ |

**Pattern:** from both sides, the outputs get closer and closer to $2$. Even though $f(1)$ does not exist, the limit does:

$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$$

**Algebraic confirmation (why it really is 2):** factor the numerator. $x^2 - 1 = (x - 1)(x + 1)$ (difference of squares). So:

$$\frac{x^2 - 1}{x - 1} = \frac{(x - 1)(x + 1)}{x - 1} = x + 1 \quad \text{(for } x \neq 1)$$

As $x$ approaches $1$, the expression $x + 1$ approaches $1 + 1 = 2$. ✓

![The limit of a function with a hole at x=1](/maths-images/calc-limit.png)

### The three rules of limits you will actually use

1. **Limit of a sum = sum of limits:** $\lim (f + g) = \lim f + \lim g$.
2. **Limit of a product = product of limits:** $\lim (f \cdot g) = \lim f \cdot \lim g$.
3. **Limit of a constant is the constant:** $\lim_{x \to a} c = c$.

**Example:** $\lim_{x \to 3} (x^2 + 5) = \lim x^2 + \lim 5 = (3 \cdot 3) + 5 = 9 + 5 = 14$.

### Where in ML?

Almost nowhere directly — but **every** gradient your ML framework computes internally relies on the limit definition of the derivative. You are standing on this idea every time you call `.backward()` in PyTorch.

### How this differs from things that look similar

- **Limit vs. value:** $\lim_{x\to a} f(x)$ is about *nearby behavior*; $f(a)$ is the *value at the point*. They can be different, or $f(a)$ can not exist while the limit does (as above). A function is **continuous** at $a$ exactly when they are equal: $\lim_{x\to a} f(x) = f(a)$.
- **Limit vs. derivative:** the derivative *is* a special limit (see next section) — the limit of a *slope*. Don't confuse the two: limits are the general tool, the derivative is one specific use of it.

---

## 1.2 The Derivative — the slope of a curve at a single point

### What is it?

You already know the slope of a *straight* line: rise over run. But what is the *slope* of a *curved* function? A curve has a different steepness at every point. The **derivative** answers: *"what is the slope of the curve at exactly one specific point?"*

**Definition (the one formula to memorize):**

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Read it in words, piece by piece:

- $f(x + h)$ = the output at a slightly advanced input ($x$ moved forward by a tiny amount $h$)
- $f(x + h) - f(x)$ = the **rise** (how much the output changed)
- $\div h$ = divide by the **run** (the tiny input change) → this is a *slope*
- $\lim_{h \to 0}$ = shrink the run to nothing → the slope *at the exact point* (not over an interval)

**Why the "limit" part is essential:** with any *finite* $h$, the fraction $\frac{f(x+h)-f(x)}{h}$ is the slope of a **secant line** (a line through two points on the curve). As $h \to 0$, the two points merge into one, and the secant becomes the **tangent line** (the line that just touches the curve at that one point, matching its steepness exactly).

![Secant lines approaching the tangent line](/maths-images/calc-secant-tangent.png)

### Full numeric derivation: the derivative of $f(x) = x^2$ at $x = 3$

**Step 1 — write the definition:**

$$f'(3) = \lim_{h \to 0} \frac{(3 + h)^2 - 3^2}{h}$$

**Step 2 — expand the square.** $(3 + h)^2 = (3 + h)(3 + h) = 9 + 3h + 3h + h^2 = 9 + 6h + h^2$. And $3^2 = 9$. So:

$$f'(3) = \lim_{h \to 0} \frac{9 + 6h + h^2 - 9}{h} = \lim_{h \to 0} \frac{6h + h^2}{h}$$

**Step 3 — factor $h$ out of the numerator:** $\frac{6h + h^2}{h} = \frac{h(6 + h)}{h} = 6 + h$ (for $h \neq 0$).

**Step 4 — take the limit:** as $h \to 0$, the term $6 + h$ approaches $6 + 0 = 6$.

**Answer:** $f'(3) = 6$.

**What does $6$ mean?** At $x = 3$, if you move forward a tiny bit, the output grows **6 times as fast** as the input. Equivalently, the tangent line at $x = 3$ has slope $6$.

### Sanity-check the same answer with actual numbers (a table, not algebra)

Compute the *secant slope* $\frac{f(x+h) - f(x)}{h}$ at $x = 3$ with smaller and smaller $h$:

| $h$ | $x + h$ | $f(3+h) = (3+h)^2$ | rise $= f(3+h) - 9$ | slope $= \text{rise}/h$ |
| :--- | :--- | :--- | :--- | :--- |
| $1$ | $4$ | $16$ | $7$ | $7$ |
| $0.5$ | $3.5$ | $12.25$ | $3.25$ | $6.5$ |
| $0.1$ | $3.1$ | $9.61$ | $0.61$ | $6.1$ |
| $0.01$ | $3.01$ | $9.0601$ | $0.0601$ | $6.01$ |
| $0.001$ | $3.001$ | $9.006001$ | $0.006001$ | $6.001$ |

The slopes are $7, 6.5, 6.1, 6.01, 6.001$ — marching toward $6$. The limit is exactly $6$. This is the same conclusion from the formula, seen with plain arithmetic. **This table is the derivative.** Nothing mysterious.

### The general pattern (the power rule, sneak preview)

Doing the same algebra with a general power $n$ gives the **power rule**:

$$\frac{d}{dx}\left(x^n\right) = n \cdot x^{n-1}$$

For $f(x) = x^2$: $n = 2$, so $f'(x) = 2 \cdot x^{2-1} = 2x$. At $x = 3$: $2 \cdot 3 = 6$. ✓ matches our full derivation.

![Tangent line on f(x)=x²](/maths-images/calc-tangent.png)

### The two interpretations you must hold simultaneously

| Interpretation | Meaning | Used for |
| :--- | :--- | :--- |
| **Slope** | steepness of the tangent at a point | understanding curves |
| **Rate of change** | how fast output changes per tiny input change | loss functions, physics analogies (speed = derivative of distance) |

### Where, why, how in ML — the "sign of the slope" rule

- **Where:** every loss function $L(w)$ of a single parameter $w$.
- **Why:** the model must know whether turning the dial $w$ up *increases* or *decreases* the error.
- **How — the golden rule of training:**
  - If $f'(x) > 0$ (slope positive): output rises as input rises → to *reduce* the output, move the input **left** (decrease it).
  - If $f'(x) < 0$ (slope negative): output falls as input rises → to *reduce* the output, move the input **right** (increase it).
  - If $f'(x) = 0$: flat spot — you are at a minimum, maximum, or saddle (Module 4).

### How the derivative differs from things that look similar

- **Derivative vs. slope of a line:** a line has one slope for its entire length; a curve has a *different* derivative at *every* point. The derivative is a *function* $f'(x)$, not a single number.
- **Derivative vs. secant slope:** secant slope = average rate of change *over an interval*; derivative = instantaneous rate of change *at a point*.
- **Derivative vs. gradient:** gradient (Module 2) is the *same* idea packaged for *many* inputs — a vector of derivatives, one per input.

---

## 1.3 The Rules of Differentiation — shortcuts so you never use the limit again

The limit definition works but is slow. These rules let you differentiate in one line. **Every rule below is followed by a complete worked example showing every arithmetic step.**

### Rule 1 — The Constant Rule

$$\frac{d}{dx}(c) = 0$$

*A constant never changes, so its rate of change is zero.*

**Example:** $f(x) = 7 \Rightarrow f'(x) = 0$. (No matter what $x$ does, 7 stays 7.)

### Rule 2 — The Constant Multiple Rule

$$\frac{d}{dx}(c \cdot f(x)) = c \cdot f'(x)$$

*A constant multiplier just rides along.*

**Example:** $f(x) = 5x^3$. Step 1: differentiate $x^3$ → $3x^2$. Step 2: multiply by 5 → $f'(x) = 5 \cdot 3x^2 = 15x^2$.

### Rule 3 — The Sum Rule

$$\frac{d}{dx}(f(x) + g(x)) = f'(x) + g'(x)$$

*Differentiate each piece separately and add.*

**Example:** $f(x) = x^2 + 3x$. Step 1: $f'(x^2) = 2x$. Step 2: $f'(3x) = 3$. Step 3: add → $f'(x) = 2x + 3$.

### Rule 4 — The Power Rule

$$\frac{d}{dx}\left(x^n\right) = n \cdot x^{n-1}$$

*Bring the exponent down front, then reduce the exponent by one.*

**Example:** $f(x) = x^4 \Rightarrow f'(x) = 4x^{4-1} = 4x^3$.

**The power rule also works for negative and fractional exponents** (exactly the same formula!):

- $f(x) = \frac{1}{x} = x^{-1} \Rightarrow f'(x) = -1 \cdot x^{-1-1} = -x^{-2} = -\frac{1}{x^2}$
- $f(x) = \sqrt{x} = x^{1/2} \Rightarrow f'(x) = \frac{1}{2} x^{1/2 - 1} = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$

### Rule 5 — The Product Rule

$$\frac{d}{dx}(f(x) \cdot g(x)) = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

*"First times derivative of second, plus second times derivative of first"* — or memorize the jingle: *"left d-right plus right d-left."*

**Why it is NOT simply $f' \cdot g'$:** because a product changes from *both* factors changing. If two things each grow, the product grows by *two* contributions. The rule is the only formula that captures both.

**Worked example:** $h(x) = x^2 \cdot e^x$. (We use $e^x$ here; its derivative rule is in section 1.4.)

- Step 1: label $f = x^2$, $g = e^x$.
- Step 2: $f' = 2x$, $g' = e^x$.
- Step 3: apply the formula: $h' = f' \cdot g + f \cdot g' = (2x)(e^x) + (x^2)(e^x)$.
- Step 4: tidy up: $h' = e^x(2x + x^2)$.

**Verification with numbers (why the rule must look like this):** take tiny step $h = 0.01$ from $x = 1$. True change: $f(1.01)g(1.01) - f(1)g(1) = (1.0201)(2.7456) - (1)(2.7183) = 2.8008 - 2.7183 = 0.0825$. Predicted by rule: $h' \cdot h = e(2 + 1) \cdot 0.01 = 2.7183 \cdot 3 \cdot 0.01 = 0.0815$. Close (the tiny difference is the $h^2$ term the limit removes). ✓

### Rule 6 — The Quotient Rule

$$\frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{g(x)^2}$$

*"Bottom d-top minus top d-bottom, all over bottom squared."*

**Worked example:** $h(x) = \frac{x^2}{x + 1}$.

- Step 1: label $f = x^2$ (top), $g = x + 1$ (bottom).
- Step 2: $f' = 2x$, $g' = 1$.
- Step 3: numerator $= (2x)(x + 1) - (x^2)(1) = 2x^2 + 2x - x^2 = x^2 + 2x$.
- Step 4: denominator $= (x + 1)^2$.
- Answer: $h'(x) = \frac{x^2 + 2x}{(x + 1)^2}$.

### Rule 7 — The Chain Rule (single variable) — the most important rule in deep learning

$$\frac{d}{dx}\left(f(g(x))\right) = f'(g(x)) \cdot g'(x)$$

**Read it in words:** *"derivative of the outer function (evaluated at the inner function), times derivative of the inner function."* Or the mnemonic: **"derivative of the outside, times derivative of the inside."**

**Why it matters more than every other rule combined:** a neural network is a *stack of functions inside functions*: output = layer$_3$(layer$_2$(layer$_1$(input))). The chain rule is the only rule that can differentiate a stack — and backpropagation *is* the chain rule applied to networks (Module 2).

**Worked example:** $h(x) = (3x^2 + 2)^5$.

- Step 1: identify the pieces. Outer function: $f(\square) = \square^5$. Inner function: $g(x) = 3x^2 + 2$.
- Step 2: derivative of the *outer* function: $f'(\square) = 5\square^4$.
- Step 3: evaluate the outer derivative *at the inner function*: $f'(g(x)) = 5(3x^2 + 2)^4$.
- Step 4: derivative of the *inner* function: $g'(x) = 6x$.
- Step 5: multiply: $h'(x) = 5(3x^2 + 2)^4 \cdot 6x = 30x(3x^2 + 2)^4$.

**Real-world intuition (the bicycle):** pedals turn the gear, the gear turns the wheel, the wheel moves the bike. If the wheel turns 2× faster than the pedals and the bike moves 3× faster than the wheel, then the bike moves $2 \times 3 = 6\times$ faster than the pedals. **Rates of change multiply along a chain.** That is the chain rule.

### The complete rules table (your one-page reference)

| Rule | Formula | Quick example |
| :--- | :--- | :--- |
| Constant | $\frac{d}{dx}c = 0$ | $\frac{d}{dx}9 = 0$ |
| Constant multiple | $\frac{d}{dx}c\,f = c\,f'$ | $\frac{d}{dx}4x = 4$ |
| Sum | $\frac{d}{dx}(f+g) = f'+g'$ | $\frac{d}{dx}(x^2+x) = 2x+1$ |
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ | $\frac{d}{dx}x^3 = 3x^2$ |
| Product | $\frac{d}{dx}(fg) = f'g + fg'$ | $\frac{d}{dx}(x^2e^x) = e^x(x^2+2x)$ |
| Quotient | $\frac{d}{dx}\frac{f}{g} = \frac{f'g - fg'}{g^2}$ | $\frac{d}{dx}\frac{x^2}{x+1} = \frac{x^2+2x}{(x+1)^2}$ |
| Chain | $\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x)$ | $\frac{d}{dx}(3x^2+2)^5 = 30x(3x^2+2)^4$ |

### Where, why, how in ML

- **Where:** automatic differentiation engines (PyTorch's autograd, TensorFlow's GradientTape).
- **Why:** a model with 100 layers is 100 nested functions. Computing a derivative through the whole stack without the chain rule would be impossible.
- **How:** frameworks store every elementary operation of the forward pass, then mechanically apply the chain rule backward — the product of local derivatives along each path.

### How the rules differ from each other (so you never pick the wrong one)

- **Sum rule** → the pieces are *added* (separate rooms in the machine).
- **Product rule** → the pieces are *multiplied* (both affect the result jointly → two terms).
- **Chain rule** → one function is *inside* another (assembly line → multiply rates).
- **Quotient rule** → one function *divided* by another (product rule's cousin with a minus sign and a squared denominator).

---

## 1.4 Derivatives of the special functions you meet in ML

These five appear constantly. Each is given with its rule, a numeric example, and *why* it appears in ML.

### $e^x$ — the self-replicating function

$$\frac{d}{dx} e^x = e^x$$

**The most beautiful fact in calculus:** $e^x$ is its own derivative — its slope at any point equals its height at that point. Nothing else in math behaves this way, and it is exactly why $e$ is the natural base for growth processes.

**Example:** $f(x) = 3e^x \Rightarrow f'(x) = 3e^x$ (constant multiple rule).

**In ML:** the softmax function (probability chapter) and the Gaussian distribution both contain $e^x$. Their derivatives keep the same exponential shape, which keeps gradients stable.

### $\ln x$ — the logarithm

$$\frac{d}{dx} \ln(x) = \frac{1}{x}$$

**Example:** $f(x) = \ln(2x)$. Chain rule with inner $g(x) = 2x$, $g' = 2$: $f'(x) = \frac{1}{2x} \cdot 2 = \frac{1}{x}$.

**In ML:** cross-entropy loss is $-\ln(\text{probability})$. Its derivative $-\frac{1}{p}$ is what makes gradient updates work — and you will derive it fully in the Probability document.

### $\sin x$ and $\cos x$ — the waves

$$\frac{d}{dx}\sin(x) = \cos(x), \qquad \frac{d}{dx}\cos(x) = -\sin(x)$$

*They swap: sine becomes cosine, cosine becomes negative sine.*

**In ML:** sine and cosine appear in transformer *positional encodings* (how GPT knows word order). Their cyclic derivatives also show up in any periodic modeling.

### The sigmoid $\sigma(x)$ — the most important function in neural networks, fully derived

The **sigmoid** squashes any real number into $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

It is the classic neuron activation and the source of the logistic-regression probability. **Its derivative is the jewel:**

$$\sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big)$$

**Full derivation (every step, nothing skipped):**

- Step 1: rewrite with a negative exponent: $\sigma(x) = (1 + e^{-x})^{-1}$.
- Step 2: chain rule. Outer: $\square^{-1}$, derivative $-\square^{-2}$. Inner: $1 + e^{-x}$, derivative $-e^{-x}$.
- Step 3: multiply: $\sigma'(x) = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$.
- Step 4: **the clever rewrite.** Multiply top and bottom by $e^{2x}$? No — instead note $e^{-x} = (1 + e^{-x}) - 1$. So:
  $$\sigma'(x) = \frac{(1 + e^{-x}) - 1}{(1 + e^{-x})^2} = \frac{1 + e^{-x}}{(1 + e^{-x})^2} - \frac{1}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} - \frac{1}{(1 + e^{-x})^2}$$
- Step 5: recognize $\frac{1}{1+e^{-x}} = \sigma(x)$ and $\frac{1}{(1+e^{-x})^2} = \sigma(x)^2$. Therefore:
  $$\sigma'(x) = \sigma(x) - \sigma(x)^2 = \sigma(x)\big(1 - \sigma(x)\big) \quad \blacksquare$$

**Numeric check:** at $x = 0$: $\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{1+1} = 0.5$. Derivative $= 0.5 \times (1 - 0.5) = 0.5 \times 0.5 = 0.25$.

**Why this formula is so beloved:** it lets us compute the derivative of the activation *using only the activation's own output*. During backprop, the network already knows $\sigma(x)$; the gradient is one multiplication away.

### Where, why, how in ML (the special five together)

| Function | Derivative | First place you meet it in ML |
| :--- | :--- | :--- |
| $e^x$ | $e^x$ | softmax, Gaussian, exponential distributions |
| $\ln x$ | $\frac{1}{x}$ | cross-entropy loss, MLE |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ | positional encodings in transformers |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | neurons, logistic regression, binary classifiers |

---

# MODULE 2: MULTIVARIABLE CALCULUS

---

## 2.1 Multivariable Functions — functions with many inputs

### What is it?

Everything so far had **one** input $x$. Real ML models have **billions** of inputs (the weights). A **multivariable function** takes several inputs and produces one output:

$$z = f(x_1, x_2, x_3, \dots, x_n)$$

- Each $x_i$ is one input (one model weight).
- $z$ is the single scalar output (the total loss).

**Example (concrete):** house-price prediction with 3 features:

$$z = f(\text{size}, \text{bedrooms}, \text{age}) = 50 \cdot \text{size} + 30000 \cdot \text{bedrooms} - 2000 \cdot \text{age}$$

If size $= 1500$, bedrooms $= 3$, age $= 10$:
$$z = 50(1500) + 30000(3) - 2000(10) = 75000 + 90000 - 20000 = 145000$$

**The mental picture:** with 2 inputs, the graph is not a curve but a **surface** (a landscape with hills and valleys — a 3D "terrain"). With $n$ inputs you cannot draw it, but the math is the same idea extended.

**How this differs from a single-variable function:** one input → the picture is a curve and the "slope" is one number. Many inputs → the picture is a landscape and the "slope" is many numbers at once (the gradient, section 2.3).

### Where, why, how in ML

- **Where:** the loss function of *any* model with more than one weight. An LLM's loss is $f(w_1, w_2, \dots, w_{100\text{ billion}})$.
- **Why:** we must decide how to adjust *each* weight separately — a single "slope" is useless; we need one slope *per weight*.
- **How:** that is exactly what partial derivatives (next section) give us.

---

## 2.2 Partial Derivatives — one weight at a time

### What is it?

A **partial derivative** is the derivative of a multivariable function with respect to **one** variable while **freezing all other variables as constants**. We write:

$$\frac{\partial f}{\partial x} \quad \text{(the partial derivative of } f \text{ with respect to } x)$$

The curly $\partial$ (read "del" or "partial") replaces the straight $d$ to signal *"this is only part of the story — we moved one variable while locking the others."*

**The 3-step procedure:**
1. Pick the variable you care about.
2. Treat **every other variable as if it were a number** (a constant).
3. Differentiate with the ordinary rules from Module 1.

### Worked example — every arithmetic step shown

$$f(x, y) = 3x^2 + 2y^3$$

**Part A — $\frac{\partial f}{\partial x}$ (derivative with respect to $x$ only):**
- Step 1: the term $3x^2$ contains $x$ → power rule: $\frac{\partial}{\partial x}(3x^2) = 3 \cdot 2x = 6x$.
- Step 2: the term $2y^3$ contains **no** $x$ → since $y$ is frozen, $2y^3$ is just a number → derivative is $0$.
- Step 3: add: $\frac{\partial f}{\partial x} = 6x + 0 = 6x$.

**Part B — $\frac{\partial f}{\partial y}$ (derivative with respect to $y$ only):**
- Step 1: $3x^2$ has no $y$ → constant → derivative $0$.
- Step 2: $2y^3$ → power rule: $2 \cdot 3y^2 = 6y^2$.
- Step 3: $\frac{\partial f}{\partial y} = 0 + 6y^2 = 6y^2$.

**Part C — plug in numbers.** At the point $(x, y) = (1, 1)$:
- $\frac{\partial f}{\partial x} = 6(1) = 6$ → *"if I nudge $x$ forward, $z$ rises 6× as fast (while $y$ is locked)."*
- $\frac{\partial f}{\partial y} = 6(1)^2 = 6$ → *"if I nudge $y$ forward, $z$ rises 6× as fast (while $x$ is locked)."*

![Partial derivatives as tangent lines on a 3D surface](/maths-images/calc-partial-derivative.png)

### Trickier worked example (product rule inside a partial derivative)

$$f(x, y) = 3x^2 y + 2y^3$$

- $\frac{\partial f}{\partial x}$: treat $y$ as a constant. The term $3x^2y$ is *constant $\times$ $x^2$* → derivative $= 3y \cdot 2x = 6xy$. The term $2y^3$ is fully constant → $0$. **Answer:** $\frac{\partial f}{\partial x} = 6xy$.
- $\frac{\partial f}{\partial y}$: treat $x$ as a constant. The term $3x^2y$ is *constant $\times$ $y$* → derivative $= 3x^2 \cdot 1 = 3x^2$. The term $2y^3$ → $6y^2$. **Answer:** $\frac{\partial f}{\partial y} = 3x^2 + 6y^2$.
- At $(2, 3)$: $\frac{\partial f}{\partial x} = 6 \cdot 2 \cdot 3 = 36$; $\frac{\partial f}{\partial y} = 3(2)^2 + 6(3)^2 = 12 + 54 = 66$.

### How partial derivatives differ from ordinary derivatives

- **Ordinary $\frac{df}{dx}$:** only one input exists, period.
- **Partial $\frac{\partial f}{\partial x}$:** many inputs exist; we pretend only $x$ moves. Every partial derivative is a *single slice* of the full landscape — the slope when walking *only* in the $x$ direction.
- **They differ from each other (why we need one per input):** each partial answers a different question — *"what happens if THIS weight moves?"* — and one weight's answer is irrelevant to another's.

### Where, why, how in ML

- **Where:** computing $\frac{\partial \text{Loss}}{\partial w_i}$ for *every* weight $w_i$ in the model.
- **Why:** training must update each weight based on *its own* effect on error, with all other weights held conceptually fixed during the measurement.
- **How:** frameworks compute all $n$ partials in one backward pass; they are then packaged into the gradient (next section) and used to update all weights.

---

## 2.3 The Gradient Vector — all partial derivatives packaged into one arrow

### What is it?

The **gradient** of a scalar function is the vector containing *every* partial derivative:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The symbol $\nabla$ is called **nabla**. Read $\nabla f$ as *"the gradient of f."* It is simply a list: *"all n slopes, one per input, in one package."*

### Worked example (every number shown)

$f(x, y) = x^2 + y^2$ (a bowl-shaped surface).

- $\frac{\partial f}{\partial x} = 2x$, $\frac{\partial f}{\partial y} = 2y$, so $\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$.
- At the point $(x, y) = (2, 1)$: $\nabla f = \begin{bmatrix} 2(2) \\ 2(1) \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$.
- At the point $(-1, -2)$: $\nabla f = \begin{bmatrix} -2 \\ -4 \end{bmatrix}$.

**The crucial property (memorize this sentence):** *the gradient always points in the direction of steepest ascent — the direction in which the function increases the fastest.* Its opposite, $-\nabla f$, points in the direction of steepest *descent*.

![Gradient arrows point uphill; negative gradient points downhill](/maths-images/calc-gradient-contour.png)

**Why this is true (intuition, not proof):** each component $\frac{\partial f}{\partial x_i}$ says *"how fast the function rises when you walk in the $x_i$ direction."* Assembling them into a vector and following *all* of them at once is like following the combined best direction on a foggy hillside: at every spot, feel the slope underfoot and step in the steepest uphill direction. The gradient is that "feel."

### How the gradient differs from the derivative and the partial derivatives

| Concept | Inputs | Output | Meaning |
| :--- | :--- | :--- | :--- |
| Derivative $f'$ | 1 | 1 number | slope in the only direction |
| Partial $\frac{\partial f}{\partial x_i}$ | many | 1 number | slope in ONE direction (others frozen) |
| **Gradient** $\nabla f$ | many | **a vector** | ALL slopes at once → the best combined direction |

### Where, why, how in ML

- **Where:** gradient descent (Module 4), backpropagation's final output.
- **Why:** with billions of weights, we need *one* vector that tells us the direction to move *all* of them to reduce loss fastest.
- **How:** update rule: $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha \nabla L$. Because $\nabla L$ points uphill (error up), subtracting it walks downhill (error down).

---

## 2.4 The Multivariable Chain Rule — backpropagation in one equation

### What is it?

The single-variable chain rule multiplies two rates. The **multivariable chain rule** does the same, but the chain can branch: an early variable may influence the final output through several intermediate variables, and you must add up *all* paths.

**The rule (two-variable version):** if $z = f(g_1(x, y), g_2(x, y))$, then

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial g_1}\frac{\partial g_1}{\partial x} + \frac{\partial z}{\partial g_2}\frac{\partial g_2}{\partial x}$$

**Plain words:** *"to find how $x$ affects $z$, multiply along every path from $x$ to $z$ and add the results."* Multiply along a path, add across paths.

### Why neural networks are chains (the big picture)

A tiny 2-layer network computes, for input $x$:

$$\hat{y} = f_2\big(f_1(x \cdot w_1 + b_1) \cdot w_2 + b_2\big)$$

The loss $L$ depends on $\hat{y}$, which depends on layer 2, which depends on layer 1, which depends on $w_1$. To learn, we need $\frac{\partial L}{\partial w_1}$ — the effect of the *first* weight on the *final* error. That is a product of adjacent derivatives (chain rule), and computing it backward is **backpropagation**:

$$\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial \hat{y}}}_{\text{loss step}} \cdot \underbrace{\frac{\partial \hat{y}}{\partial z_2}}_{\text{layer 2 step}} \cdot \underbrace{\frac{\partial z_2}{\partial z_1}}_{\text{between layers}} \cdot \underbrace{\frac{\partial z_1}{\partial w_1}}_{\text{layer 1 step}}$$

### Full numeric walkthrough (real numbers, every step)

Network with **no activations** (linear layers) to keep arithmetic clean — the mechanism is identical with activations:

- Layer 1: $z_1 = w_1 \cdot x$, with $x = 3$, $w_1 = 2$.
- Layer 2: $z_2 = w_2 \cdot z_1$, with $w_2 = 5$.
- Output: $\hat{y} = z_2$.
- Loss: $L = (\hat{y} - y_{\text{true}})^2$ (squared error), with true answer $y_{\text{true}} = 7$.

**Forward pass (compute the numbers):**
1. $z_1 = 2 \cdot 3 = 6$.
2. $z_2 = 5 \cdot 6 = 30$.
3. $\hat{y} = 30$.
4. $L = (30 - 7)^2 = 23^2 = 529$.

**Backward pass (chain rule, starting at the end):**
1. $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y_{\text{true}}) = 2(30 - 7) = 2 \cdot 23 = 46$.
2. $\frac{\partial \hat{y}}{\partial z_2} = 1$ (identity layer).
3. $\frac{\partial z_2}{\partial z_1} = w_2 = 5$.
4. $\frac{\partial z_1}{\partial w_1} = x = 3$.

**Multiply the chain:**
$$\frac{\partial L}{\partial w_1} = 46 \times 1 \times 5 \times 3 = 690$$

**Interpretation:** increasing $w_1$ by a tiny amount (say 0.001) increases the loss by roughly $690 \times 0.001 = 0.69$. A large positive gradient → we must *decrease* $w_1$ to reduce loss. (You can verify: $w_1 = 2.001$ → $z_1 = 6.003$ → $z_2 = 30.015$ → $L = (30.015 - 7)^2 = 23.015^2 = 529.69$ — an increase of $0.69$. ✓)

**ASCII diagram of the chain (the path we multiplied along):**

```
 x = 3 ──▶ z₁ = w₁·x = 6 ──▶ z₂ = w₂·z₁ = 30 ──▶ ŷ = 30 ──▶ L = (ŷ−7)² = 529
          ↑                                          ↑
      ∂z₁/∂w₁ = 3                          ∂L/∂ŷ = 46
          │                                          │
          └─────────── 46 × 1 × 5 × 3 = 690 ─────────┘
                     (multiply along the path backward)
```

### How the multivariable chain rule differs from the single-variable one

- **Single-variable:** one path, one product.
- **Multivariable:** possibly many paths; each path is a product, and the paths are **added**. Neural networks are usually *one* long path (no branching), so in practice you see a single product — but the "add across paths" case appears in networks with skip connections (ResNet).

### Where, why, how in ML

- **Where:** every neural network ever trained — backpropagation *is* this rule.
- **Why:** only by chaining derivatives can an early-layer weight receive a learning signal from the final loss.
- **How:** modern frameworks build a *computational graph* of the forward pass, then walk it backward multiplying local derivatives (exactly the 4 multiplications above, automated).

---

# MODULE 3: HIGHER-ORDER DERIVATIVES, JACOBIAN AND HESSIAN

---

## 3.1 Second Derivatives — the curvature of a function

### What is it?

The **second derivative** $f''(x)$ is the derivative *of the derivative*:

$$f''(x) = \frac{d}{dx}\big(f'(x)\big)$$

- First derivative $f'(x)$ = slope = *how fast the output changes*.
- Second derivative $f''(x)$ = curvature = *how fast the slope itself changes* (like acceleration in physics: velocity is $f'$, acceleration is $f''$).

**Worked example:** $f(x) = x^3 - 3x$.
- Step 1: $f'(x) = 3x^2 - 3$.
- Step 2: differentiate again: $f''(x) = 6x$.

**The two shapes you must recognize:**

| $f''(x)$ | Shape | Example | Meaning |
| :--- | :--- | :--- | :--- |
| $f'' > 0$ | **Concave up** $\cup$ (smiling bowl) | $f(x) = x^2$, $f'' = 2$ | slope is increasing → bottom is a **minimum** |
| $f'' < 0$ | **Concave down** $\cap$ (frowning hill) | $f(x) = -x^2$, $f'' = -2$ | slope is decreasing → top is a **maximum** |
| $f'' = 0$ with sign change | **Inflection point** | $f(x) = x^3$ at $x = 0$ | curve changes from smile to frown |

![Concavity: f, f' and f'' on the same axes](/maths-images/calc-concavity.png)

### The second derivative test (how to classify a flat spot)

When $f'(x) = 0$ (a critical point — see Module 4), look at $f''(x)$:
- $f''(x) > 0$ → local **minimum** (bowl bottom).
- $f''(x) < 0$ → local **maximum** (hill top).
- $f''(x) = 0$ → inconclusive (could be an inflection or a saddle).

**Worked example:** $f(x) = x^2 - 4x + 5$.
- Step 1: $f'(x) = 2x - 4$. Set $= 0$: $2x - 4 = 0 \Rightarrow x = 2$. Critical point at $x = 2$.
- Step 2: $f''(x) = 2 > 0$ everywhere.
- Step 3: conclusion — $x = 2$ is a **minimum**. (Indeed $f(2) = 4 - 8 + 5 = 1$ is the bottom of the parabola.)

### How the second derivative differs from the first

- **$f'$** tells *which way* things are moving (up or down) — direction.
- **$f''$** tells whether that movement is *speeding up or slowing down* — shape. A slope of $+2$ with $f'' > 0$ means "getting steeper"; with $f'' < 0$ it means "flattening out."

### Where, why, how in ML

- **Where:** second-order optimizers (Newton's method, L-BFGS), learning-rate tuning, diagnosing whether gradient descent is about to overshoot.
- **Why:** knowing *curvature* lets an optimizer choose smarter step sizes — a flat region (small $f''$) can be crossed with big steps, a sharp valley (large $f''$) needs small steps.
- **How:** when $f''$ is large, the function bends sharply, so a big step would leap across the minimum — optimizers shrink the step. That is the essence of adaptive methods (RMSProp, Adam).

---

## 3.2 The Jacobian Matrix — derivatives of vector-in → vector-out functions

### What is it?

Until now, functions output one number. But a neural network layer takes a **vector** of inputs and outputs a **vector** of activations. The **Jacobian** is the complete table of partial derivatives for such a function:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

**Reading the table:** the entry at row $i$, column $j$ answers *"how much does output $f_i$ change when input $x_j$ changes?"* — i.e. $\frac{\partial f_i}{\partial x_j}$.

- If the function outputs 1 number → the Jacobian is just the gradient (a vector).
- If it outputs $m$ numbers from $n$ inputs → the Jacobian is an $m \times n$ matrix.

### Worked example — every derivative computed

$$\mathbf{f}(x_1, x_2) = \begin{bmatrix} f_1 \\ f_2 \end{bmatrix} = \begin{bmatrix} x_1^2 + 3x_2 \\ x_1 \cdot x_2 \end{bmatrix}$$

**Compute the four partial derivatives:**
- $\frac{\partial f_1}{\partial x_1} = 2x_1$ (power rule on $x_1^2$; $3x_2$ is constant).
- $\frac{\partial f_1}{\partial x_2} = 3$ (derivative of $3x_2$; $x_1^2$ is constant).
- $\frac{\partial f_2}{\partial x_1} = x_2$ (derivative of $x_1 \cdot x_2$ w.r.t. $x_1$, $x_2$ frozen).
- $\frac{\partial f_2}{\partial x_2} = x_1$ (derivative w.r.t. $x_2$, $x_1$ frozen).

**Assemble:**
$$J = \begin{bmatrix} 2x_1 & 3 \\ x_2 & x_1 \end{bmatrix}$$

**Plug in a point, say $(x_1, x_2) = (2, 5)$:**
$$J = \begin{bmatrix} 2(2) & 3 \\ 5 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 3 \\ 5 & 2 \end{bmatrix}$$

**Interpretation:** if we nudge $x_1$ a little, $f_1$ responds about 4× as fast and $f_2$ about 5× as fast; if we nudge $x_2$, $f_1$ responds 3× and $f_2$ 2×. The Jacobian is the *local linear approximation* of the whole transformation — "the stretching factor" of the mapping at that point.

![Jacobian: how a grid of input vectors stretches into output vectors](/maths-images/calc-jacobian.png)

### How the Jacobian differs from the gradient

| | Gradient $\nabla f$ | Jacobian $J$ |
| :--- | :--- | :--- |
| Function output | **scalar** (one number) | **vector** (many numbers) |
| Shape | $n \times 1$ vector | $m \times n$ matrix |
| Question answered | "which way does the single output rise fastest?" | "how does EVERY output respond to EVERY input?" |

*If the function's output is a single scalar, the Jacobian and the gradient are the same thing written in matrix form.*

### Where, why, how in ML

- **Where:** GANs (generator/discriminator gradients), normalizing flows (change-of-variable in density estimation), multi-output networks, and any code that reasons about a *whole layer*'s response.
- **Why:** when both the input and output of a layer are vectors, a single number (like a partial derivative) is not enough — you need the whole table of sensitivities.
- **How:** backpropagation through a layer multiplies by that layer's Jacobian; frameworks compute these automatically.

---

## 3.3 The Hessian Matrix — curvature for many variables

### What is it?

The **Hessian** is the matrix of *all second-order* partial derivatives of a scalar function:

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

- Diagonal entries $\frac{\partial^2 f}{\partial x_i^2}$: curvature along each individual axis.
- Off-diagonal entries $\frac{\partial^2 f}{\partial x_i \partial x_j}$: **mixed partials** — how the slope in the $x_i$ direction changes as $x_j$ moves (interaction curvature).

**Important property:** for "nice" functions the mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$, so the Hessian is **symmetric**.

### Worked example — every derivative computed

$$f(x, y) = 3x^2 + 2xy + y^2$$

**Step 1 — first partials:**
- $\frac{\partial f}{\partial x} = 6x + 2y$
- $\frac{\partial f}{\partial y} = 2x + 2y$

**Step 2 — second partials (differentiate the first partials again):**
- $\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}(6x + 2y) = 6$
- $\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}(2x + 2y) = 2$
- $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial y}(6x + 2y) = 2$
- $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial x}(2x + 2y) = 2$  ← same! (symmetric ✓)

**Step 3 — assemble:**
$$H = \begin{bmatrix} 6 & 2 \\ 2 & 2 \end{bmatrix}$$

**Step 4 — use it.** At a critical point (where $\nabla f = 0$ — here $x = -0.5, y = 0.5$), the Hessian's *eigenvalues* tell us the type of the point (Module 4): both eigenvalues of $\begin{bmatrix}6&2\\2&2\end{bmatrix}$ are positive (they are $6.47$ and $1.53$), so this is a **minimum**.

### The three curvature cases (visual)

![The three Hessian cases: min, max, saddle](/maths-images/calc-hessian-surfaces.png)

| Eigenvalues of $H$ at a critical point | Shape | Type |
| :--- | :--- | :--- |
| All positive | bowl $\cup$ in every direction | **local minimum** |
| All negative | hill $\cap$ in every direction | **local maximum** |
| Mixed (+ and −) | bowl in one direction, hill in another | **saddle point** |

### How the Hessian differs from the Jacobian

- **Jacobian:** first derivatives, vector→vector functions, $m \times n$ table. Tells you *how fast* everything changes.
- **Hessian:** second derivatives, scalar functions, square symmetric matrix. Tells you *how the change itself changes* (curvature).
- **They relate like slope vs. acceleration:** Jacobian = "slope table", Hessian = "curvature table."

### Where, why, how in ML

- **Where:** second-order optimizers (Newton's method, L-BFGS), convergence analysis of gradient descent, saddle-point escape research.
- **Why:** gradient descent sees only slope; curvature information (Hessian) tells it *how large a step is safe* and whether a flat spot is a true minimum or a saddle.
- **How:** computing the full Hessian is impossible for LLMs (it would be a $10^{20}$-entry matrix), so practical methods approximate it (diagonal approximations, Hessian-vector products, Adam's per-parameter adaptivity as a cheap proxy).

---

# MODULE 4: OPTIMIZATION — WHERE CALCULUS DOES THE MACHINE LEARNING

---

## 4.1 Minima, Maxima, and Saddle Points — the destinations of training

### What is it?

Training = finding the parameter values $\mathbf{w}$ that minimize the loss $L(\mathbf{w})$. The places worth stopping are the **critical points** — where the gradient is zero:

$$\nabla L(\mathbf{w}) = \mathbf{0}$$

At a critical point the landscape is locally flat (the slope is 0 in every direction). There are three kinds:

| Type | Description | Loss analogy |
| :--- | :--- | :--- |
| **Local minimum** | lowest point of its surrounding region | a good model in this neighborhood |
| **Global minimum** | lowest point of the *entire* landscape | the best possible model |
| **Local maximum** | highest point of its region | the worst model (we avoid these) |
| **Saddle point** | flat spot that is a valley in one direction and a ridge in another | stuck training — gradient is 0 but we are NOT at a minimum |

**Critical subtlety:** at a saddle, $\nabla L = 0$ (flat), so gradient descent *stops moving* — yet the loss is not minimal. High-dimensional neural networks are full of saddles, and escaping them is one of the main reasons modern optimizers (momentum, Adam) exist.

**Worked example — find and classify all critical points of $f(x, y) = x^2 - y^2$:**

- Step 1 — gradient: $\frac{\partial f}{\partial x} = 2x$, $\frac{\partial f}{\partial y} = -2y$.
- Step 2 — set to zero: $2x = 0 \Rightarrow x = 0$; $-2y = 0 \Rightarrow y = 0$. One critical point: $(0, 0)$.
- Step 3 — Hessian: $\frac{\partial^2 f}{\partial x^2} = 2$, $\frac{\partial^2 f}{\partial y^2} = -2$, mixed $= 0$. So $H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$.
- Step 4 — eigenvalues: $+2$ and $-2$ → **mixed signs** → the point $(0,0)$ is a **saddle**. (Walking along the $x$-axis you go up; along the $y$-axis you go down.)

![The three critical point shapes in 3D](/maths-images/calc-hessian-surfaces.png)

### How the three critical points differ from each other (one table)

| | Gradient | Second derivative / Hessian | Are we done training? |
| :--- | :--- | :--- | :--- |
| Local min | $= 0$ | positive curvature | ✓ best nearby |
| Global min | $= 0$ | positive curvature | ✓ best anywhere (may be hard to confirm) |
| Local max | $= 0$ | negative curvature | ✗ worst nearby |
| Saddle | $= 0$ | mixed curvature | ✗ stuck, but not at a minimum |

### Where, why, how in ML

- **Where:** every training run, in the shape of the loss landscape; especially relevant for deep networks (non-convex, full of saddles and local minima).
- **Why:** knowing *what kind* of flat spot you hit decides whether to stop (min) or push through (saddle).
- **How:** optimizers with momentum carry "velocity" through saddles instead of stopping; that's why plain gradient descent is rarely used on deep nets.

---

## 4.2 Convex vs. Non-Convex — when is the minimum guaranteed?

### What is it?

- **Convex function:** shaped like a single clean bowl. Every tangent line lies *below* the curve. It has **exactly one** minimum, which is the global minimum — no traps.
- **Non-convex function:** wavy landscape with multiple valleys and hills. Multiple local minima, saddles, no guarantees.

![Convex bowl vs non-convex wavy function](/maths-images/calc-convex-nonconvex.png)

**Formal definition (for the curious, with plain translation):** $f$ is convex if for any two points $x_1, x_2$ and any weight $t \in [0, 1]$,

$$f(t x_1 + (1-t) x_2) \leq t\, f(x_1) + (1-t)\, f(x_2)$$

*Translation:* the function value at any point *between* two inputs never rises above the straight line connecting the two function values. No dips, no bumps — one bowl.

**How to test convexity with calculus (single variable):** if $f''(x) \geq 0$ everywhere, $f$ is convex.

**Worked example:** $f(x) = x^2$ → $f'' = 2 > 0$ → convex ✓. $f(x) = x^4 - 2x^2$ → $f'' = 12x^2 - 4$, which is negative for $|x| < \sqrt{1/3}$ → not convex ✗.

**Worked example (multivariable):** $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ (linear regression loss). Its Hessian is $2\mathbf{X}^T\mathbf{X}$, which is always positive semi-definite → convex → gradient descent *always* finds the global minimum. This is the mathematical guarantee behind linear regression.

### How convex differs from non-convex (the decision table)

| Property | Convex | Non-convex |
| :--- | :--- | :--- |
| Shape | one bowl | many hills/valleys |
| Number of minima | exactly one (global) | many local minima |
| Gradient descent outcome | guaranteed global minimum | depends on start + luck |
| Example ML models | linear regression, logistic regression, SVM, ridge/lasso | neural networks, deep learning |

### Where, why, how in ML

- **Where:** choosing optimizers and knowing whether "good enough" is guaranteed.
- **Why:** for convex problems (linear/logistic regression) you can train until convergence and trust the result; for deep networks you must rely on the empirical fact that SGD still finds *good* (not optimal) minima.
- **How:** practitioners don't abandon non-convex problems — modern deep learning works precisely because good local minima generalize well.

---

## 4.3 Gradient Descent — the algorithm that trains everything

### What is it?

**Gradient descent** is an iterative loop: compute the gradient (which way the loss rises), then step in the *opposite* direction to reduce the loss, and repeat until the loss stops decreasing.

**The update rule (memorize):**

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha \nabla L(\mathbf{w}_{\text{old}})$$

**Reading each piece:**
- $\mathbf{w}_{\text{old}}$ — current weight values (current position on the landscape)
- $\nabla L(\mathbf{w}_{\text{old}})$ — the gradient at that position (points uphill, toward higher loss)
- $\alpha$ (alpha) — the **learning rate**: how far to step (a small positive number like $0.01$)
- minus sign — because gradient points *uphill*, subtracting it walks *downhill*
- $\mathbf{w}_{\text{new}}$ — updated weights (new, lower-loss position)

### Worked example — full arithmetic of every step

Minimize $L(w) = (w - 1)^2 + 2$ (a parabola with minimum at $w = 1$, minimum loss 2). Start at $w = 3.0$, learning rate $\alpha = 0.25$.

**Step 1:**
- Gradient: $L'(w) = 2(w - 1)$. At $w = 3$: $L'(3) = 2(3-1) = 4$.
- Update: $w = 3 - 0.25 \times 4 = 3 - 1 = 2.0$. Loss at 2.0: $(1)^2 + 2 = 3$.

**Step 2:**
- Gradient at $w = 2$: $L'(2) = 2(2-1) = 2$.
- Update: $w = 2 - 0.25 \times 2 = 2 - 0.5 = 1.5$. Loss: $(0.5)^2 + 2 = 2.25$.

**Step 3:**
- Gradient at $w = 1.5$: $L'(1.5) = 2(0.5) = 1$.
- Update: $w = 1.5 - 0.25 \times 1 = 1.25$. Loss: $(0.25)^2 + 2 = 2.0625$.

**Steps 4–7 (same recipe):**

| Step | $w$ | gradient $2(w-1)$ | new $w = w - 0.25 \cdot g$ | loss |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 3.0 | 4 | — | 6 |
| 1 | 2.0 | 2 | 1.5 | 3 |
| 2 | 1.5 | 1 | 1.25 | 2.25 |
| 3 | 1.25 | 0.5 | 1.125 | 2.0625 |
| 4 | 1.125 | 0.25 | 1.0625 | 2.0156 |
| 5 | 1.0625 | 0.125 | 1.03125 | 2.0039 |
| 6 | 1.03125 | 0.0625 | 1.015625 | 2.0010 |
| 7 | 1.015625 | 0.03125 | 1.0078125 | 2.0002 |

**Observe the pattern:** the steps get smaller automatically (because the gradient shrinks near the minimum), and the loss approaches $2$ — the true minimum. Gradient descent *converges*.

![Gradient descent stepping down a 1D loss curve](/maths-images/calc-gd-1d.png)

![Gradient descent following a contour map in 2D](/maths-images/calc-gd-2d.png)

### The three variants you'll hear about (how they differ)

| Variant | Data used per step | Typical use |
| :--- | :--- | :--- |
| **Batch GD** | all training samples | tiny datasets, smooth updates |
| **Stochastic GD (SGD)** | 1 random sample | huge datasets, noisy but fast steps |
| **Mini-batch GD** | a small batch (e.g. 32) | **the standard** — compromise between the two |

They differ only in *how the gradient estimate is computed* — the update rule is identical.

### Where, why, how in ML

- **Where:** the core loop of virtually every ML and deep learning training run.
- **Why:** no closed-form solution exists for most models, so we must *walk* to the minimum numerically.
- **How:** repeat: forward pass → compute loss → backward pass (gradient) → update weights with $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L$ → repeat until loss plateaus.

---

## 4.4 Learning Rate, Momentum, and Adam — making gradient descent fast and stable

### What is it? (Learning rate $\alpha$)

The **learning rate** $\alpha$ is the step size — how far we move in the gradient's direction each iteration. It is the single most important hyperparameter in ML.

**The three regimes (each with a picture):**

![Learning rates: too small, just right, too large](/maths-images/calc-learning-rate.png)

| $\alpha$ | What happens | Analogy |
| :--- | :--- | :--- |
| **Too small** (e.g. 0.001) | steps are tiny; training takes forever, may stall | shuffling toward a destination |
| **Just right** (e.g. 0.1) | converges quickly and stably | brisk walking |
| **Too large** (e.g. 1.5) | overshoots the minimum, loss *increases*, diverges | running past the door forever |

**Worked demonstration (overshoot):** on $L(w) = (w-1)^2+2$ with $\alpha = 1.5$ starting at $w = 3$: gradient $= 4$ → new $w = 3 - 1.5(4) = 3 - 6 = -3$ → loss jumps from 6 to $(-4)^2 + 2 = 18$ — worse! Next step: gradient $= 2(-4) = -8$ → $w = -3 - 1.5(-8) = -3 + 12 = 9$ → loss = $8^2 + 2 = 66$ — exploding. This is **divergence**.

### What is it? (Momentum)

**Momentum** keeps a running average of past gradients (a "velocity") and moves with it:

$$\mathbf{v} \leftarrow \beta \mathbf{v} + \nabla L, \qquad \mathbf{w} \leftarrow \mathbf{w} - \alpha \mathbf{v}$$

- $\beta$ (typically $0.9$) = how much past velocity is remembered.
- Effect: oscillations cancel out (zigzags smooth), and flat saddle regions are *glided across* instead of stopping.

![Plain GD zigzags; momentum smooths the path](/maths-images/calc-momentum.png)

**Why it beats plain GD on real problems:** loss landscapes are often *elongated valleys* (steep in one direction, flat in another). Plain GD zigzags across the valley walls; momentum averages the zigzag into a smooth forward glide.

### What is it? (Adam — the industry default)

**Adam (Adaptive Moment Estimation)** gives *each parameter its own adaptive step size*:
1. It keeps a momentum-like running average of gradients (**first moment** $m$).
2. It also keeps a running average of *squared* gradients (**second moment** $v$) — a proxy for *how much that parameter has been moving*.
3. The step size for parameter $i$ is scaled by $\frac{m_i}{\sqrt{v_i} + \epsilon}$ — parameters with large fluctuating gradients get small steps; calm parameters get bigger steps.

**Intuition:** Adam is "momentum with a personalized speed limit per weight."

### How the three optimizers differ (the decision table)

| Optimizer | Step size | Adaptivity | Behavior |
| :--- | :--- | :--- | :--- |
| Plain GD / SGD | one global $\alpha$ | none | simple, zigzags, stops at saddles |
| SGD + Momentum | one global $\alpha$ | none | smooths zigzag, crosses saddles |
| Adam | per-parameter | yes | fast, stable, robust; the default choice |

### Where, why, how in ML

- **Where:** transformers and LLMs are trained with Adam/AdamW; CNNs often use SGD+momentum; the choice is part of "training recipe."
- **Why:** plain GD is too slow and fragile for high-dimensional non-convex landscapes; adaptive methods need far less tuning.
- **How:** in code it's one line — `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)` — but the math above is exactly what that line executes.

---

# MODULE 5: INTEGRALS AND PROBABILITY CONNECTIONS

---

## 5.1 Integrals — the opposite of derivatives, and the area under curves

### What is it?

**Integration is differentiation's inverse.** If differentiation tells you the *rate* (speed), integration tells you the *total* (distance).

Two flavors:

| Type | Question it answers | Notation |
| :--- | :--- | :--- |
| **Indefinite integral** | "what function has this derivative?" (anti-derivative) | $\int f(x)\,dx = F(x) + C$ |
| **Definite integral** | "what is the total area under $f$ between $a$ and $b$?" | $\int_a^b f(x)\,dx = F(b) - F(a)$ |

The constant $C$ appears because many functions share a derivative (derivative of $x^2$, $x^2 + 1$, $x^2 - 7$ is all $2x$) — the indefinite integral lists the whole family.

### The Riemann sum — how area is actually computed (and why it's a limit)

To find the area under a curve exactly, we first *approximate* it with rectangles:

1. Split the interval $[a, b]$ into $n$ equal slices of width $\Delta x = \frac{b - a}{n}$.
2. On each slice, draw a rectangle whose height matches the function at some point in the slice.
3. Sum all rectangle areas.
4. Take the limit as $n \to \infty$ (rectangles become infinitely thin):

$$\int_a^b f(x)\,dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i)\,\Delta x$$

![Approximating the area with rectangles](/maths-images/calc-riemann.png)

### Worked example — compute $\int_0^2 x^2 \, dx$ with rectangles first, then exactly

**Rough estimate with $n = 4$ rectangles** (right endpoints, width $\Delta x = \frac{2-0}{4} = 0.5$):

| slice | right endpoint $x_i$ | height $x_i^2$ | area $= 0.5 \times x_i^2$ |
| :--- | :--- | :--- | :--- |
| [0, 0.5] | 0.5 | 0.25 | 0.125 |
| [0.5, 1] | 1 | 1 | 0.5 |
| [1, 1.5] | 1.5 | 2.25 | 1.125 |
| [1.5, 2] | 2 | 4 | 2 |
| **total** | | | **3.75** |

**Exact value using the power rule for integrals:** $\int x^2\,dx = \frac{x^3}{3}$, so

$$\int_0^2 x^2\,dx = \frac{2^3}{3} - \frac{0^3}{3} = \frac{8}{3} - 0 = 2.667$$

Our 4-rectangle estimate (3.75) is above; with $n = 100$ rectangles the estimate would be ≈ 2.686, with $n = 1000$ ≈ 2.669 — converging to $2.667$. The limit is exact.

**Rule for the definite integral evaluation (Fundamental Theorem of Calculus):**
1. Find an anti-derivative $F$ of $f$.
2. Evaluate $F$ at the top limit $b$.
3. Evaluate $F$ at the bottom limit $a$.
4. Subtract: $F(b) - F(a)$.

![The definite integral as shaded area between a and b](/maths-images/calc-definite-integral.png)

### A small table of common anti-derivatives (the reverse of the derivative table)

| $f(x)$ | $\int f(x)\,dx$ | Check: derivative of the answer |
| :--- | :--- | :--- |
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ | $\frac{d}{dx}\frac{x^{n+1}}{n+1} = x^n$ ✓ |
| $e^x$ | $e^x + C$ | $\frac{d}{dx} e^x = e^x$ ✓ |
| $\frac{1}{x}$ | $\ln|x| + C$ | $\frac{d}{dx}\ln x = \frac{1}{x}$ ✓ |
| $\cos x$ | $\sin x + C$ | $\frac{d}{dx}\sin x = \cos x$ ✓ |

### How integrals differ from derivatives (side by side)

| | Derivative | Integral |
| :--- | :--- | :--- |
| Answers | rate of change / slope | accumulated total / area |
| Symbol | $\frac{d}{dx}$ | $\int$ |
| Units | output per input (e.g. mph) | output × input (e.g. miles) |
| ML role | training (gradients) | probability (areas), expectation |

### Where, why, how in ML

- **Where:** continuous probability (Module 5.2), Bayesian inference, generative models (5.3).
- **Why:** probabilities of continuous outcomes are *areas under curves*, and areas are integrals. Expectations of continuous variables are also integrals.
- **How:** in practice most integrals are computed *numerically* (sampling/Monte Carlo — see Probability doc) because closed-form anti-derivatives rarely exist for real ML math.

---

## 5.2 Probability Density Functions (PDFs) and Expected Value — where integrals meet ML

### What is it?

A **probability density function (PDF)** $f(x)$ describes where a continuous random variable is likely to be. Two defining rules:

1. The total area under the PDF is exactly 1 (probability must sum to 100%): $\int_{-\infty}^{\infty} f(x)\,dx = 1$.
2. The probability of landing in an interval is the *area* over that interval:

$$P(a \le X \le b) = \int_a^b f(x)\,dx$$

**Critical difference from discrete probabilities:** for a continuous variable, the probability of an *exact single point* is 0 (a point has zero width → zero area). Probabilities only make sense over *ranges*. The PDF's height is *density* (probability per unit of $x$), not probability.

![Gaussian PDF with a probability interval shaded](/maths-images/calc-gaussian-pdf.png)

### The expected value (average outcome) — an integral

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx$$

**Read it:** *"weight every possible value $x$ by how likely its neighborhood is (density $f(x)$), and add up everything."* The result is the center of mass of the distribution — the long-run average.

**Worked example — uniform distribution on $[0, 4]$:** density is constant $f(x) = \frac{1}{4}$ (so the total area $4 \times \frac{1}{4} = 1$ ✓). Then

$$E[X] = \int_0^4 x \cdot \frac{1}{4}\,dx = \frac{1}{4} \cdot \frac{x^2}{2}\Big|_0^4 = \frac{1}{4}\left(\frac{16}{2} - 0\right) = \frac{1}{4} \cdot 8 = 2$$

The average of numbers uniformly drawn from 0 to 4 is indeed 2 (the midpoint). ✓

### How PDFs differ from PMFs (the most common confusion in ML)

| | PMF (discrete) | PDF (continuous) |
| :--- | :--- | :--- |
| Function values | **probabilities** $P(X = x)$ | **densities** (not probabilities!) |
| Values add/sum to | 1 | area integrates to 1 |
| $P(X = x)$ at a single point | can be > 0 | always 0 |
| Computation | sum | integral |
| Picture | bars (staircase) | smooth curve |

### Where, why, how in ML

- **Where:** Gaussian Naive Bayes, VAEs, Bayesian deep learning, anomaly detection (fit a PDF to normal data; flag points in near-zero-density regions).
- **Why:** real-world quantities (height, loss values, latent variables) are continuous — their probability math *requires* integrals.
- **How:** models compute expected values like $E[\text{loss}]$ by averaging over samples (empirical expectation — the integral replaced by a sum over the dataset), which is the bridge between calculus and statistics.

---

## 5.3 Integrals in Generative AI — diffusion models and VAEs

### What is it?

Modern generative models treat image generation as a *continuous probability process*:

- **VAEs** learn a probability distribution over hidden ("latent") variables; generating new images = sampling from that distribution and decoding.
- **Diffusion models** (Stable Diffusion, DALL·E, Midjourney) destroy an image step by step by adding random noise, then *learn to reverse* that noise mathematically. The forward process is a continuous time evolution (a stochastic differential equation), and reversing it involves integrals over the noise schedule.

![The diffusion forward process: adding more and more noise](/maths-images/calc-diffusion-noise.png)

### The forward process (what the picture shows)

1. Start with a clean image (a high-dimensional tensor of pixel values).
2. At each tiny time step, add a small amount of Gaussian noise.
3. After enough steps, the image is pure noise — mathematically, its values follow a standard Gaussian $\mathcal{N}(0, I)$ regardless of the original image.

**ASCII sketch of the pipeline:**

```
 clean image ──(add noise t=1)──▶ noisy ──(add noise t=2)──▶ ... ──▶ pure noise
      ▲                                                               │
      │                                                               ▼
      └──────(reverse: denoise with neural network)────────────────────┘
```

### Where the calculus lives

- The noise schedule is described by integrals over time (how much noise has accumulated by step $t$).
- The reverse (denoising) process is *another* probability distribution whose form comes from Bayes' theorem — a ratio of Gaussians whose normalization constants are integrals.
- In practice, the intractable integrals are approximated by sampling (Monte Carlo) — the same theme as Module 5.2: *integration is the math, sampling is the machine.*

### How diffusion differs from VAEs (so you never mix them up)

| | VAE | Diffusion |
| :--- | :--- | :--- |
| Core idea | compress to latent, sample, decode | add noise, then learn to remove it |
| Calculus used | expectation/KL integrals (see Probability) | integrals over the noise schedule |
| Generation quality | decent, fast | state-of-the-art, slower |
| Examples | classic autoencoders | Stable Diffusion, DALL·E, Midjourney |

### Where, why, how in ML

- **Where:** all modern text-to-image systems; also audio generation, and increasingly protein design.
- **Why:** generating novel, realistic content requires modeling a full probability distribution over high-dimensional data — and distributions live in integral-land.
- **How:** the neural network learns to predict the noise; the "generation" is numerically solving the reverse integral step by step. The user sees a picture; underneath is calculus.

---

# Part 3: SUMMARY CHEAT-SHEET

| Concept | Definition in one line | Primary ML application | Key formula |
| :--- | :--- | :--- | :--- |
| **Limit** | what outputs approach as inputs approach a value | foundation of every derivative | $\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$ |
| **Derivative** $f'$ | slope / rate of change at a point | loss slope → which way to move a weight | power rule: $\frac{d}{dx}x^n = nx^{n-1}$ |
| **Chain rule** | multiply rates along a chain | backpropagation in neural networks | $f'(g(x))\,g'(x)$ |
| **Sigmoid derivative** | $\sigma(1-\sigma)$ | gradient through activations | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| **Partial derivative** | slope in one direction, others frozen | per-weight sensitivity | $\frac{\partial f}{\partial w_i}$ |
| **Gradient** $\nabla f$ | vector of all partials; points uphill | direction to move weights (negated) | $\mathbf{w} \leftarrow \mathbf{w} - \alpha\nabla L$ |
| **Jacobian** | table of partials of a vector function | multi-output layers, flows | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ |
| **Hessian** | table of second partials (curvature) | second-order optimizers, saddles | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ |
| **Critical points** | where $\nabla f = 0$ | deciding min vs max vs saddle | classify with Hessian eigenvalues |
| **Convexity** | one bowl, one global min | guarantees for linear/logistic regression | $f'' \geq 0$ everywhere |
| **Gradient descent** | step opposite the gradient repeatedly | trains every ML model | $\mathbf{w} \leftarrow \mathbf{w} - \alpha\nabla L$ |
| **Learning rate** $\alpha$ | step size | the #1 hyperparameter | too big → diverge; too small → crawl |
| **Momentum / Adam** | velocity + per-parameter step sizes | deep network training | $v \leftarrow \beta v + \nabla L$ |
| **Integral** | area under a curve / anti-derivative | probabilities, expectations | $\int_a^b f\,dx = F(b) - F(a)$ |
| **PDF & expectation** | density & average of a continuous variable | generative models, Bayes | $E[X] = \int x\,f(x)\,dx$ |

---

# Part 4: WHAT TO READ NEXT (inside this same math folder)

- **linear-algebra.md** — the language of vectors and matrices that calculus acts on (gradients are vectors!).
- **probability.md** — where the PDFs, expectations, and integrals from Module 5 get their full treatment.
- **statistics.md** — how MLE (Maximum Likelihood) *derives* the loss functions that Module 4 optimizes.