---
title: Calculus for Kids (10-year-old friendly)
description: Calculus explained like a story — machines, hills, and balls rolling down. No scary algebra. Just pictures, tables, and one idea at a time.
tags: [math, kids, calculus, easy, beginner]
---

# CALCULUS FOR KIDS

> This document is written for someone who is **10 years old** and has **never done math beyond basic numbers**. Every idea comes with a picture, a story, and real numbers you can check yourself. Read it top to bottom. If a symbol looks scary, the "say it" note tells you how to say it out loud — and that's all you need.

---

## The whole story in 10 lines (read this first!)

1. A **function** is a machine: numbers go in, numbers come out.
2. Machines can be drawn as **pictures** (graphs).
3. A straight line has a **slope** — how steep it is.
4. A curvy hill has a *different* steepness at every point.
5. The **derivative** is the steepness at one exact point.
6. Machines learn by **rolling down the hill** to the lowest point.
7. When a machine has many dials, the **gradient** is a compass that points downhill.
8. When machines are stacked inside each other, the **chain rule** multiplies their steepnesses.
9. The **integral** is the area under the hill — the "total of a curve".
10. That's it. Calculus is just steepness and area. Everything else is details.

---

# PART 0: THINGS YOU ALREADY KNOW (with a quick warm-up)

You already know most of what you need. We just give it proper names.

---

## 0.1 Numbers — the building blocks

- Numbers can be whole: $1, 2, 3, \dots$
- Numbers can be in between: $0.5, 2.25, 3.7$
- Numbers can be negative: $-2, -5.5$
- Numbers can be tiny: $0.001$

**That's all the numbers we use in this document.** No other kinds. When you see a letter like $x$ or $h$, it's just a **box** that holds a number. The letter is a shortcut for "some number, I'll tell you which one in a moment."

---

## 0.2 Functions — machines that turn numbers into other numbers

A **function** is a machine. You drop a number in the top, it does one fixed thing, and exactly one number comes out.

```
        DROP IN: x
            │
            ▼
   ┌────────────────┐
   │  THE MACHINE   │   example: "multiply by 2, then add 1"
   │  (the "rule")  │
   └────────────────┘
            │
            ▼
        COMES OUT: y
```

We write it as $y = f(x)$ and say it out loud as **"y equals f of x"**. It just means: *"the machine named f turns the number x into the number y."*

**Try it yourself with real numbers.** The machine says: *multiply by 2, add 1.*

| Number you drop in ($x$) | What the machine does | Number that comes out ($y$) |
| :--- | :--- | :--- |
| $1$ | $1 \times 2 + 1$ | $3$ |
| $2$ | $2 \times 2 + 1$ | $5$ |
| $5$ | $5 \times 2 + 1$ | $11$ |
| $0.5$ | $0.5 \times 2 + 1$ | $2$ |

**The golden rule of machines:** one input gives **exactly one** output. If a machine sometimes gives two answers for the same input, it's broken — and it's not a function.

**Why you should care:** an ML model (like a robot that guesses house prices) is one giant machine. Numbers go in (size of house, number of rooms), a number comes out (predicted price). The only difference: we *don't know* the machine's rule. The computer **learns** it. Calculus is the tool that makes learning possible.

---

## 0.3 Drawing machines as pictures (graphs)

Every machine can be drawn. Here's how:

**STEP 1:** Pick some numbers to drop in: $x = 0, 1, 2, 3$.
**STEP 2:** Compute what comes out for each.
**STEP 3:** Draw a dot for each (in, out) pair, then connect the dots.

**Example:** the machine $y = 2x + 1$ (multiply by 2, add 1):

| $x$ | $y$ | Dot |
| :--- | :--- | :--- |
| $0$ | $1$ | $(0, 1)$ |
| $1$ | $3$ | $(1, 3)$ |
| $2$ | $5$ | $(2, 5)$ |
| $3$ | $7$ | $(3, 7)$ |

```
y
7 │                              ● (3,7)
6 │                         ╱
5 │                    ● (2,5)
4 │               ╱
3 │          ● (1,3)
2 │     ╱
1 │● (0,1)
0 │────────────────────────────── x
  0     1     2     3
```

The dots all line up in a **straight line**. (Of course! The rule was "multiply by 2, add 1" — that's a straight-line rule.)

**Second example:** the machine $y = x^2$ (the square machine). Here $x^2$ means $x \times x$. "x squared".

| $x$ | $y = x \times x$ | Dot |
| :--- | :--- | :--- |
| $-2$ | $4$ | $(-2, 4)$ |
| $-1$ | $1$ | $(-1, 1)$ |
| $0$ | $0$ | $(0, 0)$ |
| $1$ | $1$ | $(1, 1)$ |
| $2$ | $4$ | $(2, 4)$ |

```
y
4 │              ●           ●
3 │            ╱   ╲
2 │         ╱         ╲
1 │      ●               ●
0 │   ●─────────────────────●──── x
  -2    -1     0      1      2
```

That's the famous **U-shape** (grown-ups call it a *parabola*). You'll see this U-shape again and again, because it's the simplest picture with a lowest point — and finding the lowest point is the whole job of machine learning.

---

## 0.4 Straight lines and slope — the most important warm-up

Every straight line can be written as

$$y = mx + b$$

Say it out loud: **"y equals m x plus b"**. It means:

- $m$ = **slope** = *how steep* the line is
- $b$ = where the line crosses the middle line (the $y$-axis) when $x = 0$

**How to find the slope of a line:** take two dots on the line and compute:

$$\text{slope} = \frac{\text{rise}}{\text{run}} = \frac{\text{how much it went up}}{\text{how much it went right}}$$

**GIVEN:** the line $y = 2x + 1$ and two dots on it: $(0.5, 2)$ and $(3, 7)$.
**STEP 1:** Rise = $7 - 2 = 5$ (it went up 5).
**STEP 2:** Run = $3 - 0.5 = 2.5$ (it went right 2.5).
**STEP 3:** Slope = $\frac{5}{2.5} = 2$.
**CHECK:** the equation said $m = 2$. ✓ Same number!
**WHAT IT MEANS:** for every 1 step right, the line climbs 2 steps up.

![Slope = rise over run](/maths-images/calc-slope-line.png)

**Slope dictionary:**
- slope $2$ → "for every 1 right, go up 2"
- slope $0.5$ → "for every 1 right, go up half"
- slope $0$ → "perfectly flat, no up at all"
- slope $-3$ → "for every 1 right, go **down** 3"

**Why this matters for ML:** when a robot is learning, it looks at a picture of errors (a "loss curve"). A curve is *almost* a straight line if you zoom in close enough. The slope of that tiny straight piece tells the robot which way to turn. **"The slope tells you which way to go" — that one sentence is the engine of all machine learning.**

> **ONE-LINE POINT of Part 0:** Functions are machines, pictures of machines are graphs, straight lines have a slope, and slope = rise ÷ run.

---

# PART 1: THE DERIVATIVE — steepness at one exact point

## 1.1 The problem: a hill has many slopes

A straight line has **one** slope. But a curvy hill has a **different** steepness at every single point. At the top of the U-shape it's flat; on the sides it's steep.

**The question calculus answers:** *What is the steepness of the hill at exactly this one point?* That's the **derivative**. Say it out loud: **"duh-riv-uh-tiv"**.

**The secret trick — zoom in.** If you zoom in super close on a curvy hill, it stops looking curvy. It looks like a straight line. Every curve is secretly made of tiny straight pieces! The slope of the tiny straight piece at a point IS the derivative at that point.

![Secant lines approaching the tangent line](/maths-images/calc-secant-tangent.png)

---

## 1.2 How to find the steepness: the shrink-and-march table

**GIVEN:** the square machine $y = x^2$ (that U-shape). We want the steepness at the point $x = 3$.
**STEP 1:** pick a "friend" point near $x = 3$. Let's use $x = 4$. Compute both outputs: at $3$, output is $3 \times 3 = 9$; at $4$, output is $4 \times 4 = 16$.
**STEP 2:** compute the slope between the two dots: rise $= 16 - 9 = 7$, run $= 4 - 3 = 1$, slope $= 7$.
**STEP 3:** now move the friend point *closer and closer* to $3$, and compute the slope each time:

| Friend point | Output of friend | Rise (output minus 9) | Run (friend minus 3) | Slope = rise ÷ run |
| :--- | :--- | :--- | :--- | :--- |
| $4$ | $16$ | $7$ | $1$ | $7$ |
| $3.5$ | $12.25$ | $3.25$ | $0.5$ | $6.5$ |
| $3.1$ | $9.61$ | $0.61$ | $0.1$ | $6.1$ |
| $3.01$ | $9.0601$ | $0.0601$ | $0.01$ | $6.01$ |
| $3.001$ | $9.006001$ | $0.006001$ | $0.001$ | $6.001$ |

**STEP 4:** look at the slopes: $7, 6.5, 6.1, 6.01, 6.001, \dots$ They are **marching toward 6**.

**ANSWER:** the steepness at $x = 3$ is **6**.
**CHECK:** no check needed — the table IS the answer. You watched the numbers march.
**WHAT IT MEANS:** at $x = 3$, if you take one tiny step right, the hill climbs about 6 times as fast.

> **ONE-LINE POINT:** The derivative is the number the slopes march toward as your friend point gets closer and closer. No magic. Just a table that marches.

---

## 1.3 The shortcut: the power rule (no more tables!)

Doing the marching table for every point is slow. Grown-ups found a shortcut, and it's the only formula you must remember:

$$\text{if } y = x^n, \text{ then the derivative is } n \cdot x^{n-1}$$

Say it out loud: **"if y equals x to the n, the derivative is n times x to the n minus 1."**

What it does: the little number $n$ **jumps down in front**, and then we subtract 1 from it.

| Machine | $n$ | Derivative (shortcut) | Check with a table? |
| :--- | :--- | :--- | :--- |
| $y = x^2$ | $2$ | $2x^{1} = 2x$ | At $x=3$: $2 \times 3 = 6$ ✓ (matches our table!) |
| $y = x^3$ | $3$ | $3x^{2}$ | At $x=2$: $3 \times 4 = 12$ |
| $y = x$ | $1$ | $1x^{0} = 1$ | At any $x$: slope is $1$ ✓ (it's a 45° line) |
| $y = x^0 = 1$ | $0$ | $0 \cdot x^{-1} = 0$ | At any $x$: slope is $0$ ✓ (perfectly flat) |

**Extra shortcuts you just trust (they were found the same way):**
- $y = \text{constant}$ (a flat line) → derivative $= 0$. *A flat line has no slope.*
- $y = e^x$ → derivative $= e^x$ (the number $e \approx 2.718$). *This machine's steepness equals its own height — that's why $e$ is everywhere in ML.*
- $y = \ln(x)$ (say: **"log natural of x"**) → derivative $= \frac{1}{x}$.

**Constants just ride along:** if the machine is $y = 5x^2$, the derivative is $5 \cdot 2x = 10x$. The 5 just tags along.

> **ONE-LINE POINT:** To take a derivative: bring the power down in front, subtract 1 from the power. That's it.

---

## 1.4 How to say the derivative

Grown-ups write the derivative in three ways that all mean the same thing:

| Writing | Say it out loud | Meaning |
| :--- | :--- | :--- |
| $f'(x)$ | **"f prime of x"** | the derivative of the machine $f$ |
| $\frac{dy}{dx}$ | **"dy over dx"** | the derivative of $y$ (small change in $y$ ÷ small change in $x$) |
| $\frac{d}{dx}$ | **"d by dx"** | "take the derivative of whatever follows" |

They're just three spellings of the same word. Don't let that scare you — it's like "soda" vs "pop" vs "fizzy drink". Same thing, different names.

---

# PART 2: FINDING THE LOWEST POINT (what ML actually does)

## 2.1 The problem: robots learn by making mistakes smaller

Imagine a robot guessing house prices. It starts by guessing badly. The **error** (how wrong it is) is a number. Small error = good robot. Big error = bad robot.

The robot has knobs (grown-ups call them **weights** or **parameters**). Turning a knob changes the error. The robot's goal: **turn the knobs so the error becomes as small as possible.**

Here's the picture. The horizontal line is a knob position. The height is the error:

```
error
  │                    ╲
  │               ╱───────╲
  │          ╱───╱          ╲
  │     ╱────╱                ╲
  │ ╱───╱                      ╲
  │●──╱──────────────────────────╲── knob position
  │  ╱                           ╲
  └────────────────────────────────────
  knob too small          knob too big
        └──────┬──────┘
          LOWEST point here
          (best knob setting)
```

The U-shape (or a hill like it) is the **loss curve**: error vs knob position. The lowest point is the best knob. The robot needs to **roll down the hill** like a ball.

![Ball rolling down a loss curve](/maths-images/calc-gd-1d.png)

## 2.2 The rolling-ball rule (gradient descent)

**GIVEN:** the robot is standing on the loss hill at some knob position.
**STEP 1:** compute the **steepness** (derivative) at its position.
**STEP 2:** if the hill goes **up** to the right (slope is positive), the lowest point is to the **left** → move the knob **left**.
**STEP 3:** if the hill goes **down** to the right (slope is negative), the lowest point is to the **right** → move the knob **right**.
**STEP 4:** move a small step, then repeat. Repeat. Repeat. (A robot can do this a million times a second.)
**CHECK:** when the slope becomes $0$, you're at the bottom (or at least the lowest part nearby). Stop.
**WHAT IT MEANS:** *go opposite the slope.* The slope says "up!" — you go down.

Grown-ups call this **gradient descent** (say: **"gray-dee-ent dee-sent"**). "Gradient" = slope, "descent" = going down. Literally "walking down the slope."

> **ONE-LINE POINT:** Training a robot = rolling a ball down the error hill: move opposite the slope, step by step, until the slope is flat.

## 2.3 How big should each step be? (the learning rate)

The **step size** is called the **learning rate**. Grown-ups write it as $\eta$ (say: **"eta"**, like "eta" in "e-tah").

- Step **too big** → the ball jumps over the lowest point and bounces forever. Bad!
- Step **too small** → the ball crawls so slowly you'd wait forever. Bad!
- Step **just right** → the ball rolls to the bottom quickly and smoothly. Good!

![Learning rate too big, too small, just right](/maths-images/calc-learning-rate.png)

**Rules of thumb:** start with a small number like $0.01$ (one hundredth). If the error is bouncing around wildly, make it smaller. That's it — that's the whole trick.

## 2.4 Two kinds of lowest points

Sometimes the hill has more than one dip:

```
error
  │      ╲        ╲
  │   ╱───╲    ╱───╲
  │╱──╱     ╲╱╱     ╲
  │╱──╱      ╱╱       ╲
  └──────────────────────── knob
      A        B
   (small dip) (deep dip)
```

- Dip **A** is a *local* minimum — lowest point *around here*, but not the best overall.
- Dip **B** is the *global* minimum — the lowest point of the whole hill. That's where we want to be.

The ball might get stuck in dip A. Grown-ups try to fix this by giving the ball a little push (that's what **momentum** and smarter optimizers like **Adam** do). You don't need to know how — just know it's a real problem, and it's why training has some randomness in it.

> **ONE-LINE POINT:** Gradient descent can get stuck in a small dip; smarter optimizers give the ball a push so it can climb out.

---

# PART 3: MORE THAN ONE DIAL (partial derivatives and the gradient)

## 3.1 The problem: robots have thousands of knobs

A real robot doesn't have one knob. It has thousands, even billions. The error hill now depends on *many* knob positions at once — it's not a line, it's a **bumpy landscape** (a map of hills and valleys).

**Question:** which way should we turn *all* the knobs at once?

**Answer:** look at each knob one at a time. While holding every other knob still, find the steepness *of just this one knob*. That's a **partial derivative** (say: **"par-shull duh-riv-uh-tiv"**).

Written like this:

$$\frac{\partial f}{\partial x}$$

Say it out loud: **"partial f, partial x"** — meaning: *the steepness of $f$ when only $x$ is allowed to move.* The curly $\partial$ (say: **"partial"**) is just a "d" that got fancy so you know only one knob is moving.

![Partial derivative: one slice of the hill at a time](/maths-images/calc-partial-derivative.png)

**GIVEN:** $f = x^2 + y^2$ (two knobs).
**STEP 1:** partial with respect to $x$: treat $y$ as a frozen number. Derivative of $x^2$ is $2x$; derivative of the frozen $y^2$ is $0$. Result: $\frac{\partial f}{\partial x} = 2x$.
**STEP 2:** partial with respect to $y$: treat $x$ as frozen. Result: $\frac{\partial f}{\partial y} = 2y$.
**CHECK:** at $x = 3, y = 4$: steepness in the $x$-direction is $6$, in the $y$-direction is $8$.
**WHAT IT MEANS:** the robot knows how much each knob matters right now — and can turn them all in the good direction.

## 3.2 The gradient — a compass that points downhill

Put all the partial derivatives in a list, and you get the **gradient**, written $\nabla f$ (say: **"nabla f"** — that upside-down triangle is called "nabla" or just "grad"):

$$\nabla f = \left[\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right]$$

**The gradient is a compass.** It points *uphill* (toward the steepest climb). To go down, walk the opposite way — that's exactly the rolling-ball rule from before, now with a compass. The robot steps a little bit in the **negative gradient** direction, over and over.

![Contour map with gradient arrows](/maths-images/calc-gradient-contour.png)

> **ONE-LINE POINT:** The gradient is a list of all the partial derivatives — a compass showing which way is up. Robots walk the opposite way.

---

# PART 4: MACHINES INSIDE MACHINES (the chain rule)

## 4.1 The problem: robots are stacked machines

Real robot brains are **layers of machines** stacked inside each other. The output of one machine feeds into the next:

```
x ──▶ [machine A] ──▶ [machine B] ──▶ output
```

If you change the input $x$ a tiny bit, how much does the final output change? The change has to travel through *both* machines. Grown-ups call this the **chain rule** — a chain of machines.

## 4.2 The chain rule in one sentence

**Multiply the steepnesses.**

$$\text{total steepness} = \text{steepness of A} \times \text{steepness of B}$$

**Why it works — the bicycle story.** Imagine you're riding a bike. If pedaling 1 step forward makes the wheel turn 2 times, and 1 wheel turn makes you move 3 meters, then 1 step of pedaling moves you $2 \times 3 = 6$ meters. **Two steps of change multiply.** Same idea: the change from the input passes through machine A (gets multiplied by A's steepness), then through machine B (multiplied by B's steepness).

## 4.3 The 4-step recipe (copy-paste this for any robot)

**STEP 1:** find the steepness of the *last* machine (B) at its input.
**STEP 2:** find the steepness of the *previous* machine (A) at its input.
**STEP 3:** multiply them.
**STEP 4:** the answer is how much the output changes when the input changes a tiny bit.

**GIVEN:** machine A = "double the number" (slope 2 everywhere), machine B = "square the number" (at input 5, slope $2 \times 5 = 10$).
**STEP 1:** steepness of B at its input (which is $2 \times 3 = 6$): $2 \times 6 = 12$.
**STEP 2:** steepness of A at $3$: $2$.
**STEP 3:** multiply: $12 \times 2 = 24$.
**CHECK:** if we raise $x$ from 3 to 3.01, machine A gives $6.02$, machine B gives $6.02^2 = 36.2404$ (vs $36$ before) → grew by $0.2404$, and $0.01 \times 24 = 0.24$. ✓ (Close enough — the tiny leftover is from the curve not being a straight line.)
**WHAT IT MEANS:** one tiny step of the input causes a $24\times$ bigger step in the output. The chain rule told us without doing any of that arithmetic.

> **ONE-LINE POINT:** Chain rule = multiply the steepnesses of the stacked machines. That single rule lets robots learn from their mistakes — it's why deep learning works.

---

# PART 5: THE INTEGRAL — the area under the hill

## 5.1 What it is

The **integral** (say: **"in-tuh-gral"**) answers: *how much total stuff is under a curve?* It's the **area** between the curve and the flat line, from one point to another.

```
y
  │      ╱╲
  │     ╱  ╲
  │    ╱ ▓▓ ╲
  │   ╱ ▓▓▓▓ ╲          ▓▓▓▓ = the area
  │  ╱ ▓▓▓▓▓▓ ╲
  │ ╱▓▓▓▓▓▓▓▓▓╲
  └─────────────── x
     a         b
```

Written as $\int_a^b f(x)\,dx$ (say: **"the integral from a to b of f of x, d x"**). The $\int$ symbol (say: **"integral"**) is a stretched-out S — it stands for **S**um. That's the whole secret: *an integral is just a fancy sum of many tiny pieces.*

## 5.2 How to find it: count the boxes

**GIVEN:** the flat-top curve $y = 3$ from $x = 0$ to $x = 4$.
**STEP 1:** this is a rectangle. Area = width × height.
**STEP 2:** width $= 4 - 0 = 4$, height $= 3$.
**STEP 3:** area $= 4 \times 3 = 12$.
**ANSWER:** $\int_0^4 3\,dx = 12$.

**For a curvy curve:** slice the hill into skinny rectangles, add up their areas. The skinnier the slices, the better the guess.

![Skinny rectangles under a curve](/maths-images/calc-riemann.png)

**Grown-up secret:** for most curves, *the integral is the opposite of the derivative.* Going down the hill = derivative; going up the hill (rebuilding the curve from its slopes) = integral. They undo each other, like putting on and taking off your shoes.

## 5.3 Where ML uses it

- **Expected value** ("average, but weighted"): add up *value × chance* for every possible value — that's an integral in disguise.
- **Probabilities of a range:** the chance that a random number falls between $a$ and $b$ is the area under its curve between $a$ and $b$.
- **Continuous "sums":** whenever ML needs to add up infinitely many tiny pieces, it uses an integral.

> **ONE-LINE POINT:** Integral = area under a curve = a super-sum of tiny rectangles. The opposite of a derivative.

---

# PART 6: WORDS YOU ONLY NEED TO RECOGNIZE (not master)

Some tools show up in ML books all the time. You don't need to *compute* them — you need to *recognize* them and know what they're for. That's it.

## 6.1 Limits (say: **"lim-its"**)

"Where is this machine heading as the input gets closer and closer to a spot?" Written $\lim_{x \to a} f(x)$ (say: **"the limit as x approaches a of f of x"**). It's the "marching" idea from Part 1 — the number the slopes were marching toward. You already used limits; you just didn't know their name.

## 6.2 Jacobian (say: **"juh-koh-bee-un"**)

A table of *all* the partial derivatives of a machine with many outputs. It's a "steepness spreadsheet" — tells you how sensitive every output is to every input. In ML it appears in fancy optimizers and in normalizing flows (image-generation math). Just recognize the name.

## 6.3 Hessian (say: **"hess-ee-un"**)

The *second* derivative, made into a table: "how fast is the steepness itself changing?" It tells you whether the ball is near the bottom of a dip or on top of a bump. Most ML training ignores it because it's expensive to compute. Recognize, don't fear.

## 6.4 Momentum (say: **"moh-men-tum"**)

Remember the ball getting stuck in small dips (Part 2)? Momentum makes the ball keep its speed from previous steps, so it rolls *through* small dips. That's the whole idea — a ball with momentum.

> **ONE-LINE POINT of Part 6:** Limit = marching number, Jacobian = steepness spreadsheet, Hessian = "is it a dip or a bump?" table, momentum = ball with speed. Recognize all four, master none.

---

# THE ONE-PAGE GLOSSARY (print this!)

| Word | Say it | Means |
| :--- | :--- | :--- |
| function $f(x)$ | f of x | a machine: number in, number out |
| graph | graph | the picture of a machine |
| slope $m$ | m | steepness of a line = rise ÷ run |
| derivative $f'(x)$ | f prime of x | steepness of a curve at one point |
| $\frac{dy}{dx}$ | dy over dx | same as derivative (spelling #2) |
| $\frac{\partial f}{\partial x}$ | partial f partial x | steepness of one knob only |
| gradient $\nabla f$ | nabla f | compass of all partials, points uphill |
| power rule | — | $x^n \to n \cdot x^{n-1}$ |
| chain rule | — | stacked machines: multiply steepnesses |
| integral $\int$ | integral | area under the curve = super-sum |
| learning rate $\eta$ | eta | step size when rolling downhill |
| local vs global minimum | — | small dip vs the deepest dip |
| momentum | — | ball keeps its speed |

**Final check — can you answer these?**
1. What's the derivative of $x^2$ at $x = 3$? (Hint: the table marched to...)
2. If the slope at your position is $+5$, should the knob move left or right?
3. Two machines stacked, steepnesses 3 and 4. Total steepness?
4. The integral of a flat curve from $0$ to $4$ with height $3$?

(Answers: 1. 6. 2. Left — go opposite the slope. 3. $3 \times 4 = 12$. 4. $12$.)