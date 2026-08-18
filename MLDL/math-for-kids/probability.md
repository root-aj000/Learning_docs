---
title: Probability for Kids (10-year-old friendly)
description: Probability explained like a story — coin flips, dice, surprise meters, and guessing machines. No scary algebra. Just pictures, tables, and one idea at a time.
tags: [math, kids, probability, easy, beginner]
---

# PROBABILITY FOR KIDS

> This document is written for someone who is **10 years old** and has **never done math beyond basic numbers**. Every idea comes with a picture, a story, and real numbers you can check yourself. Read it top to bottom. If a symbol looks scary, the "say it" note tells you how to say it out loud — and that's all you need.

---

## The whole story in 10 lines (read this first!)

1. **Probability** = "how likely is something?" — a number between 0 and 1.
2. "AND" (two things both happen) → **multiply**.
3. "OR" (either of two things) → **add**.
4. A **random variable** is a machine that turns luck into numbers.
5. The **expected value** is the "average you'd get over many tries".
6. **PMF/PDF/CDF** are just "how much weight sits at each spot" — the dartboard picture.
7. **Bayes** flips "if A then B" into "if B then A" — and it's just counting people in a town.
8. **Normal distribution** = the bell curve = "most things are average".
9. **Entropy** = the surprise meter. More surprise = more information.
10. **Monte Carlo** = guess many times, average the guesses, get the answer.

---

# PART 0: THINGS YOU ALREADY KNOW (with a quick warm-up)

## 0.1 Fractions — the probability language

- $\frac{1}{2}$ = one half = 0.5 = 50% — all the same number, different costumes.
- Chance is always between 0 (never) and 1 (always).
- Coin flip, heads: $\frac{1}{2}$. Die roll, 4: $\frac{1}{6}$.

**The two rules you'll use forever:**
1. Chance of a *both things happen* = **multiply**: two coins both heads $= \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$.
2. Chance of *either of two things* (that can't both happen) = **add**: die shows 1 or 6 $= \frac{1}{6} + \frac{1}{6} = \frac{2}{6} = \frac{1}{3}$.

> **ONE-LINE POINT:** AND → multiply. OR → add. That's 80% of probability right there.

## 0.2 Counting — how many ways?

- **3! (say: "three factorial")** means $3 \times 2 \times 1 = 6$. It counts arrangements: 3 books can sit on a shelf in 6 orders (ABC, ACB, BAC, BCA, CAB, CBA).
- **The counting rule:** if choice 1 has $a$ options and choice 2 has $b$ options, together they have $a \times b$ options. (3 shirts × 2 pants = 6 outfits. You know this from real life!)

## 0.3 The sum sign $\sum$ (say: **"sum"**)

$$\sum_{i=1}^{3} x_i = x_1 + x_2 + x_3$$

Say it out loud: **"the sum from i equals 1 to 3 of x sub i"**. It just means: add up the things. If $x_1 = 3$, $x_2 = 5$, $x_3 = 2$, then the sum is $3 + 5 + 2 = 10$. That's the whole symbol. It's just "add them all" written fancy.

## 0.4 The only 6 symbols you need

| Symbol | Say it | Means |
| :--- | :--- | :--- |
| $P(A)$ | P of A | the chance that A happens |
| $P(A \mid B)$ | P of A given B | chance of A, knowing B already happened |
| $X$ | X | a machine that turns luck into a number |
| $P(X = x)$ | P of X equals x | chance the luck machine gives the number x |
| $E[X]$ | E of X | expected value = average over many tries |
| $f(x)$, $F(x)$ | f of x, F of x | continuous probability curves (Part 3) |

**That's it.** Everything else is built from these six.

> **ONE-LINE POINT of Part 0:** Chance is a number 0–1. AND multiplies, OR adds. Sum sign = "add them all".

---

# PART 1: WHAT IS PROBABILITY?

## 1.1 The sample space — the menu of everything that can happen

The **sample space** (written $\Omega$, say: **"omega"** — the big Greek O) is the *full menu of possible outcomes*.

- One coin flip: $\Omega = \{$ heads, tails $\}$
- One die roll: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Two coin flips: $\Omega = \{HH, HT, TH, TT\}$ (4 things!)

An **event** (say: **"ee-vent"**) is any part of the menu you care about: "roll an even number" = $\{2, 4, 6\}$.

**The golden rule:** the chance of an event = (how many menu items it covers) ÷ (how many menu items exist).

**GIVEN:** a die. Event: "roll an even number".
**STEP 1:** even numbers on the menu: $\{2, 4, 6\}$ → 3 items.
**STEP 2:** menu size: 6.
**STEP 3:** chance $= \frac{3}{6} = \frac{1}{2} = 0.5$.
**CHECK:** half of all rolls are even — you can verify with a real die in 10 rolls (you'll get roughly 5 even).
**WHAT IT MEANS:** 50% chance. "Even" is half the menu.

![Venn diagram of events](/maths-images/prob-venn.png)

## 1.2 The AND/OR rules, officially

- **AND** (independent things — one doesn't affect the other): **multiply**. Two coins both heads: $\frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$.
- **OR** (things that can't both happen, say: **"mutually exclusive"**): **add**. Die shows 1 or 6: $\frac{1}{6} + \frac{1}{6} = \frac{1}{3}$.
- **NOT**: 1 minus. Chance of NOT rolling a 6 $= 1 - \frac{1}{6} = \frac{5}{6}$.

**Why it matters in ML:** when a robot makes a prediction (say, "there's a 70% chance this email is spam"), it's doing probability under the hood — combining chances with multiply and add rules.

## 1.3 Conditional probability — "given that..."

**Conditional probability**, written $P(A \mid B)$ (say: **"P of A given B"**), answers: *"knowing B happened, what's the chance of A now?"*

**The great example — the pizza prediction:** on Mondays (B), the school cafeteria serves pizza 80% of the time.
- $P(\text{pizza} \mid \text{Monday})$ = 0.8
- $P(\text{pizza} \mid \text{Friday})$ = 0.2 (fish sticks on Fridays!)

The $\mid$ symbol (say: **"given"**) means: *the bar narrows down the world to just the days matching the thing after the bar.* Knowing extra information changes the chances. That's the whole idea.

**Why it matters in ML:** every ML prediction is conditional: $P(\text{spam} \mid \text{the email text})$. "Given this email, what's the chance it's spam?"

> **ONE-LINE POINT:** probability = menu counting. AND multiplies, OR adds, and "given" means "the world is now only the cases after the bar".

---

# PART 2: RANDOM VARIABLES AND THE AVERAGE

## 2.1 Random variables (say: **"ran-dom vair-ee-uh-bull"**) — luck machines

A **random variable** $X$ is a machine that turns luck into a number:

| Machine $X$ | Luck | Number it gives |
| :--- | :--- | :--- |
| "number of heads in 2 flips" | HH | 2 |
| | HT | 1 |
| | TH | 1 |
| | TT | 0 |

$X$ isn't one number — it's the *whole list of numbers with their chances*. $P(X = 1) = \frac{2}{4} = 0.5$ (say: **"the chance that X equals 1 is 0.5"**).

## 2.2 Expected value — the "average over many tries" (say: **"ex-pect-ed val-yoo"**)

The **expected value** $E[X]$ answers: *"if I play this game a million times, what's my average result?"*

**The recipe:** for every number the machine can give: multiply (number × its chance), then add everything up.

**GIVEN:** the lucky-dice game. Roll a die: you win the number it shows ($1, 2, \dots, 6$), each with chance $\frac{1}{6}$.
**STEP 1:** $1 \times \frac{1}{6} = \frac{1}{6}$
**STEP 2:** $2 \times \frac{1}{6} = \frac{2}{6}$ … and so on up to $6 \times \frac{1}{6} = \frac{6}{6}$
**STEP 3:** add: $\frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5$
**ANSWER:** $E[X] = 3.5$.
**CHECK:** roll a die 60 times and average — you'll land near 3.5.
**WHAT IT MEANS:** the average roll is 3.5, even though you can never *roll* a 3.5. The average is not always a real outcome!

**Why it matters in ML:** robots train by minimizing expected error. "On average over all possible inputs, how wrong am I?" That's $E[\text{error}]$, and making it small is the goal of all training.

> **ONE-LINE POINT:** Expected value = for each outcome, multiply number × chance, then add. It's the long-run average.

---

# PART 3: HOW PROBABILITY IS DRAWN — THE DARTBOARD

## 3.1 The dartboard picture (PMF, PDF, CDF)

Imagine a dartboard with the possible outcomes drawn along a line. Every outcome has a *pile of weight* (its chance). Where the weight piles up high, outcomes are likely; where it's low, they're unlikely.

**The three drawings you'll meet:**

| Name | Say it | What it is |
| :--- | :--- | :--- |
| PMF | P-M-F | the **chance** at each exact spot (for coins/dice — "discrete") |
| PDF | P-D-F | the **weight** along the line (for measuring — "continuous") |
| CDF | C-D-F | the **running total** of weight from left to right |

![PMF, PDF, CDF](/maths-images/prob-pmf-pdf-cdf.png)

**PMF (discrete — countable outcomes):** a bar chart. Each bar = the chance of exactly that outcome. Coin flips, dice rolls, number of spam emails.

**PDF (continuous — a measuring line):** for "how much does this apple weigh?" there are *infinitely many* possible weights. No single weight has a chance (it's infinitely thin) — but *ranges* do: the chance of weighing between 100 and 110 grams = the **area** of the curve between those two spots. **Bigger area = more likely.** That's the whole secret of the PDF: *probability = area under the curve.*

**CDF (the running total):** at any point, the CDF says "what fraction of the weight is to the left of here?" It always starts near 0 and climbs to 1.

> **ONE-LINE POINT:** PMF = bars at exact spots (coins/dice). PDF = weight line where *area* = chance (measurements). CDF = running total of weight.

## 3.2 The most famous PDF: the bell curve (normal)

Most natural measurements make the same shape — the **bell curve** (say: **"nor-mal dist-ri-bu-tion"**):

- Most weights sit near the middle (the average)
- Fewer and fewer as you walk away from the middle
- Perfectly symmetric: same shape left and right

![The bell curve](/maths-images/prob-gaussian.png)

**The magic numbers (say: **"68, 95, 99.7"**):** for bell-shaped data,
- 68% of everything sits within 1 spread-unit of the middle
- 95% within 2 spread-units
- 99.7% within 3 spread-units

![68-95-99.7 rule](/maths-images/prob-gaussian-689599.png)

(We'll learn what a "spread-unit" is in the Statistics doc — for now, know the picture.)

**Why it matters in ML:** robots assume many real-world noises and errors are bell-shaped. When you see a bell curve, you know: *most things are average, extremes are rare.*

## 3.3 The two workhorse distributions

**Binomial (say: **"bye-no-mee-ull"**) — "how many successes in N tries?"** Flip a coin 10 times: the chance of exactly 6 heads is a bar in the binomial distribution. It counts successes.

![Binomial distribution](/maths-images/prob-binomial.png)

**Uniform (say: **"yoo-ni-form"**) — "every outcome equally likely".** A fair die: each number gets the same bar height. Random number pickers use it.

![Uniform distribution](/maths-images/prob-uniform.png)

**You don't need to compute these by hand.** PyTorch and NumPy do it. You just need to recognize which picture fits which situation.

> **ONE-LINE POINT:** bell curve = most things are average; binomial = counting successes; uniform = everything equally likely.

---

# PART 4: BAYES — FLIPPING THE QUESTION (say: **"bayz"**)

## 4.1 The setup

Sometimes you know $P(\text{disease} \mid \text{test positive})$ is what you *want*, but the easy-to-measure thing is $P(\text{test positive} \mid \text{disease})$. **Bayes' theorem** is the machine that flips them. But you don't need the formula — you need the **counting trick**:

> **The town trick: pretend 10,000 people live in the town. Count. Done.**

## 4.2 The medical test example (the most famous example in all of ML)

**GIVEN:** a rare disease hits 1% of people. The test is 90% accurate *for sick people* (if you're sick, it says "sick" 90% of the time). It also has a 5% *false alarm* rate (if you're healthy, it wrongly says "sick" 5% of the time). You test positive. **What's the chance you're actually sick?**

**STEP 1 — the town:** pretend 10,000 people.
**STEP 2 — who's sick?** 1% of 10,000 = 100 sick, 9,900 healthy.
**STEP 3 — sick people with positive tests:** 90% of 100 = 90 (the true positives).
**STEP 4 — healthy people with positive tests (false alarms):** 5% of 9,900 = 495.
**STEP 5 — everyone who tested positive:** 90 + 495 = 585.
**STEP 6 — your real chance:** sick people among the positive-testers $= \frac{90}{585} \approx 0.154$.

**ANSWER:** about **15.4%**. Only about 1 in 6.5 people who test positive are actually sick!
**CHECK:** this feels shocking — but check the numbers: the 495 false alarms massively outnumber the 90 true positives. That's why doctors re-test.
**WHAT IT MEANS:** $P(\text{sick} \mid \text{positive})$ is small even when the test "sounds" accurate, because the disease is rare and false alarms add up.

**The one-sentence formula (for when you grow up):**

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

But seriously — the town counting trick always works, and it's what the formula secretly does.

> **ONE-LINE POINT:** Bayes = count the town. Positive tests = true positives + false alarms. Your chance = true positives ÷ (all positives).

---

# PART 5: ENTROPY — THE SURPRISE METER (say: **"en-tro-pee"**)

## 5.1 What it measures

**Entropy** measures *how surprised you should be* — or equivalently, *how much information an event carries*.

**GIVEN:** coin A is fair ($\frac{1}{2}$ heads), coin B is rigged ($\frac{99}{100}$ heads).
**STEP 1:** which result surprises you more? Getting tails on coin B ($\frac{1}{100}$ chance) — very surprising.
**STEP 2:** which coin gives more information per flip? Coin B's rare tails tell you a lot. But *on average*, coin A (fair) is more surprising per flip — every flip is a genuine coin toss.
**WHAT IT MEANS:** **rare events carry more information.** "The sun rose today" = no information (entropy ~0). "It snowed in July" = lots of information (high entropy).

## 5.2 The formula (just recognize it)

$$H = -\sum P(x) \cdot \log_2 P(x)$$

Say it out loud: **"H equals minus the sum of P of x times log base 2 of P of x"**. It's a machine: for each outcome, take (chance × log of chance), add up, flip the sign.

**The log (say: **"log"**) is just "how many times do I multiply 2 by itself to get this number?"**
- $\log_2(1) = 0$ (2 multiplied 0 times = 1 — no surprise)
- $\log_2(\frac{1}{2}) = -1$ (need one "divide by 2" — some surprise)
- $\log_2(\frac{1}{100}) \approx -6.6$ (need ~6.6 halvings — big surprise!)

**The pattern:** tiny chances → big negative logs → big entropy. **Rare = surprising = informative.**

**The punchline for ML:** "cross-entropy loss" (what every chatbot and classifier is trained with) is a surprise meter for *wrong answers*: the more surprised the model is by the right answer, the bigger the error, the harder it learns to fix itself. **Training = making the model less surprised.**

![Entropy as surprise](/maths-images/prob-entropy.png)

> **ONE-LINE POINT:** Entropy = surprise meter. Rare events = high surprise = high information. Training a model = making it less surprised.

---

# PART 6: MONTE CARLO — GUESSING YOUR WAY TO THE ANSWER (say: **"mon-tay kar-lo"**)

## 6.1 The idea

Some math problems are too hard to solve exactly. **Monte Carlo** says: *don't solve — guess. Many, many times. Average the guesses.*

**The classic example — finding π (pi) with rain:** draw a square with a circle inside it. Throw random dots (like raindrops) at the square. Count how many land inside the circle vs inside the square.

**STEP 1:** 1,000 dots thrown. Say 785 land in the circle.
**STEP 2:** fraction inside $= \frac{785}{1000} = 0.785$.
**STEP 3:** magic: this fraction × 4 ≈ the circle-to-square area ratio → $\pi \approx 3.14$. ✓
**WHAT IT MEANS:** random guessing, repeated enough times, gives the right answer. No formula needed.

![Monte Carlo estimation of pi](/maths-images/prob-montecarlo-pi.png)

## 6.2 Why ML uses it

- **Robot training uses random guesses** to explore: try a random action, see what happens, learn from it.
- **"How good is this strategy on average?"** — can't test every possible input, so robots test a random sample of inputs and average. The more samples, the closer to the true answer (that's the **Law of Large Numbers** — averages settle down with more tries).

![Monte Carlo convergence](/maths-images/prob-montecarlo-convergence.png)

> **ONE-LINE POINT:** Monte Carlo = guess a lot, average the guesses. The more guesses, the closer to the true answer.

---

# THE ONE-PAGE GLOSSARY (print this!)

| Word | Say it | Means |
| :--- | :--- | :--- |
| probability $P(A)$ | prob-uh-bil-i-tee | chance of A, between 0 and 1 |
| sample space $\Omega$ | omega | the full menu of outcomes |
| event | ee-vent | a part of the menu you care about |
| $P(A \mid B)$ | P of A given B | chance of A knowing B happened |
| random variable $X$ | ran-dom vair-ee-uh-bull | a machine turning luck into numbers |
| expected value $E[X]$ | ex-pect-ed val-yoo | long-run average (number × chance, added) |
| PMF | P-M-F | bars = chances at exact spots (coins, dice) |
| PDF | P-D-F | curve where *area* = chance (measurements) |
| CDF | C-D-F | running total of chance from left to right |
| normal / bell curve | nor-mul | most things are average; 68–95–99.7 |
| binomial | bye-no-mee-ull | counting successes in N tries |
| uniform | yoo-ni-form | every outcome equally likely |
| Bayes | bayz | flip the question; count the town |
| entropy $H$ | en-tro-pee | the surprise meter; rare = informative |
| Monte Carlo | mon-tay kar-lo | guess many times, average the guesses |

**Final check — can you answer these?**
1. Chance of two coins both landing heads?
2. Expected value of a fair die roll?
3. What does "area under the PDF" mean?
4. In the medical test story, why is your real chance only 15.4%?
5. Which is more surprising: a fair coin's tails, or a 99%-heads coin's tails?

(Answers: 1. $\frac{1}{4}$ — multiply. 2. 3.5. 3. The chance that a measurement lands in that range. 4. Because 495 false alarms swamp the 90 true positives. 5. The rigged coin's tails — rarer = more surprising = more information.)