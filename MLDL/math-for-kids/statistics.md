---
title: Statistics for Kids (10-year-old friendly)
description: Statistics explained like a story — pocket money, test scores, coin flips, and guessing games. No scary algebra. Just pictures, tables, and one idea at a time.
tags: [math, kids, statistics, easy, beginner]
---

# STATISTICS FOR KIDS

> This document is written for someone who is **10 years old** and has **never done math beyond basic numbers**. Every idea comes with a picture, a story, and real numbers you can check yourself. Read it top to bottom. If a symbol looks scary, the "say it" note tells you how to say it out loud — and that's all you need.

---

## The whole story in 10 lines (read this first!)

1. **Statistics** = learning from a pile of numbers.
2. The **mean** is the average; the **median** is the middle value.
3. **Variance and standard deviation** measure how spread out the numbers are.
4. **Z-scores** tell you how unusual one number is.
5. The **CLT** says: average lots of things → you get a bell curve. Always.
6. A **confidence interval** is the "we're pretty sure it's in here" range.
7. **Hypothesis testing** answers: "is this difference real or just luck?"
8. The **p-value** is the luck meter: how often would luck alone explain what we saw?
9. **MLE** = pick the guess that makes the data most likely.
10. **Bootstrap** = replay the data many times to measure uncertainty.

---

# PART 0: THINGS YOU ALREADY KNOW (with a quick warm-up)

## 0.1 Data — the raw material (say: **"day-tuh"**)

**Data** is a pile of numbers (or words) measured from the real world. Think of it as a table:

```
┌────────────────────────────────┐
│ Friend │ Age │ Pocket money ($) │
├────────────────────────────────┤
│ Alice  │ 10  │      5.00        │   ← one row = one observation
│ Bob    │ 11  │      2.50        │
│ Chris  │ 10  │      8.00        │
└────────────────────────────────┘
   ↑ one column = one thing we measured
```

- One **row** = one person (one *observation*)
- One **column** = one thing we measured (one *variable* — say: **"vair-ee-uh-bull"**)
- In ML, each row is one training example, each column is one feature

**Two kinds of columns:**
- **Numbers** (age, money) → you can do math: averages, spread
- **Categories** (city, color) → you can only count: how many of each

## 0.2 Average and middle — the two "typical" numbers

**The mean** (say: **"mean"**) is the average: add everything, divide by how many.

**GIVEN:** pocket money: 5, 3, 8, 2, 7.
**STEP 1:** add: $5 + 3 + 8 + 2 + 7 = 25$.
**STEP 2:** divide by how many (5): $25 \div 5 = 5$.
**ANSWER:** mean = 5.
**WHAT IT MEANS:** if you shared all the money equally, each kid would get $5.

**The median** (say: **"mee-dee-un"**) is the middle value after sorting.
**STEP 1:** sort: 2, 3, 5, 7, 8.
**STEP 2:** middle one: 5.
**ANSWER:** median = 5. (Same here — but not always!)

**When they differ:** add one super-rich kid with $100 to the list. Now: mean $= \frac{125}{6} \approx 20.8$, but the median stays near 5. The mean got dragged up by one weird value; the median didn't. **The mean cares about every number; the median only cares about the middle.** For data with weird outliers, the median is the safer "typical" number.

> **ONE-LINE POINT:** mean = add ÷ count (the equal-share average). median = middle after sorting (ignores weird extremes).

## 0.3 The sum sign $\sum$ (say: **"sum"**)

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n$$

Just means "add them all". If $x_1 = 5, x_2 = 3, x_3 = 8$ then the sum is 16. Fancy look, simple job.

## 0.4 The only 6 symbols you need

| Symbol | Say it | Means |
| :--- | :--- | :--- |
| $\bar{x}$ | x bar | the average of our data (sample mean) |
| $\mu$ | myoo | the average of EVERYTHING (population mean) |
| $s$ | s | how spread out our data is (sample std dev) |
| $\sigma$ | sigma | how spread out EVERYTHING is (population std dev) |
| $n$ | n | how many numbers we have |
| $p$ | p | a chance, between 0 and 1 |

The difference between the bar/plain letters: Greek letters ($\mu$, $\sigma$) are for the *whole world* (we usually can't measure that), plain letters ($\bar{x}$, $s$) are for *our sample* (what we actually measured).

> **ONE-LINE POINT of Part 0:** Statistics = learning from tables of numbers. Mean = equal-share average, median = middle value.

---

# PART 1: HOW SPREAD OUT ARE THE NUMBERS?

## 1.1 The problem

Two classes both have an average test score of 70. But Class A has everyone scoring 68–72, while Class B has kids scoring 10 to 100. The average hid the difference! We need a **spread meter**.

## 1.2 Variance and standard deviation (say: **"vair-ee-ans"** and **"stan-dard dev-ee-ay-shun"**)

**The recipe — "the distance game":**
1. Find the average.
2. For each number, find how far it is from the average (the *deviation*).
3. Square each distance (so negatives become positive).
4. Average the squares → that's the **variance**.
5. Square-root it → that's the **standard deviation** ($s$ or $\sigma$). The square-root step brings it back to "normal units" (dollars, points).

**GIVEN:** scores 70, 72, 68 (average = 70).
**STEP 1:** distances from 70: 0, 2, −2.
**STEP 2:** squares: 0, 4, 4.
**STEP 3:** average of squares: $\frac{0 + 4 + 4}{3} \approx 2.67$ → that's the variance.
**STEP 4:** square root: $\sqrt{2.67} \approx 1.63$ → that's the standard deviation.
**CHECK:** the numbers are indeed all within ~1.6 of 70. ✓
**WHAT IT MEANS:** standard deviation ≈ "the typical distance from the average". Small $s$ = numbers bunched together. Big $s$ = numbers all over the place.

![Variance = spread](/maths-images/stat-variance.png)

**Rule of thumb:** for bell-shaped data, about 68% of everything sits within one standard deviation of the average, 95% within two, 99.7% within three.

> **ONE-LINE POINT:** standard deviation = typical distance from the average. Small = bunched, big = scattered.

## 1.3 Z-scores — "how unusual is THIS one number?"

The **z-score** (say: **"zee score"**) of a number = *how many standard deviations away from the average it is*.

$$z = \frac{\text{the number} - \text{average}}{\text{standard deviation}}$$

**GIVEN:** class average 70, standard deviation 10. Alice scored 90.
**STEP 1:** her distance from average: $90 - 70 = 20$.
**STEP 2:** divide by the standard deviation: $20 \div 10 = 2$.
**ANSWER:** $z = 2$ — Alice is 2 standard deviations above average.
**WHAT IT MEANS:** with the 68–95–99.7 rule: being 2 away is in the top ~2.5% of the class. Very unusual — in a good way!

**The z-score is a universal ruler.** "2 standard deviations above average" means the same thing for exam scores, heights, or apple weights — even though the numbers themselves are totally different.

> **ONE-LINE POINT:** z-score = "how many standard deviations from average". 0 = average, 2 = quite unusual, −2 = quite unusual the other way.

---

# PART 2: THE MAGIC OF AVERAGES — THE CLT (say: **"C-L-T"**)

## 2.1 The experiment everyone must do once

**CLT = Central Limit Theorem** (say: **"sen-truhl lim-it thee-uh-rum"**). It's the most magical fact in all of statistics, and it says:

> **Average lots of random things → you ALWAYS get a bell curve, no matter what the original things looked like.**

Even if the original pile of numbers is wildly lopsided, the *averages* of small groups from it form a nice bell. That's the CLT.

## 2.2 Try it with a die (a real worked example)

**GIVEN:** one die. Its rolls are uniform (each number equally likely). Average roll = 3.5, and the standard deviation of a single roll ≈ 1.71.

**Experiment part 1 — average 5 rolls at a time.** We roll 4 groups of 5:

| Group | The 5 rolls | Mean of the group |
| :--- | :--- | :--- |
| 1 | 4, 3, 2, 5, 3 | $\frac{17}{5} = 3.4$ |
| 2 | 6, 1, 4, 3, 5 | $\frac{19}{5} = 3.8$ |
| 3 | 2, 2, 5, 3, 3 | $\frac{15}{5} = 3.0$ |
| 4 | 5, 1, 3, 4, 4 | $\frac{17}{5} = 3.4$ |

**STEP 1:** the means are: 3.4, 3.8, 3.0, 3.4.
**STEP 2:** mean of the means $= \frac{3.4 + 3.8 + 3.0 + 3.4}{4} = \frac{13.6}{4} = 3.4$ — right around 3.5, the true average. ✓
**STEP 3:** the means are also *less spread out* than single rolls: they cluster near 3.5 instead of being scattered across 1–6.

**Experiment part 2 — average 30 rolls at a time.** Do this 100 times and draw a histogram of the 100 means. You get a **bell curve** centered at 3.5, and its spread is about $\frac{1.71}{\sqrt{30}} \approx 0.31$.

**WHAT IT MEANS:** averaging *shrinks the noise*. Bigger groups → tighter bells. The bell always centers on the true average (3.5), and its spread = (original spread) ÷ √(group size).

![CLT: averages form a bell](/maths-images/stat-clt.png)

**The magic formula (just recognize it):**

$$\text{spread of averages} = \frac{\sigma}{\sqrt{n}}$$

(say: **"sigma over the square root of n"**). More data → tighter bell → better guesses.

> **ONE-LINE POINT:** CLT = averages always make a bell curve, centered at the true average, with spread $\sigma / \sqrt{n}$. This is why polls and ML work at all.

---

# PART 3: "WE'RE PRETTY SURE IT'S IN HERE" — CONFIDENCE INTERVALS (say: **"kon-fih-dents in-ter-vull"**)

## 3.1 The problem

We measured a sample (e.g., 100 kids' pocket money). We want to know the *true* average for ALL kids. We can't measure everyone. What can we honestly say?

**Answer:** "We're 95% sure the true average is somewhere between $4.10 and $5.90." That range is a **confidence interval**.

## 3.2 How to build one (the recipe)

**GIVEN:** sample of 100 kids. Sample mean $\bar{x} = 5.00$, sample standard deviation $s = 4.60$.
**STEP 1:** the CLT tells us the sample mean's own spread: $\frac{4.60}{\sqrt{100}} = \frac{4.60}{10} = 0.46$. (Grown-ups call this the **standard error** — the "how wrong our average might be" number.)
**STEP 2:** for 95% confidence, go about 2 spreads out (from the 68–95–99.7 rule, 95% of a bell lives within ~2 standard deviations).
**STEP 3:** lower end: $5.00 - 2 \times 0.46 = 4.08$.
**STEP 4:** upper end: $5.00 + 2 \times 0.46 = 5.92$.
**ANSWER:** 95% confidence interval ≈ **[$4.08, $5.92]**.
**CHECK:** the interval is mean ± 2×(standard error). Wider = more confident but less precise.
**WHAT IT MEANS:** if we repeated the whole survey many times, about 95% of the intervals we built would contain the true average. We're 95% sure the true average is inside ours.

![Confidence interval](/maths-images/stat-confidence-interval.png)

**The important trap:** "95% confident" does NOT mean "the true average has a 95% chance of being in here." It means "this method works 95% of the time." Subtle difference — but the recipe is what matters here.

> **ONE-LINE POINT:** confidence interval = average ± 2 × (standard error). "We're 95% sure the truth is in this range."

---

# PART 4: "IS IT REAL OR LUCK?" — HYPOTHESIS TESTING (say: **"hy-poth-uh-sis"**)

## 4.1 The setup

Your friend claims their coin is lucky (biased toward heads). You flip it 20 times. It lands heads 16 times. **Is the coin rigged, or did you just get lucky?**

Statistics answers this with a **p-value** (say: **"pee-val-yoo"**) — the luck meter.

## 4.2 The recipe

**STEP 1 — the boring assumption (say: **"null hypothesis"**):** pretend the coin is perfectly fair (50/50). Assume no magic.
**STEP 2 — the question:** *if the coin is fair, how often would luck alone produce 16+ heads in 20 flips?*
**STEP 3 — the counting:** with a fair coin, the chance of 16 or more heads in 20 flips is about **0.59%** (a fair coin does this rarely — you'd need ~170 attempts to see it once).
**STEP 4 — the two-sided rule:** we also count the other extreme (4 or fewer heads), because a "lucky" coin could be biased *either* way: $0.59\% \times 2 \approx 1.2\%$.
**STEP 5 — the decision:** if the luck number is *tiny* (below 5%), we say: "luck alone almost never does this — the coin is probably rigged."

**ANSWER:** p ≈ 0.012 (1.2%) → below 5% → **the coin is suspicious**. The result is called **statistically significant** (say: **"sig-nif-ih-kant"**).

![p-value = luck meter](/maths-images/stat-pvalue.png)

**The decision table:**

| p-value | Say it | What you conclude |
| :--- | :--- | :--- |
| above 5% | not significant | "could easily be luck — no verdict" |
| below 5% | significant | "luck alone rarely does this — probably real" |
| below 1% | highly significant | "luck almost never does this — almost certainly real" |

**The two mistakes you can make (say: **"type 1"** and **"type 2"**):**
- **Type 1 error:** crying "it's real!" when it was just luck (false alarm — the 5% is the chance of this).
- **Type 2 error:** saying "it's nothing" when something was actually there (missed it).

> **ONE-LINE POINT:** p-value = "how often would pure luck produce what we saw?" Tiny p → real effect. Big p → can't tell.

## 4.3 Why ML cares

When a robot's accuracy improves from 71% to 72%, is that real learning or just luck? Hypothesis testing (comparing p-values on test sets) is how researchers tell the difference. Every serious ML paper uses it.

---

# PART 5: PICKING THE BEST GUESS — MLE AND MAP (say: **"M-L-E"**, **"M-A-P"**)

## 5.1 The problem

We saw some data (10 coin flips: 7 heads). We want to guess the coin's *true* heads-chance $p$. Which $p$ should we pick?

## 5.2 MLE — the "most likely" guess (Maximum Likelihood Estimation)

**The idea:** pick the $p$ that makes *the data we actually saw* the most likely.

**GIVEN:** 10 flips, 7 heads. Candidate guess: $p = 0.7$.
**STEP 1:** chance of 7 heads out of 10 when $p = 0.7$: the probability is
$$0.7^7 \times 0.3^3 \times \text{(number of ways)} \approx 0.267$$

**STEP 2:** try other guesses:
| Guess $p$ | Chance of seeing 7 heads out of 10 |
| :--- | :--- |
| $0.5$ | $\approx 0.117$ |
| $0.7$ | $\approx 0.267$ |
| $0.9$ | $\approx 0.057$ |

**STEP 3:** $p = 0.7$ wins — it makes our data the most likely.
**ANSWER:** MLE guess = **0.7**.
**CHECK:** makes sense — 7 heads out of 10 is exactly 70%.
**WHAT IT MEANS:** MLE = "which guess would most often produce what I saw?" The answer is usually just the obvious proportion: 7/10.

![MLE: pick the guess that fits the data best](/maths-images/stat-mle-bernoulli.png)

**The grown-up secret:** MLE is where the word "loss" comes from. "Loss" = how unlikely the model thinks the true answer is. Training = making the true answers as likely as possible = MLE in disguise.

## 5.3 MAP — MLE with prior opinions (Maximum A Posteriori)

**MAP** = MLE + *prior knowledge*. If we *already* believe the coin is probably fair (a "prior opinion"), MAP blends the data with that opinion.

**GIVEN:** our prior says "this coin is probably fair" (like a beta distribution with strength $\alpha = 2, \beta = 2$ — don't fear the names, just the idea). We see 10 flips, 7 heads.
**STEP 1:** MAP's recipe: *"data count + prior strength"*.
**STEP 2:** heads: $7 + 2 = 9$; tails: $3 + 2 = 5$.
**STEP 3:** guess $= \frac{9}{9 + 5} = \frac{9}{14} \approx 0.64$.
**ANSWER:** MAP guess ≈ **0.64** — between the data's 0.7 and the prior's 0.5.
**WHAT IT MEANS:** MAP pulls the answer toward our prior belief. More data → the data wins; less data → the prior wins. It's the "we already kind of knew" guess.

![MAP blends data with prior belief](/maths-images/stat-map.png)

**The one-liner that ties them together:**

$$\text{MLE} = \text{trust only the data} \qquad \text{MAP} = \text{data + prior opinion}$$

> **ONE-LINE POINT:** MLE = "the guess that makes my data most likely". MAP = "same, but nudged by what I already believed."

---

# PART 6: BOOTSTRAP — REPLAYING THE DATA (say: **"boot-strap"**)

## 6.1 The problem

We want to know how trustworthy our average is, but we only have one sample and can't collect more data. Can we still measure uncertainty? **Yes — by pretending.**

**Bootstrap** = *resample with replacement*: pick random items from your data, allowing repeats, forming a "new fake sample" the same size. Do this thousands of times, compute the average each time, and look at how much those averages vary.

**GIVEN:** 5 numbers: 4, 5, 6, 7, 9 (average = 6.2).
**STEP 1:** draw 5 numbers randomly *with repeats allowed* (like picking names from a hat, putting each back): e.g., 5, 7, 4, 9, 7 → average 6.4.
**STEP 2:** repeat: 9, 6, 5, 6, 4 → average 6.0. And 6, 7, 7, 5, 6 → average 6.2. And so on, 1,000 times.
**STEP 3:** collect all 1,000 averages into a histogram → it's a bell!
**STEP 4:** take the 2.5% and 97.5% cut points of the bell → that's the 95% confidence interval.
**WHAT IT MEANS:** by replaying our own data over and over, we measure how shaky our average really is. The data plays all possible "what-if" worlds for us.

![Bootstrap resampling](/maths-images/stat-bootstrap.png)

**Why it's cool:** no math formula needed — just random replays, repeated a lot. That's Monte Carlo thinking (from the Probability doc) applied to real data.

> **ONE-LINE POINT:** Bootstrap = replay your data thousands of times (allowing repeats), recompute your average each time, and see how much it wobbles. The wobble = the uncertainty.

---

# THE ONE-PAGE GLOSSARY (print this!)

| Word | Say it | Means |
| :--- | :--- | :--- |
| mean $\bar{x}$ | x bar | the average (add ÷ count) |
| median | mee-dee-un | the middle value after sorting |
| variance | vair-ee-ans | average of squared distances from the mean |
| standard deviation $s$, $\sigma$ | stan-dard dev-ee-ay-shun | typical distance from the average |
| z-score | zee score | "how many standard deviations away" a number is |
| CLT | C-L-T | averages always form a bell curve with spread $\sigma/\sqrt{n}$ |
| standard error | — | the spread of the sample mean itself |
| confidence interval | kon-fih-dents in-ter-vull | "we're 95% sure the truth is in this range" |
| p-value | pee-val-yoo | luck meter: how often luck alone produces what we saw |
| significant | sig-nif-ih-kant | p below 5% → probably real, not luck |
| type 1 / type 2 errors | — | false alarm / missed it |
| MLE | M-L-E | pick the guess that makes the data most likely |
| MAP | M-A-P | MLE + prior opinion (data nudged by belief) |
| bootstrap | boot-strap | replay the data (with repeats) to measure uncertainty |

**Final check — can you answer these?**
1. Scores: 60, 70, 80. Mean? Median?
2. A z-score of 2 means the number is...?
3. What does the CLT say averages do?
4. p-value of 0.03 — significant or not?
5. If you see 8 heads in 10 flips, what's the MLE guess for the coin's heads-chance?

(Answers: 1. Mean 70, median 70. 2. 2 standard deviations above average — quite unusual. 3. They always form a bell curve. 4. Significant (below 5%). 5. 0.8.)