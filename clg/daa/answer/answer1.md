═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 1 Answers — [6004]-480A (P3147)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 1**.
> New answers will always be added at the bottom.

---

┌─────────────────────────────────┐
│ 📝 Solving Question 1 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q1(a)              │
│ ✅ Questions Done: 0            │
│ 📋 Questions Remaining: 125     │
└─────────────────────────────────┘

**## ✏️ Question 1 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 10

---
### ❓ Full Question
Consider the following instance of the knapsack problem. Find the optimal solution by using dynamic programming approach.

| Item | Weight | Profit |
|------|--------|--------|
| 1 | 2 | 12 |
| 2 | 1 | 10 |
| 3 | 3 | 20 |
| 4 | 2 | 15 |

Capacity of the knapsack = 5.

---
### 📌 What is this question about?
This question is about the **0/1 Knapsack problem**, where each item can be either **taken completely** or **not taken at all**. We must choose items in such a way that the **total weight does not cross the bag capacity** and the **total profit becomes maximum**.

Think of it like packing a school bag before a trip. You can pick only a few things because the bag has a weight limit, so you want to choose the most useful combination.

---
### 📖 Formula & What Each Symbol Means
┌──────────────────────────────────────────────────────────────┐
│ FORMULA:                                                     │
│ V[i][w] = max( V[i-1][w] , P[i] + V[i-1][w - W[i]] )        │
│                                                              │
│ WHEN W[i] ≤ w                                                │
│                                                              │
│ If W[i] > w, then:                                           │
│ V[i][w] = V[i-1][w]                                          │
│                                                              │
│ WHERE:                                                       │
│ V[i][w]   = Maximum profit using first i items and capacity w│
│ i         = Number of items considered                       │
│ w         = Current capacity of knapsack                     │
│ P[i]      = Profit of item i                                 │
│ W[i]      = Weight of item i                                 │
│ V[i-1][w] = Profit if we do NOT include item i               │
│ P[i] + V[i-1][w-W[i]] = Profit if we include item i          │
└──────────────────────────────────────────────────────────────┘

**Important idea:**
For every cell, we check two choices:
1. **Do not take the item**
2. **Take the item** if its weight fits

Then we choose the option giving **higher profit**.

---
### 🔢 Step-by-Step Solution

#### Step 1: Write the input clearly

| Item | Weight W[i] | Profit P[i] |
|------|-------------|-------------|
| 1 | 2 | 12 |
| 2 | 1 | 10 |
| 3 | 3 | 20 |
| 4 | 2 | 15 |

Knapsack capacity = **5**

We will build a DP table `V[i][w]` where:
- rows = items from 0 to 4
- columns = capacity from 0 to 5

#### Step 2: Initial DP table
If there are **0 items** or **0 capacity**, profit is **0**.

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 |   |   |   |   |   |
| 2 | 0 |   |   |   |   |   |
| 3 | 0 |   |   |   |   |   |
| 4 | 0 |   |   |   |   |   |

---
#### Step 3: Fill Row 1 for Item 1
Item 1 has:
- Weight = 2
- Profit = 12

**Calculating Cell V[1][1]:**
- Current capacity = 1
- Weight of Item 1 = 2
- Check condition: **2 > 1** → CANNOT include item 1
- So we copy value from row above
- `V[1][1] = V[0][1] = 0`

**Calculating Cell V[1][2]:**
- Current capacity = 2
- Weight of Item 1 = 2
- Check condition: **2 ≤ 2** → CAN include item 1
- Option A: Do not include item 1 = `V[0][2] = 0`
- Option B: Include item 1 = `12 + V[0][2-2] = 12 + V[0][0] = 12 + 0 = 12`
- `V[1][2] = max(0, 12) = 12`

**Calculating Cell V[1][3]:**
- Current capacity = 3
- Weight of Item 1 = 2
- Check condition: **2 ≤ 3** → CAN include item 1
- Option A: Do not include item 1 = `V[0][3] = 0`
- Option B: Include item 1 = `12 + V[0][3-2] = 12 + V[0][1] = 12 + 0 = 12`
- `V[1][3] = max(0, 12) = 12`

**Calculating Cell V[1][4]:**
- Current capacity = 4
- Weight of Item 1 = 2
- Check condition: **2 ≤ 4** → CAN include item 1
- Option A: Do not include item 1 = `V[0][4] = 0`
- Option B: Include item 1 = `12 + V[0][4-2] = 12 + V[0][2] = 12 + 0 = 12`
- `V[1][4] = max(0, 12) = 12`

**Calculating Cell V[1][5]:**
- Current capacity = 5
- Weight of Item 1 = 2
- Check condition: **2 ≤ 5** → CAN include item 1
- Option A: Do not include item 1 = `V[0][5] = 0`
- Option B: Include item 1 = `12 + V[0][5-2] = 12 + V[0][3] = 12 + 0 = 12`
- `V[1][5] = max(0, 12) = 12`

**DP Table after completing Row 1:**

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|--> 1 | 0 | 0 | 12 | 12 | 12 | 12 |
| 2 | 0 |   |   |   |   |   |
| 3 | 0 |   |   |   |   |   |
| 4 | 0 |   |   |   |   |   |

---
#### Step 4: Fill Row 2 for Item 2
Item 2 has:
- Weight = 1
- Profit = 10

**Calculating Cell V[2][1]:**
- Current capacity = 1
- Weight of Item 2 = 1
- Check condition: **1 ≤ 1** → CAN include item 2
- Option A: Do not include item 2 = `V[1][1] = 0`
- Option B: Include item 2 = `10 + V[1][1-1] = 10 + V[1][0] = 10 + 0 = 10`
- `V[2][1] = max(0, 10) = 10`

**Calculating Cell V[2][2]:**
- Current capacity = 2
- Weight of Item 2 = 1
- Check condition: **1 ≤ 2** → CAN include item 2
- Option A: Do not include item 2 = `V[1][2] = 12`
- Option B: Include item 2 = `10 + V[1][2-1] = 10 + V[1][1] = 10 + 0 = 10`
- `V[2][2] = max(12, 10) = 12`

**Calculating Cell V[2][3]:**
- Current capacity = 3
- Weight of Item 2 = 1
- Check condition: **1 ≤ 3** → CAN include item 2
- Option A: Do not include item 2 = `V[1][3] = 12`
- Option B: Include item 2 = `10 + V[1][3-1] = 10 + V[1][2] = 10 + 12 = 22`
- `V[2][3] = max(12, 22) = 22`

**Calculating Cell V[2][4]:**
- Current capacity = 4
- Weight of Item 2 = 1
- Check condition: **1 ≤ 4** → CAN include item 2
- Option A: Do not include item 2 = `V[1][4] = 12`
- Option B: Include item 2 = `10 + V[1][4-1] = 10 + V[1][3] = 10 + 12 = 22`
- `V[2][4] = max(12, 22) = 22`

**Calculating Cell V[2][5]:**
- Current capacity = 5
- Weight of Item 2 = 1
- Check condition: **1 ≤ 5** → CAN include item 2
- Option A: Do not include item 2 = `V[1][5] = 12`
- Option B: Include item 2 = `10 + V[1][5-1] = 10 + V[1][4] = 10 + 12 = 22`
- `V[2][5] = max(12, 22) = 22`

**DP Table after completing Row 2:**

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 12 | 12 | 12 | 12 |
|--> 2 | 0 | 10 | 12 | 22 | 22 | 22 |
| 3 | 0 |   |   |   |   |   |
| 4 | 0 |   |   |   |   |   |

---
#### Step 5: Fill Row 3 for Item 3
Item 3 has:
- Weight = 3
- Profit = 20

**Calculating Cell V[3][1]:**
- Current capacity = 1
- Weight of Item 3 = 3
- Check condition: **3 > 1** → CANNOT include item 3
- So we copy value from row above
- `V[3][1] = V[2][1] = 10`

**Calculating Cell V[3][2]:**
- Current capacity = 2
- Weight of Item 3 = 3
- Check condition: **3 > 2** → CANNOT include item 3
- So we copy value from row above
- `V[3][2] = V[2][2] = 12`

**Calculating Cell V[3][3]:**
- Current capacity = 3
- Weight of Item 3 = 3
- Check condition: **3 ≤ 3** → CAN include item 3
- Option A: Do not include item 3 = `V[2][3] = 22`
- Option B: Include item 3 = `20 + V[2][3-3] = 20 + V[2][0] = 20 + 0 = 20`
- `V[3][3] = max(22, 20) = 22`

**Calculating Cell V[3][4]:**
- Current capacity = 4
- Weight of Item 3 = 3
- Check condition: **3 ≤ 4** → CAN include item 3
- Option A: Do not include item 3 = `V[2][4] = 22`
- Option B: Include item 3 = `20 + V[2][4-3] = 20 + V[2][1] = 20 + 10 = 30`
- `V[3][4] = max(22, 30) = 30`

**Calculating Cell V[3][5]:**
- Current capacity = 5
- Weight of Item 3 = 3
- Check condition: **3 ≤ 5** → CAN include item 3
- Option A: Do not include item 3 = `V[2][5] = 22`
- Option B: Include item 3 = `20 + V[2][5-3] = 20 + V[2][2] = 20 + 12 = 32`
- `V[3][5] = max(22, 32) = 32`

**DP Table after completing Row 3:**

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 12 | 12 | 12 | 12 |
| 2 | 0 | 10 | 12 | 22 | 22 | 22 |
|--> 3 | 0 | 10 | 12 | 22 | 30 | 32 |
| 4 | 0 |   |   |   |   |   |

---
#### Step 6: Fill Row 4 for Item 4
Item 4 has:
- Weight = 2
- Profit = 15

**Calculating Cell V[4][1]:**
- Current capacity = 1
- Weight of Item 4 = 2
- Check condition: **2 > 1** → CANNOT include item 4
- So we copy value from row above
- `V[4][1] = V[3][1] = 10`

**Calculating Cell V[4][2]:**
- Current capacity = 2
- Weight of Item 4 = 2
- Check condition: **2 ≤ 2** → CAN include item 4
- Option A: Do not include item 4 = `V[3][2] = 12`
- Option B: Include item 4 = `15 + V[3][2-2] = 15 + V[3][0] = 15 + 0 = 15`
- `V[4][2] = max(12, 15) = 15`

**Calculating Cell V[4][3]:**
- Current capacity = 3
- Weight of Item 4 = 2
- Check condition: **2 ≤ 3** → CAN include item 4
- Option A: Do not include item 4 = `V[3][3] = 22`
- Option B: Include item 4 = `15 + V[3][3-2] = 15 + V[3][1] = 15 + 10 = 25`
- `V[4][3] = max(22, 25) = 25`

**Calculating Cell V[4][4]:**
- Current capacity = 4
- Weight of Item 4 = 2
- Check condition: **2 ≤ 4** → CAN include item 4
- Option A: Do not include item 4 = `V[3][4] = 30`
- Option B: Include item 4 = `15 + V[3][4-2] = 15 + V[3][2] = 15 + 12 = 27`
- `V[4][4] = max(30, 27) = 30`

**Calculating Cell V[4][5]:**
- Current capacity = 5
- Weight of Item 4 = 2
- Check condition: **2 ≤ 5** → CAN include item 4
- Option A: Do not include item 4 = `V[3][5] = 32`
- Option B: Include item 4 = `15 + V[3][5-2] = 15 + V[3][3] = 15 + 22 = 37`
- `V[4][5] = max(32, 37) = 37`

**DP Table after completing Row 4:**

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 12 | 12 | 12 | 12 |
| 2 | 0 | 10 | 12 | 22 | 22 | 22 |
| 3 | 0 | 10 | 12 | 22 | 30 | 32 |
|--> 4 | 0 | 10 | 15 | 25 | 30 | 37 |

The bottom-right value is `V[4][5] = 37`.
So, the **maximum profit is 37**.

---
### 📊 Visualization
#### Final DP Table

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 12 | 12 | 12 | 12 |
| 2 | 0 | 10 | 12 | 22 | 22 | 22 |
| 3 | 0 | 10 | 12 | 22 | 30 | 32 |
| 4 | 0 | 10 | 15 | 25 | 30 | 37 |

#### Meaning of the last cell
`V[4][5] = 37`

This means:
- using first 4 items
- and knapsack capacity 5
- the best possible profit is **37**

---
### 🔄 Finding the Actual Items Selected (Backtracking)
Now we find **which items give profit 37**.

We start from the bottom-right cell:
- Start at `V[4][5] = 37`

**Step 1: Check Item 4**
- Compare `V[4][5]` with `V[3][5]`
- `V[4][5] = 37`
- `V[3][5] = 32`
- Since `37 ≠ 32`, **Item 4 is included**
- Record Item 4
- New capacity = `5 - 2 = 3`
- Move to `V[3][3]`

**Step 2: Check Item 3**
- Compare `V[3][3]` with `V[2][3]`
- `V[3][3] = 22`
- `V[2][3] = 22`
- Since `22 = 22`, **Item 3 is NOT included**
- Capacity remains `3`
- Move to `V[2][3]`

**Step 3: Check Item 2**
- Compare `V[2][3]` with `V[1][3]`
- `V[2][3] = 22`
- `V[1][3] = 12`
- Since `22 ≠ 12`, **Item 2 is included**
- Record Item 2
- New capacity = `3 - 1 = 2`
- Move to `V[1][2]`

**Step 4: Check Item 1**
- Compare `V[1][2]` with `V[0][2]`
- `V[1][2] = 12`
- `V[0][2] = 0`
- Since `12 ≠ 0`, **Item 1 is included**
- Record Item 1
- New capacity = `2 - 2 = 0`
- Stop

#### Backtracking Path

```text
V[4][5] = 37  --> Item 4 included
   |
   v
V[3][3] = 22  --> Item 3 not included
   |
   v
V[2][3] = 22  --> Item 2 included
   |
   v
V[1][2] = 12  --> Item 1 included
   |
   v
V[0][0] = 0   --> Stop
```

#### Selected Items Table

| Selected Item | Weight | Profit |
|---------------|--------|--------|
| 1 | 2 | 12 |
| 2 | 1 | 10 |
| 4 | 2 | 15 |
| **Total** | **5** | **37** |

---
### ✅ Final Answer
╔══════════════════════════════════════════╗
║  FINAL ANSWER:                           ║
║  Optimal items selected are 1, 2 and 4. ║
║  Total weight = 2 + 1 + 2 = 5           ║
║  Total profit = 12 + 10 + 15 = 37       ║
║  Hence, maximum profit = 37.            ║
╚══════════════════════════════════════════╝

---
### 🎯 Marking Tip
To score full marks, do not write only the final profit. You should write the **DP formula, complete DP table, every comparison for each cell, and backtracking steps showing items 1, 2, and 4 were selected**.

---
<!-- END OF QUESTION 1 -->
<!-- =================== -->

┌──────────────────────────────────────────────┐
│ ✅ Question 1 of 126 Complete                │
│ 📎 Successfully APPENDED to answer1.md       │
│ 📚 File now contains 1 question              │
│ ⏭️  Type "Next" for Question 2              │
│ 🔢 Next: Paper P3147 - Q1(b)                 │
└──────────────────────────────────────────────┘

---

┌─────────────────────────────────┐
│ 📝 Solving Question 2 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q1(b)              │
│ ✅ Questions Done: 1            │
│ 📋 Questions Remaining: 124     │
└─────────────────────────────────┘

**## ✏️ Question 2 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 8

---
### ❓ Full Question
What is job scheduling algorithm? How job scheduling algorithm can be solved using Greedy algorithmic approach? Explain your answer with respect to Principle, control abstraction, time analysis of control abstraction, of greedy approach for the following instance of job scheduling problem.

| Job | J1 | J2 | J3 | J4 | J5 |
|-----|----|----|----|----|----|
| Deadline | 2 | 1 | 3 | 2 | 1 |
| Profit | 60 | 100 | 20 | 40 | 20 |

---
### 📌 What is this question about?
This question is about **Job Sequencing with Deadlines**, a famous greedy problem. We have many jobs, each job takes **1 unit time**, and we want to choose the jobs in such an order that all selected jobs finish before their deadlines and the **total profit becomes maximum**.

Real-life example: imagine you have several small assignments, each with a last date and reward. Since you cannot finish all of them, you choose the most profitable ones first.

---
### 📖 Formula & What Each Symbol Means
There is no big mathematical formula here, but the greedy rule is:

┌────────────────────────────────────────────────────────────┐
│ GREEDY RULE:                                               │
│ 1. Sort jobs in decreasing order of profit                 │
│ 2. Place each job in the latest free slot before deadline  │
└────────────────────────────────────────────────────────────┘

**Why latest free slot?**
- Because it keeps earlier slots open for other jobs with smaller deadlines.

**Symbols:**
- `n` = number of jobs
- `d[i]` = deadline of job `i`
- `p[i]` = profit of job `i`
- `maxDeadline` = largest deadline among all jobs
- `slot[j]` = whether time slot `j` is free or filled

---
### 🔢 Step-by-Step Solution

#### 1) Definition of Job Scheduling Algorithm
A **job scheduling algorithm** arranges jobs in available time slots so that:
- each selected job is completed before its deadline,
- only one job is done at a time,
- total profit is maximum.

#### 2) Greedy Principle
The greedy approach says:
- always try the **highest profit job first**,
- place it in the **latest possible free slot** before its deadline.

This works because:
- choosing high-profit jobs early helps maximize total profit,
- placing a job as late as possible leaves more room for other jobs.

#### 3) Input Table

| Job | Deadline | Profit |
|-----|----------|--------|
| J1 | 2 | 60 |
| J2 | 1 | 100 |
| J3 | 3 | 20 |
| J4 | 2 | 40 |
| J5 | 1 | 20 |

#### 4) Sort jobs by profit in decreasing order

| Order | Job | Deadline | Profit |
|------|-----|----------|--------|
| 1 | J2 | 1 | 100 |
| 2 | J1 | 2 | 60 |
| 3 | J4 | 2 | 40 |
| 4 | J3 | 3 | 20 |
| 5 | J5 | 1 | 20 |

#### 5) Find total number of slots
Maximum deadline = **3**

So available slots are:
- Slot 1
- Slot 2
- Slot 3

Initially all are empty.

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| Empty | Empty | Empty |

#### 6) Place jobs one by one

**Job J2**
- Deadline = 1
- We can place it only in slot 1
- Slot 1 is free → place J2

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| J2 | Empty | Empty |

Current profit = 100

**Job J1**
- Deadline = 2
- Try latest slot first → slot 2
- Slot 2 is free → place J1

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| J2 | J1 | Empty |

Current profit = 100 + 60 = 160

**Job J4**
- Deadline = 2
- Try latest slot first → slot 2, but slot 2 already filled
- Try slot 1, but slot 1 also filled
- No free slot available before deadline
- So **J4 cannot be scheduled**

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| J2 | J1 | Empty |

Current profit = 160

**Job J3**
- Deadline = 3
- Try latest slot first → slot 3
- Slot 3 is free → place J3

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| J2 | J1 | J3 |

Current profit = 160 + 20 = 180

**Job J5**
- Deadline = 1
- Try slot 1
- Slot 1 already filled
- So **J5 cannot be scheduled**

Final schedule remains:

| Slot 1 | Slot 2 | Slot 3 |
|--------|--------|--------|
| J2 | J1 | J3 |

#### 7) Final job sequence
So the scheduled jobs are:
- **J2 → J1 → J3**

Total profit = **100 + 60 + 20 = 180**

---
### 📊 Visualization
#### Greedy Scheduling Table

| Step | Job Chosen | Deadline | Latest Free Slot Found | Action | Total Profit |
|------|------------|----------|------------------------|--------|--------------|
| 1 | J2 | 1 | 1 | Scheduled | 100 |
| 2 | J1 | 2 | 2 | Scheduled | 160 |
| 3 | J4 | 2 | None | Rejected | 160 |
| 4 | J3 | 3 | 3 | Scheduled | 180 |
| 5 | J5 | 1 | None | Rejected | 180 |

#### Simple Timeline

```text
Time Slot:   1     2     3
           +-----+-----+-----+
           | J2  | J1  | J3  |
           +-----+-----+-----+
```

---
### 🔄 Control Abstraction
```text
JobSequencing(J, n)
1. Sort all jobs in decreasing order of profit
2. Let m = maximum deadline
3. Mark all m slots as empty
4. for i = 1 to n do
5.     for j = min(m, deadline[i]) down to 1 do
6.         if slot[j] is empty then
7.             slot[j] = job[i]
8.             break
9. return scheduled jobs in slots
```

#### Explanation of control abstraction
- First, jobs are sorted by profit.
- Then for each job, we search backwards from its deadline.
- If a free slot is found, the job is placed there.

---
### ⏱️ Time Analysis of Control Abstraction
Let there be `n` jobs.

1. Sorting jobs by profit takes:
- **O(n log n)**

2. For each job, we may check several slots.
- In the worst case, checking slots takes **O(n)** for one job.
- For `n` jobs, total becomes **O(n²)**

So total time complexity is:

**O(n log n + n²) = O(n²)**

If a better data structure is used, this can be improved, but the standard classroom version is **O(n²)**.

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Job scheduling means arranging jobs so that       ║
║ deadlines are met and profit is maximum.          ║
║ Using greedy method, we sort jobs by profit and   ║
║ place each job in the latest free slot before     ║
║ its deadline.                                     ║
║                                                    ║
║ Optimal schedule: J2, J1, J3                      ║
║ Total profit = 100 + 60 + 20 = 180                ║
║ Time complexity of standard greedy abstraction    ║
║ = O(n²)                                           ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
To score full marks, write these three exact things: **definition of job scheduling, greedy rule (sort by profit + latest free slot), and the final slot table showing J2, J1, J3 with total profit 180**.

---
<!-- END OF QUESTION 2 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 3 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q2(a)              │
│ ✅ Questions Done: 2            │
│ 📋 Questions Remaining: 123     │
└─────────────────────────────────┘

**## ✏️ Question 3 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 8

---
### ❓ Full Question
What is greedy approach? Explain Job scheduling algorithm using Greedy approach for following examples. Give the sequence of job scheduling.

**Input 1:** Four jobs with following deadlines and profits:

| JobID | Deadline | Profit |
|-------|----------|--------|
| a | 4 | 20 |
| b | 1 | 10 |
| c | 1 | 40 |
| d | 1 | 30 |

**Input 2:** Five Jobs with following deadlines and profits:

| JobID | Deadline | Profit |
|-------|----------|--------|
| a | 2 | 100 |
| b | 1 | 19 |
| c | 2 | 27 |
| d | 1 | 25 |
| e | 3 | 15 |

---
### 📌 What is this question about?
The **greedy approach** makes the best immediate choice at each step, hoping that these local best choices will give the global best answer. In job scheduling, the greedy idea is to take the **highest profit job first**.

It is like picking the most rewarding task first, but still making sure it fits before its last date.

---
### 📖 Formula & What Each Symbol Means
**Greedy Rule for Job Scheduling:**
1. Sort jobs in decreasing order of profit.
2. Put each job in the latest free slot before its deadline.

**Symbols:**
- `d[i]` = deadline of job `i`
- `p[i]` = profit of job `i`
- `slot[j]` = time slot `j`

---
### 🔢 Step-by-Step Solution

#### Part A: What is Greedy Approach?
A **greedy approach** is an algorithm design method in which we choose the **best option available at the current step** without changing previous choices.

**Important points:**
- It is simple and fast.
- It works only when the problem has the **greedy-choice property**.
- It does not always work for every problem, but it works well for **job sequencing**, **fractional knapsack**, **activity selection**, etc.

---
#### Input 1 Solution

##### Step 1: Original table

| Job | Deadline | Profit |
|-----|----------|--------|
| a | 4 | 20 |
| b | 1 | 10 |
| c | 1 | 40 |
| d | 1 | 30 |

##### Step 2: Sort by profit in decreasing order

| Order | Job | Deadline | Profit |
|------|-----|----------|--------|
| 1 | c | 1 | 40 |
| 2 | d | 1 | 30 |
| 3 | a | 4 | 20 |
| 4 | b | 1 | 10 |

Maximum deadline = **4**
So slots are 1, 2, 3, 4.

Initially:

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| Empty | Empty | Empty | Empty |

##### Step 3: Schedule jobs

**Job c**
- deadline = 1
- try slot 1 → free
- place c in slot 1

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| c | Empty | Empty | Empty |

**Job d**
- deadline = 1
- try slot 1 → already full
- cannot schedule d

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| c | Empty | Empty | Empty |

**Job a**
- deadline = 4
- try slot 4 → free
- place a in slot 4

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| c | Empty | Empty | a |

**Job b**
- deadline = 1
- try slot 1 → already full
- cannot schedule b

Final schedule:

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| c | Empty | Empty | a |

##### Step 4: Sequence of scheduling for Input 1
Scheduled jobs are:
- **c, a**

Slot-wise sequence is:
- **Slot 1 = c**
- **Slot 4 = a**

Total profit = **40 + 20 = 60**

---
#### Input 2 Solution

##### Step 1: Original table

| Job | Deadline | Profit |
|-----|----------|--------|
| a | 2 | 100 |
| b | 1 | 19 |
| c | 2 | 27 |
| d | 1 | 25 |
| e | 3 | 15 |

##### Step 2: Sort by profit in decreasing order

| Order | Job | Deadline | Profit |
|------|-----|----------|--------|
| 1 | a | 2 | 100 |
| 2 | c | 2 | 27 |
| 3 | d | 1 | 25 |
| 4 | b | 1 | 19 |
| 5 | e | 3 | 15 |

Maximum deadline = **3**
So slots are 1, 2, 3.

Initially:

| 1 | 2 | 3 |
|---|---|---|
| Empty | Empty | Empty |

##### Step 3: Schedule jobs

**Job a**
- deadline = 2
- try slot 2 → free
- place a in slot 2

| 1 | 2 | 3 |
|---|---|---|
| Empty | a | Empty |

**Job c**
- deadline = 2
- try slot 2 → full
- try slot 1 → free
- place c in slot 1

| 1 | 2 | 3 |
|---|---|---|
| c | a | Empty |

**Job d**
- deadline = 1
- try slot 1 → full
- cannot schedule d

**Job b**
- deadline = 1
- try slot 1 → full
- cannot schedule b

**Job e**
- deadline = 3
- try slot 3 → free
- place e in slot 3

| 1 | 2 | 3 |
|---|---|---|
| c | a | e |

##### Step 4: Sequence of scheduling for Input 2
Scheduled jobs are:
- **c, a, e** in slots 1, 2, 3

Total profit = **27 + 100 + 15 = 142**

---
### 📊 Visualization
#### Final schedules

**Input 1**
```text
Slots:   1    2    3    4
        +----+----+----+----+
        | c  | -- | -- | a  |
        +----+----+----+----+
```

**Input 2**
```text
Slots:   1    2    3
        +----+----+----+
        | c  | a  | e  |
        +----+----+----+
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Greedy approach means choosing the best immediate ║
║ option at each step. In job scheduling, we sort   ║
║ jobs by profit and place each job in the latest   ║
║ possible free slot before its deadline.           ║
║                                                    ║
║ Input 1 sequence: c, a                            ║
║ Total profit = 60                                 ║
║                                                    ║
║ Input 2 sequence: c, a, e                         ║
║ Total profit = 142                                ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Write the **sorted profit table** and the **slot filling process**. Many students lose marks by writing only the final sequence without showing why rejected jobs were rejected.

---
<!-- END OF QUESTION 3 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 4 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q2(b)              │
│ ✅ Questions Done: 3            │
│ 📋 Questions Remaining: 122     │
└─────────────────────────────────┘

**## ✏️ Question 4 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 10

---
### ❓ Full Question
What is optimal binary search tree? How dynamic programming approach is used to build OBST for following table.

| Keys | 10 | 20 | 30 | 40 |
|------|----|----|----|----|
| Frequency | 4 | 2 | 6 | 3 |

---
### 📌 What is this question about?
An **Optimal Binary Search Tree (OBST)** is a binary search tree arranged in such a way that the **expected search cost is minimum**. The key that is searched more often should be placed closer to the root so that it can be found faster.

Real-life analogy: if one chapter is asked very often in the exam, you keep it on top of your revision list, not at the end.

---
### 📖 Formula & What Each Symbol Means
For successful searches only, the dynamic programming formula is:

┌──────────────────────────────────────────────────────────────┐
│ cost[i][j] = min { cost[i][r-1] + cost[r+1][j] + sum(i,j) } │
│                 for every root r from i to j                │
└──────────────────────────────────────────────────────────────┘

**Where:**
- `cost[i][j]` = minimum search cost for keys from `i` to `j`
- `r` = chosen root between `i` and `j`
- `sum(i,j)` = total frequency from key `i` to key `j`

**Why do we add sum(i,j)?**
Because when the whole subtree goes one level deeper, every key inside it contributes one extra comparison.

---
### 🔢 Step-by-Step Solution

#### Step 1: Input table

| Index | Key | Frequency |
|------|-----|-----------|
| 1 | 10 | 4 |
| 2 | 20 | 2 |
| 3 | 30 | 6 |
| 4 | 40 | 3 |

Let:
- f1 = 4
- f2 = 2
- f3 = 6
- f4 = 3

#### Step 2: Base case for single keys
A single key tree has cost equal to its own frequency.

- `cost[1][1] = 4`
- `cost[2][2] = 2`
- `cost[3][3] = 6`
- `cost[4][4] = 3`

#### Step 3: Subtrees of length 2

##### cost[1][2]
Keys = 10, 20
Sum of frequencies = 4 + 2 = 6

Try root = 10:
- left cost = 0
- right cost = cost[2][2] = 2
- total = 0 + 2 + 6 = 8

Try root = 20:
- left cost = cost[1][1] = 4
- right cost = 0
- total = 4 + 0 + 6 = 10

Take minimum:
- `cost[1][2] = min(8,10) = 8`
- Root = 10

##### cost[2][3]
Keys = 20, 30
Sum of frequencies = 2 + 6 = 8

Try root = 20:
- left cost = 0
- right cost = cost[3][3] = 6
- total = 0 + 6 + 8 = 14

Try root = 30:
- left cost = cost[2][2] = 2
- right cost = 0
- total = 2 + 0 + 8 = 10

Take minimum:
- `cost[2][3] = min(14,10) = 10`
- Root = 30

##### cost[3][4]
Keys = 30, 40
Sum of frequencies = 6 + 3 = 9

Try root = 30:
- left cost = 0
- right cost = cost[4][4] = 3
- total = 0 + 3 + 9 = 12

Try root = 40:
- left cost = cost[3][3] = 6
- right cost = 0
- total = 6 + 0 + 9 = 15

Take minimum:
- `cost[3][4] = min(12,15) = 12`
- Root = 30

#### Step 4: Subtrees of length 3

##### cost[1][3]
Keys = 10, 20, 30
Sum of frequencies = 4 + 2 + 6 = 12

Try root = 10:
- left cost = 0
- right cost = cost[2][3] = 10
- total = 0 + 10 + 12 = 22

Try root = 20:
- left cost = cost[1][1] = 4
- right cost = cost[3][3] = 6
- total = 4 + 6 + 12 = 22

Try root = 30:
- left cost = cost[1][2] = 8
- right cost = 0
- total = 8 + 0 + 12 = 20

Take minimum:
- `cost[1][3] = min(22,22,20) = 20`
- Root = 30

##### cost[2][4]
Keys = 20, 30, 40
Sum of frequencies = 2 + 6 + 3 = 11

Try root = 20:
- left cost = 0
- right cost = cost[3][4] = 12
- total = 0 + 12 + 11 = 23

Try root = 30:
- left cost = cost[2][2] = 2
- right cost = cost[4][4] = 3
- total = 2 + 3 + 11 = 16

Try root = 40:
- left cost = cost[2][3] = 10
- right cost = 0
- total = 10 + 0 + 11 = 21

Take minimum:
- `cost[2][4] = min(23,16,21) = 16`
- Root = 30

#### Step 5: Subtree of length 4

##### cost[1][4]
Keys = 10, 20, 30, 40
Sum of frequencies = 4 + 2 + 6 + 3 = 15

Try root = 10:
- left cost = 0
- right cost = cost[2][4] = 16
- total = 0 + 16 + 15 = 31

Try root = 20:
- left cost = cost[1][1] = 4
- right cost = cost[3][4] = 12
- total = 4 + 12 + 15 = 31

Try root = 30:
- left cost = cost[1][2] = 8
- right cost = cost[4][4] = 3
- total = 8 + 3 + 15 = 26

Try root = 40:
- left cost = cost[1][3] = 20
- right cost = 0
- total = 20 + 0 + 15 = 35

Take minimum:
- `cost[1][4] = min(31,31,26,35) = 26`
- Root = 30

---
### 📊 Visualization
#### Cost Table

| i\j | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| 1 | 4 | 8 | 20 | 26 |
| 2 | - | 2 | 10 | 16 |
| 3 | - | - | 6 | 12 |
| 4 | - | - | - | 3 |

#### Root choices
- root[1,2] = 10
- root[2,3] = 30
- root[3,4] = 30
- root[1,3] = 30
- root[2,4] = 30
- root[1,4] = 30

#### Final OBST
```text
        30
       /  \
     10    40
       \
        20
```

Explanation:
- Root is 30
- Left subtree for 10 and 20 has root 10
- 20 becomes right child of 10
- 40 is right child of 30

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ An Optimal Binary Search Tree is a BST with       ║
║ minimum search cost. Using dynamic programming,   ║
║ the minimum cost for the given keys is 26.        ║
║                                                    ║
║ The OBST is:                                      ║
║                30                                 ║
║               /  \                                ║
║             10    40                              ║
║               \                                   ║
║                20                                 ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Always show **all root choices for each subproblem**. In OBST questions, marks are usually given for the **cost table + final tree**, not only for the final root.

---
<!-- END OF QUESTION 4 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 5 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q3(a)              │
│ ✅ Questions Done: 4            │
│ 📋 Questions Remaining: 121     │
└─────────────────────────────────┘

**## ✏️ Question 5 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 8

---
### ❓ Full Question
Explain with suitable example Backtracking: Principle, control abstraction, time analysis of control abstraction.

---
### 📌 What is this question about?
Backtracking is a method used to solve problems by **trying a choice, checking whether it is promising, and undoing it if it leads to failure**. It is useful when a problem has many possible answers but only some are valid.

It is like solving a maze: move ahead, and if you hit a dead end, come back and try another path.

---
### 📖 Formula & What Each Symbol Means
There is no single formula, but the standard **control abstraction** for backtracking is:

```text
BACKTRACK(k)
1. if current solution is complete
2.      output solution
3. else
4.      generate candidates for next position
5.      for each candidate c
6.          if c is promising
7.               include c
8.               BACKTRACK(k+1)
9.               remove c
```

**Symbols:**
- `k` = current level or current position in the solution
- `candidate` = a possible next choice
- `promising` = a test telling whether the partial solution can still lead to a valid full solution

---
### 🔢 Step-by-Step Solution

#### 1) Definition of Backtracking
Backtracking is an algorithmic strategy in which we:
1. build the solution step by step,
2. check whether the current partial solution is valid,
3. continue if it is valid,
4. go back if it is invalid.

#### 2) Principle of Backtracking
The principle is:
- explore only **promising** nodes,
- do not explore nodes that cannot lead to a valid solution.

This saves time compared to checking every possible case fully.

#### 3) Suitable Example: 4-Queens Problem
Problem: place 4 queens on a 4×4 chessboard so that no two queens attack each other.

A queen attacks along:
- same row
- same column
- diagonal

Backtracking idea:
- place one queen row by row,
- if a queen attacks another queen, go back and change the position.

One valid solution is:
- Row 1 → Column 2
- Row 2 → Column 4
- Row 3 → Column 1
- Row 4 → Column 3

#### 4) State space idea
At each row we try different columns.
- Some choices become invalid quickly.
- Those branches are pruned.

Example path:
- Put queen in row 1, column 1
- Try row 2, but many columns become invalid
- If no valid column exists, backtrack to row 1 and try next column

#### 5) Generic Control Abstraction
```text
BACKTRACK(a, k, n)
1. if k = n then
2.      print a[1..n]
3. else
4.      k = k + 1
5.      construct all candidates for a[k]
6.      for each candidate x do
7.          a[k] = x
8.          if a[k] is promising then
9.              BACKTRACK(a, k, n)
```

#### 6) What does “promising” mean?
A node is **promising** if the partial solution can still become a full valid solution.

For 4-Queens:
- a new queen is promising only if it is not in the same column or diagonal as previous queens.

#### 7) Time Analysis of Control Abstraction
Backtracking is usually exponential in the worst case.

If:
- branching factor = `b`
- depth of tree = `d`

then worst-case time is:
- **O(b^d)**

For many problems such as:
- N-Queens
- Sum of subsets
- Graph coloring

worst-case time is exponential because many possibilities may need to be explored.

For the 4-Queens example, the algorithm may try many placements, but promising checks cut down invalid branches.

So we write:
- **Worst-case time complexity: Exponential**
- commonly written as **O(m^n)** or **O(b^d)** depending on the problem

---
### 📊 Visualization
#### Backtracking tree idea

```text
Start
 ├── Choice 1 ✓
 │    ├── Choice 1.1 ✗
 │    ├── Choice 1.2 ✓
 │    │    ├── Choice 1.2.1 ✗
 │    │    └── Choice 1.2.2 ✓
 │    └── ...
 ├── Choice 2 ✓
 └── Choice 3 ✗
```

- `✓` = promising node
- `✗` = dead end, so backtrack

#### 4-Queens one valid arrangement
```text
Row 1 -> Col 2
Row 2 -> Col 4
Row 3 -> Col 1
Row 4 -> Col 3
```

Chessboard view:

```text
. Q . .
. . . Q
Q . . .
. . Q .
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Backtracking is a technique in which we build a   ║
║ solution step by step and reject a partial        ║
║ solution as soon as it becomes invalid.           ║
║                                                    ║
║ Principle: explore only promising nodes.          ║
║ Control abstraction: generate candidate, test     ║
║ promising condition, recurse, and backtrack.      ║
║ Worst-case time complexity is exponential.        ║
║ A suitable example is the 4-Queens problem.       ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
For full marks, write **three parts separately**: definition, generic control abstraction, and worst-case exponential time complexity. Add one small example such as **4-Queens** or **sum of subsets**.

---
<!-- END OF QUESTION 5 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 6 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q3(b)              │
│ ✅ Questions Done: 5            │
│ 📋 Questions Remaining: 120     │
└─────────────────────────────────┘

**## ✏️ Question 6 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 9

---
### ❓ Full Question
Compare between greedy method and dynamic programming with respect to:
1. Feasibility
2. Optimality
3. Recursion
4. Memorization
5. Time complexity

---
### 📌 What is this question about?
This question asks for a comparison between two important algorithm design strategies: **Greedy Method** and **Dynamic Programming**. Both solve optimization problems, but they work in different ways.

A simple idea is this: greedy takes the **best decision now**, while dynamic programming studies **all important smaller answers first** and then builds the final answer.

---
### 📖 Formula & What Each Symbol Means
No numerical formula is needed here. This is a **theory comparison** question.

---
### 🔢 Step-by-Step Solution

#### Greedy Method
Greedy method makes the best available choice at the current step and never changes it later.

Examples:
- Fractional Knapsack
- Job Sequencing
- Activity Selection

#### Dynamic Programming
Dynamic programming solves problems by breaking them into smaller overlapping subproblems and storing their answers for reuse.

Examples:
- 0/1 Knapsack
- Matrix Chain Multiplication
- Optimal Binary Search Tree

#### Comparison Table

| Basis | Greedy Method | Dynamic Programming |
|------|---------------|--------------------|
| **1. Feasibility** | Chooses a locally best feasible option at each step | Builds solution from smaller feasible subproblems |
| **2. Optimality** | Gives optimal solution only for problems having greedy-choice property | Gives optimal solution when problem has optimal substructure and overlapping subproblems |
| **3. Recursion** | Usually does not require deep recursive formulation | Often naturally expressed recursively, then solved by memoization or tabulation |
| **4. Memorization** | Does not store solutions of subproblems | Stores subproblem results in table or memo structure |
| **5. Time Complexity** | Usually faster and simpler | Usually slower than greedy because more states are explored |

#### Detailed Explanation Point by Point

**1) Feasibility**
- Greedy checks whether the current choice is feasible and takes it if it fits.
- Dynamic programming checks many feasible subproblem combinations.

**2) Optimality**
- Greedy is not always optimal.
- Dynamic programming is designed to get the optimal answer when the required properties are present.

**3) Recursion**
- Greedy usually proceeds step by step in one direction.
- DP often starts from a recursive relation like:
  `DP[i] = best of smaller previous states`

**4) Memorization**
- Greedy does not save old states.
- DP uses **memoization** or **tabulation** so that the same subproblem is not solved again and again.

**5) Time Complexity**
- Greedy methods are generally more efficient, often O(n) or O(n log n).
- DP may need O(n²), O(nW), O(n³), etc., depending on the table size.

---
### 📊 Visualization
```text
Greedy:
Current step -> Best local choice -> Move ahead

Dynamic Programming:
Small subproblems -> Store answers -> Build final optimal answer
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Greedy method takes the best immediate choice,    ║
║ while dynamic programming stores and reuses       ║
║ solutions of smaller subproblems.                 ║
║                                                    ║
║ Greedy is simpler and faster but not always       ║
║ optimal. Dynamic programming is more systematic   ║
║ and usually gives the optimal answer when the     ║
║ problem has optimal substructure.                 ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Make a **comparison table**. In theory questions, table format gets better marks than long paragraphs because the examiner can check all five points quickly.

---
<!-- END OF QUESTION 6 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 7 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q4(a)              │
│ ✅ Questions Done: 6            │
│ 📋 Questions Remaining: 119     │
└─────────────────────────────────┘

**## ✏️ Question 7 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 8

---
### ❓ Full Question
What is sum of subset problem? Solve sum of subset problem for following instance using backtracking approach.

Input: set[] = {2, 3, 5, 6, 8, 10}, sum = 10

---
### 📌 What is this question about?
The **sum of subsets problem** asks whether we can choose some elements from a set so that their total becomes exactly equal to a target sum. Backtracking is used because we must try including or excluding each element.

It is like selecting some coins from a box so that the total becomes exactly 10 rupees.

---
### 📖 Formula & What Each Symbol Means
At each step, we make two choices for each element:
1. **Include the element**
2. **Exclude the element**

A node is promising if:
- current sum is not greater than target,
- and it is still possible to reach target using remaining elements.

**Symbols:**
- `s` = current sum
- `x[k]` = whether kth element is selected (`1`) or not (`0`)
- `target` = required sum = 10

---
### 🔢 Step-by-Step Solution

#### Step 1: Given data
Set = `{2, 3, 5, 6, 8, 10}`

Target sum = `10`

#### Step 2: Idea of backtracking
For each element, we decide:
- include it,
- or exclude it.

If current sum becomes more than 10, we stop exploring that branch.

#### Step 3: Search for valid subsets

##### Branch 1: Include 2
Current subset = `{2}`
Current sum = `2`

Now try next element 3.

**Include 3**
- subset = `{2,3}`
- sum = `5`

Now try next element 5.

**Include 5**
- subset = `{2,3,5}`
- sum = `10`
- This is a valid solution ✓

So one solution is:
- `{2,3,5}`

Backtrack and continue.

##### Branch 2: From subset {2}, skip 3 and try 5
- subset = `{2,5}`
- sum = `7`
- remaining needed = `3`
- next numbers 6,8,10 cannot make exactly 3
- so this branch fails ✗

##### Branch 3: From subset {2}, try 8
- subset = `{2,8}`
- sum = `10`
- valid solution ✓

So another solution is:
- `{2,8}`

##### Branch 4: Try 10 alone
- subset = `{10}`
- sum = `10`
- valid solution ✓

So another solution is:
- `{10}`

#### Step 4: All solutions found
Valid subsets are:
1. `{2,3,5}`
2. `{2,8}`
3. `{10}`

---
### 📊 Visualization
#### State Space Tree

```text
                         {}
                       /    \
                    {2}      {}
                   /   \
               {2,3}   {2}
               /   \
         {2,3,5}  {2,3}
            ✓

Other successful branches:
{2,8} ✓
{10}  ✓
```

#### Solution Table

| Subset | Sum | Valid? |
|--------|-----|--------|
| {2,3,5} | 10 | Yes |
| {2,8} | 10 | Yes |
| {10} | 10 | Yes |

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Sum of subset problem asks us to find subsets     ║
║ whose sum is exactly equal to the target value.   ║
║ Using backtracking for the set {2,3,5,6,8,10}     ║
║ and target 10, the valid subsets are:             ║
║ {2,3,5}, {2,8}, and {10}.                         ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
In backtracking numericals, do not write only one subset. Write the **state space idea** and mention that branches are pruned when the sum becomes greater than the target.

---
<!-- END OF QUESTION 7 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 8 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q4(b)              │
│ ✅ Questions Done: 7            │
│ 📋 Questions Remaining: 118     │
└─────────────────────────────────┘

**## ✏️ Question 8 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 9

---
### ❓ Full Question
What is Branch and Bound method? Write control abstraction for Least Cost search.

---
### 📌 What is this question about?
Branch and Bound is a method used to solve optimization problems by exploring different solution paths and cutting off paths that cannot produce a better answer. Least Cost Search is a branch and bound strategy where we always expand the live node with the **minimum cost** first.

This is similar to choosing the cheapest partial route first when planning a trip.

---
### 📖 Formula & What Each Symbol Means
There is no fixed numeric formula, but these ideas are important:

- **Branch** = divide into subproblems or children
- **Bound** = compute a limit to decide whether to continue or prune
- **Live node** = generated but not expanded yet
- **Dead node** = already expanded or pruned
- **E-node** = node selected for expansion

In least cost branch and bound:
- choose the live node with **smallest cost** from a priority queue.

---
### 🔢 Step-by-Step Solution

#### 1) What is Branch and Bound?
Branch and Bound is an algorithm design technique used for optimization problems such as:
- 0/1 Knapsack
- Traveling Salesman Problem
- Assignment Problem

It works in two parts:

**Branch:**
- divide the problem into smaller choices
- create a state space tree

**Bound:**
- compute a bound value for each node
- if the node cannot lead to a better answer, prune it

#### 2) Main idea of Least Cost Search
In least cost search:
- among all live nodes, pick the node whose current cost is minimum
- expand that node first

To do this efficiently, we use a **priority queue / min-heap**.

#### 3) Control Abstraction for Least Cost Search
```text
LeastCostSearch(root)
1. Create an empty priority queue PQ
2. Insert root into PQ
3. while PQ is not empty do
4.      E = delete node with minimum cost from PQ
5.      if E is answer node then
6.           return E
7.      generate all children of E
8.      for each child x of E do
9.           compute cost/bound of x
10.          if x is promising then
11.               insert x into PQ
12. end while
```

#### 4) Explanation of control abstraction
- Step 1: start with root node
- Step 2: put it in priority queue
- Step 3: repeatedly select node with least cost
- Step 4: if it is the solution, stop
- Step 5: otherwise expand it and insert promising children

#### 5) Why is it called “least cost”?
Because we always select the node with **minimum cost** among currently available live nodes.

This increases the chance of reaching the optimal solution faster.

---
### 📊 Visualization
#### State Space View

```text
                Root
              /  |  \
            A    B   C
           / \      / \
          D   E    F   G
```

Suppose costs are:
- A = 12
- B = 8
- C = 10

Least cost search will expand:
- first B,
- then C,
- then A

because it always chooses the lowest cost live node.

#### Priority Queue Idea
```text
Live Nodes: [B:8, C:10, A:12]
Expand B first
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Branch and Bound is an optimization technique     ║
║ that explores a state space tree and prunes       ║
║ branches that cannot give a better solution.      ║
║                                                    ║
║ In Least Cost Search, the live node with minimum  ║
║ cost is selected first using a priority queue.    ║
║ The standard control abstraction uses:            ║
║ generate root -> insert into PQ -> remove least   ║
║ cost node -> expand children -> insert promising  ║
║ children -> repeat until solution is found.       ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Use the keywords **live node, dead node, E-node, priority queue, least cost node**. These words are very important in branch and bound theory answers.

---
<!-- END OF QUESTION 8 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 9 of 126    │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q5(a)              │
│ ✅ Questions Done: 8            │
│ 📋 Questions Remaining: 117     │
└─────────────────────────────────┘

**## ✏️ Question 9 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 9

---
### ❓ Full Question
What is amortized analysis? Explain aggregate and potential function methods used for amortized analysis with respect to stack operations.

---
### 📌 What is this question about?
Amortized analysis studies the **average cost per operation over a whole sequence of operations**. Some individual operations may be costly, but when we spread that total cost across many operations, the average cost may still be small.

It is like buying a yearly bus pass: one payment may look big, but the cost per trip becomes small if you travel many times.

---
### 📖 Formula & What Each Symbol Means
For the **potential method**:

┌──────────────────────────────────────────────────────────┐
│ Amortized Cost = Actual Cost + (Φ_after - Φ_before)     │
└──────────────────────────────────────────────────────────┘

**Where:**
- `Actual Cost` = real work done in that operation
- `Φ_before` = potential before operation
- `Φ_after` = potential after operation
- `ΔΦ = Φ_after - Φ_before`

For stack, we choose:
- `Φ = number of elements in stack`

---
### 🔢 Step-by-Step Solution

#### 1) What is Amortized Analysis?
Amortized analysis does **not** find average over many different inputs. Instead, it finds average cost over a **sequence of operations on the same data structure**.

It guarantees that even if one operation is expensive, the average cost per operation in the whole sequence stays low.

#### 2) Stack Operations Considered
We consider three operations:
- **PUSH(x)** → insert x at top
- **POP()** → remove top element
- **MULTIPOP(k)** → remove top k elements, or all if stack has fewer than k elements

#### 3) Aggregate Method
In aggregate analysis, we find the total cost of a sequence of operations and then divide by the number of operations.

##### Key idea for stack
- Every element can be pushed **once**.
- Every element can be popped **at most once**.
- Even in MULTIPOP, each popped element is removed only one time.

Suppose there are `n` PUSH operations in the whole sequence.
Then:
- total number of pops done by POP and MULTIPOP together can never exceed `n`

So total actual cost of all operations is at most:
- `n` for all pushes
- `n` for all pops
- total = `2n`

Hence amortized cost per operation is:
- `2n / n = 2 = O(1)`

So under aggregate analysis:
- **amortized cost of each stack operation is O(1)**

#### 4) Potential Function Method
Choose potential function:
- `Φ(S) = number of items in stack`

This means:
- each element stored in stack is like stored energy for future pop operation.

##### PUSH(x)
- Actual cost = 1
- Stack size increases by 1
- `ΔΦ = +1`
- Amortized cost = `1 + 1 = 2`

##### POP()
- Actual cost = 1
- Stack size decreases by 1
- `ΔΦ = -1`
- Amortized cost = `1 + (-1) = 0`

##### MULTIPOP(k)
Suppose it actually removes `t` elements.
- Actual cost = `t`
- Stack size decreases by `t`
- `ΔΦ = -t`
- Amortized cost = `t + (-t) = 0`

So under potential method:
- PUSH amortized cost = **2**
- POP amortized cost = **0**
- MULTIPOP amortized cost = **0**

Still, all are **O(1)** amortized.

---
### 📊 Visualization
#### Aggregate Method Summary

| Operation Type | Maximum Total Cost in Whole Sequence |
|----------------|--------------------------------------|
| All PUSH | n |
| All POP / MULTIPOP removals together | n |
| Total | 2n |

Amortized cost per operation = `2n / n = 2 = O(1)`

#### Potential Method Table

| Operation | Actual Cost | Φ Before | Φ After | ΔΦ | Amortized Cost |
|-----------|-------------|----------|---------|----|----------------|
| PUSH | 1 | s | s+1 | +1 | 1 + 1 = 2 |
| POP | 1 | s | s-1 | -1 | 1 - 1 = 0 |
| MULTIPOP(t) | t | s | s-t | -t | t - t = 0 |

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Amortized analysis gives the average cost per     ║
║ operation over a sequence of operations.          ║
║                                                    ║
║ Using aggregate method for stack operations,      ║
║ total cost of n pushes and at most n pops is 2n,  ║
║ so amortized cost per operation is O(1).          ║
║                                                    ║
║ Using potential method with Φ = number of stack   ║
║ elements:                                         ║
║ PUSH = 2, POP = 0, MULTIPOP = 0                   ║
║ Therefore all stack operations have O(1)          ║
║ amortized cost.                                   ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
In amortized questions, always write the line: **“Each element is pushed once and popped at most once.”** This single line earns important marks in stack-based analysis.

---
<!-- END OF QUESTION 9 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 10 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q5(b)              │
│ ✅ Questions Done: 9            │
│ 📋 Questions Remaining: 116     │
└─────────────────────────────────┘

**## ✏️ Question 10 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 9

---
### ❓ Full Question
What is potential function method of amortized analysis? To illustrate potential method, find amortized cost of PUSH, POP and MULTIPOP stack operations.

---
### 📌 What is this question about?
This question asks specifically about the **potential function method**, one of the three standard ways to do amortized analysis. The main idea is to store some “credit” in the data structure so that future expensive operations can be paid for later.

For a stack, the number of stored elements itself acts like saved energy.

---
### 📖 Formula & What Each Symbol Means
The potential method uses the formula:

┌──────────────────────────────────────────────────────────┐
│ Amortized Cost = Actual Cost + ΔΦ                       │
│ where ΔΦ = Φ_after - Φ_before                           │
└──────────────────────────────────────────────────────────┘

For stack operations choose:

┌──────────────────────────────────────────────────────────┐
│ Φ(S) = number of elements currently in stack            │
└──────────────────────────────────────────────────────────┘

**Symbols:**
- `Φ(S)` = potential of stack state `S`
- `Actual Cost` = true cost of operation
- `ΔΦ` = change in potential

---
### 🔢 Step-by-Step Solution

#### 1) What is Potential Function Method?
In potential function method, we assign a non-negative value called **potential** to the current state of the data structure.

- If potential increases, we are storing credit.
- If potential decreases, we are using stored credit.

#### 2) Stack Operations
Let current stack size be `s`.

##### Operation 1: PUSH(x)
- Actual cost = 1
- Before PUSH, size = `s`
- After PUSH, size = `s + 1`

So,
- `Φ_before = s`
- `Φ_after = s + 1`
- `ΔΦ = (s + 1) - s = 1`

Amortized cost:
- `Actual Cost + ΔΦ = 1 + 1 = 2`

So,
- **Amortized cost of PUSH = 2**

##### Operation 2: POP()
- Actual cost = 1
- Before POP, size = `s`
- After POP, size = `s - 1`

So,
- `Φ_before = s`
- `Φ_after = s - 1`
- `ΔΦ = (s - 1) - s = -1`

Amortized cost:
- `1 + (-1) = 0`

So,
- **Amortized cost of POP = 0**

##### Operation 3: MULTIPOP(k)
Suppose `t` elements are actually removed.
Then:
- Actual cost = `t`
- Before operation, size = `s`
- After operation, size = `s - t`

So,
- `Φ_before = s`
- `Φ_after = s - t`
- `ΔΦ = (s - t) - s = -t`

Amortized cost:
- `t + (-t) = 0`

So,
- **Amortized cost of MULTIPOP = 0**

#### 3) Sample Operation Table
Suppose stack starts empty.

| # | Operation | Actual Cost | Φ Before | Φ After | ΔΦ | Amortized Cost |
|---|-----------|-------------|----------|---------|----|----------------|
| 1 | PUSH(a) | 1 | 0 | 1 | +1 | 2 |
| 2 | PUSH(b) | 1 | 1 | 2 | +1 | 2 |
| 3 | POP | 1 | 2 | 1 | -1 | 0 |
| 4 | PUSH(c) | 1 | 1 | 2 | +1 | 2 |
| 5 | MULTIPOP(2) | 2 | 2 | 0 | -2 | 0 |

This confirms the method.

---
### 📊 Visualization
```text
Potential Φ = number of items in stack

PUSH      -> stack grows -> Φ increases -> store credit
POP       -> stack shrinks -> Φ decreases -> use credit
MULTIPOP  -> many items removed -> Φ drops sharply -> old credit pays cost
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Potential function method uses the formula:       ║
║ Amortized Cost = Actual Cost + (Φ_after - Φ_before)║
║                                                    ║
║ For a stack, choose Φ = number of elements.       ║
║ Then:                                              ║
║ PUSH amortized cost = 2                           ║
║ POP amortized cost = 0                            ║
║ MULTIPOP amortized cost = 0                       ║
║ Hence all stack operations are O(1) amortized.    ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Do not forget to write the chosen potential function explicitly: **Φ(S) = number of elements in stack**. Without this, the answer looks incomplete.

---
<!-- END OF QUESTION 10 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 11 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q6(a)              │
│ ✅ Questions Done: 10           │
│ 📋 Questions Remaining: 115     │
└─────────────────────────────────┘

**## ✏️ Question 11 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 10

---
### ❓ Full Question
Write short notes on the following:
1. Aggregate analysis
2. Accounting Analysis
3. Potential function method
4. Tractable and Non-tractable problems

---
### 📌 What is this question about?
This is a theory question covering the main methods of amortized analysis and one important complexity classification topic. The aim is to explain each concept in short, clear, exam-friendly form.

---
### 📖 Formula & What Each Symbol Means
Important formula for potential method:

`Amortized Cost = Actual Cost + (Φ_after - Φ_before)`

No other special formula is necessary.

---
### 🔢 Step-by-Step Solution

#### 1) Aggregate Analysis
Aggregate analysis finds the **total cost of a sequence of operations** and then divides it by the number of operations.

**Main idea:**
- do not study one operation separately,
- study the whole sequence together.

**Example:** Stack operations
- each element is pushed once
- each element is popped at most once
- so total cost of `n` operations is `O(n)`
- hence amortized cost per operation is `O(1)`

**Advantage:**
- simple and easy to understand

**Limitation:**
- does not give cost for each individual operation type separately

---
#### 2) Accounting Analysis
In accounting analysis, we assign an **artificial charge** to each operation.

- some operations are charged more than their actual cost,
- extra amount is stored as **credit**,
- this credit is later used to pay for expensive operations.

**Example:** Stack
- charge PUSH = 2
- actual cost of PUSH = 1
- extra 1 credit stays with the item
- when the item is popped later, that stored credit pays for it

So average cost remains small.

**Advantage:**
- gives an intuitive “credit system” explanation

---
#### 3) Potential Function Method
Potential method stores the prepaid work in a mathematical form called **potential**.

**Formula:**
- `Amortized Cost = Actual Cost + ΔΦ`

For stack:
- choose `Φ = number of elements in stack`

Then:
- PUSH increases potential
- POP decreases potential
- the decrease helps pay for future expensive work

**Advantage:**
- mathematically clean and powerful
- useful for advanced data structures

---
#### 4) Tractable and Non-tractable Problems

##### Tractable Problems
A problem is **tractable** if it can be solved in **polynomial time**, such as:
- `O(n)`
- `O(n log n)`
- `O(n²)`

**Examples:**
- Searching in a sorted list using binary search
- Minimum spanning tree
- Shortest path in many cases

##### Non-tractable Problems
A problem is **non-tractable** if no polynomial-time algorithm is known for it, and it usually needs very large time such as:
- exponential time
- factorial time

**Examples:**
- Traveling Salesman Problem
- 0/1 Knapsack in its general form
- SAT problem

**Simple difference:**
- tractable = practical to solve efficiently
- non-tractable = becomes very hard as input grows

---
### 📊 Visualization
| Topic | Main Idea | Example |
|------|-----------|---------|
| Aggregate Analysis | Total cost / number of operations | Stack operations |
| Accounting Analysis | Charge extra and save credit | PUSH charged 2 |
| Potential Method | Use mathematical stored energy | Φ = stack size |
| Tractable Problem | Polynomial-time solvable | MST |
| Non-tractable Problem | No known polynomial-time solution | TSP |

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Aggregate analysis studies total sequence cost.   ║
║ Accounting analysis uses stored credit.           ║
║ Potential method uses a potential function Φ.     ║
║ Tractable problems have polynomial-time           ║
║ algorithms, while non-tractable problems do not   ║
║ have known efficient polynomial-time solutions.   ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
For “short notes”, use **small headings + 3 to 4 points + one example**. That format looks neat and fetches better marks than one long paragraph.

---
<!-- END OF QUESTION 11 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 12 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q6(b)              │
│ ✅ Questions Done: 11           │
│ 📋 Questions Remaining: 114     │
└─────────────────────────────────┘

**## ✏️ Question 12 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 8

---
### ❓ Full Question
Write short notes on with suitable example of each:
1. Randomized algorithm
2. Approximation algorithm

---
### 📌 What is this question about?
This question asks about two important algorithm types used when exact or deterministic methods are difficult. One uses **random choices**, and the other gives a **near-optimal answer** quickly.

---
### 📖 Formula & What Each Symbol Means
No fixed formula is needed. This is a theory-based answer.

---
### 🔢 Step-by-Step Solution

#### 1) Randomized Algorithm
A **randomized algorithm** uses random numbers or random choices during execution.

So even for the same input, two runs may behave differently.

**Why use randomized algorithms?**
- simple to design
- often good average performance
- avoid bad fixed input patterns
- useful when deterministic algorithm is complex

**Types:**
- **Las Vegas algorithm**: always gives correct answer, but running time may vary
- **Monte Carlo algorithm**: running time may be fixed, but answer may have small probability of error

**Example: Randomized Quick Sort**
- choose pivot randomly
- expected time complexity becomes `O(n log n)`
- worst case is still `O(n²)`, but random pivot makes worst case less likely

**Real-life analogy:**
If a teacher wants to form fair teams, picking random captains reduces bias.

---
#### 2) Approximation Algorithm
An **approximation algorithm** is used for difficult optimization problems where finding the exact best answer may take too much time.

It gives a solution that is:
- close to optimal,
- fast to compute,
- practically useful.

**Why use approximation algorithms?**
- many optimization problems are NP-hard
- exact algorithms may be too slow for large inputs
- approximate solutions are often good enough in practice

**Example: Vertex Cover Approximation**
- choose an uncovered edge `(u,v)`
- add both `u` and `v` to vertex cover
- remove all covered edges
- repeat

This gives a solution within a factor of 2 of the optimal.

Another common example:
- nearest-neighbour idea for Traveling Salesman gives a quick near solution, though not always optimal

**Main benefit:**
- fast and useful when exact solution is very expensive

---
### 📊 Visualization
| Type | Main Idea | Example | Output Nature |
|------|-----------|---------|---------------|
| Randomized Algorithm | Uses random choices | Randomized Quick Sort | Usually exact, performance depends on randomness |
| Approximation Algorithm | Gives near-best answer | Vertex Cover Approximation | Not exact, but close to best |

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ A randomized algorithm uses random decisions      ║
║ during execution. Example: Randomized Quick Sort. ║
║                                                    ║
║ An approximation algorithm gives a near-optimal   ║
║ solution quickly for hard optimization problems.  ║
║ Example: Approximation algorithm for Vertex Cover.║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
For each short note, always include **definition + one advantage + one example**. That is usually enough to secure most of the marks.

---
<!-- END OF QUESTION 12 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 13 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q7(a)              │
│ ✅ Questions Done: 12           │
│ 📋 Questions Remaining: 113     │
└─────────────────────────────────┘

**## ✏️ Question 13 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 9

---
### ❓ Full Question
Write and explain pseudo code for multi-threaded merge sort algorithm. How parallel merging gives a significant parallelism advantage over merge Sort?

---
### 📌 What is this question about?
This question is about running merge sort using multiple processors or threads so that different parts of the array are sorted at the same time. This makes the algorithm faster on multicore systems.

It is like asking two students to sort two halves of a notebook at the same time instead of one student doing everything alone.

---
### 📖 Formula & What Each Symbol Means
Important performance measures in multithreaded algorithms:
- **Work (T1)** = total time on one processor
- **Span (T∞)** = time on infinitely many processors
- **Parallelism** = `T1 / T∞`

More parallelism means better use of multiple processors.

---
### 🔢 Step-by-Step Solution

#### 1) Idea of Multithreaded Merge Sort
- Divide the array into two halves
- Sort left half and right half in parallel
- Merge the two sorted halves

#### 2) Pseudo Code
```text
P-MERGE-SORT(A, p, r)
1. if p >= r
2.      return
3. q = floor((p + r)/2)
4. spawn P-MERGE-SORT(A, p, q)
5. spawn P-MERGE-SORT(A, q+1, r)
6. sync
7. P-MERGE(A, p, q, r)
```

Here:
- `spawn` means start a parallel subtask
- `sync` means wait until all spawned tasks complete
- `P-MERGE` means merge in parallel

#### 3) Parallel Merge Idea
A simple parallel merge works like this:
- choose middle element from one array
- find its correct position in the other sorted array using binary search
- place that element in final array
- recursively merge left parts and right parts in parallel

Pseudo code idea:
```text
P-MERGE(X, Y, Z)
1. if one array is empty
2.      copy the other array into Z
3. choose middle element of larger array
4. find its position in other array by binary search
5. place it in correct position in Z
6. spawn merge of left parts
7. spawn merge of right parts
8. sync
```

#### 4) Why multithreaded merge sort is better
In ordinary merge sort:
- recursive sorting of two halves can be parallelized,
- but merge step is often sequential.

This sequential merge becomes a bottleneck.

With **parallel merging**:
- sorting of halves is parallel,
- merging is also parallel,
- so the whole algorithm has much better parallelism.

#### 5) Advantage of Parallel Merging
If merge is sequential:
- span contains a large `O(n)` merge cost at each level
- overall parallelism is limited

If merge is parallel:
- span of merge becomes much smaller than linear
- more processors can work simultaneously
- performance improves on large inputs

So parallel merging removes the biggest bottleneck of merge sort.

---
### 📊 Visualization
```text
                 Array A
                    |
          ----------------------
          |                    |
      Left Half            Right Half
          |                    |
   sort in parallel      sort in parallel
          |                    |
          -------- parallel merge --------
                         |
                    Sorted Array
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Multithreaded merge sort divides the array into   ║
║ two halves, sorts both halves in parallel using   ║
║ spawn, waits using sync, and then merges them.    ║
║                                                    ║
║ Parallel merging gives a big advantage because    ║
║ it removes the sequential merge bottleneck and    ║
║ increases overall parallelism.                    ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Write both **pseudo code** and the sentence **“parallel merge removes the sequential bottleneck”**. That sentence is usually the key theoretical mark.

---
<!-- END OF QUESTION 13 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 14 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q7(b)(i)           │
│ ✅ Questions Done: 13           │
│ 📋 Questions Remaining: 112     │
└─────────────────────────────────┘

**## ✏️ Question 14 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q7(b)(i)
**⭐ Marks:** 4

---
### ❓ Full Question
Explain an algorithm for Distributed Minimum Spanning Tree.

---
### 📌 What is this question about?
A **Distributed Minimum Spanning Tree (DMST)** algorithm finds the minimum spanning tree of a graph when the graph is spread across different processors or nodes in a network. No single node knows the entire graph at the beginning.

This is useful in computer networks where each machine knows only its neighbours.

---
### 📖 Formula & What Each Symbol Means
No single formula is required. The standard idea is based on the **GHS algorithm** (Gallager-Humblet-Spira).

Important terms:
- **Fragment** = a partial tree
- **MOE** = Minimum Outgoing Edge of a fragment
- **Level** = stage number of a fragment
- **Core edge** = edge through which fragments merge

---
### 🔢 Step-by-Step Solution

#### 1) Basic idea
Initially:
- each node is its own fragment
- so there are many tiny fragments

Then repeatedly:
1. each fragment finds its **minimum outgoing edge**,
2. fragments connected by these minimum edges merge,
3. larger fragments are formed,
4. process repeats until only one fragment remains.

That final fragment is the **minimum spanning tree**.

#### 2) Steps of the distributed MST algorithm
1. Start with every node as a separate fragment.
2. Mark each fragment with level 0.
3. Each fragment finds its MOE.
4. Send messages along that MOE to connect to another fragment.
5. Merge two fragments.
6. Update fragment level and fragment identity.
7. Repeat until all nodes belong to one fragment.

#### 3) Why it works
At every stage, choosing the minimum outgoing edge of a fragment is safe because of the **cut property** of MST.

The cut property says:
- the lightest edge crossing a cut can safely belong to an MST.

---
### 📊 Visualization
```text
Initially:
A   B   C   D   E   (all separate fragments)

After choosing MOEs:
(A-B)   (C-D)   E

After next merge:
(A-B-C-D-E)

Final result: one spanning tree with minimum total cost
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ In Distributed MST, each node first acts as a     ║
║ separate fragment. Each fragment finds its        ║
║ minimum outgoing edge and merges with another     ║
║ fragment through that edge. This process repeats  ║
║ until one fragment remains, which is the MST.     ║
║ A well-known algorithm for this is GHS.           ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Use the terms **fragment**, **minimum outgoing edge (MOE)**, and **merge**. These are the most important keywords in DMST answers.

---
<!-- END OF QUESTION 14 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 15 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q7(b)(ii)          │
│ ✅ Questions Done: 14           │
│ 📋 Questions Remaining: 111     │
└─────────────────────────────────┘

**## ✏️ Question 15 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q7(b)(ii)
**⭐ Marks:** 4

---
### ❓ Full Question
Write and explain Rabin-Karp algorithm for string matching.

---
### 📌 What is this question about?
Rabin-Karp is a string matching algorithm used to find whether a pattern appears inside a larger text. It uses **hash values** so that instead of comparing every character again and again, we first compare numbers.

It is like first checking a quick code on two labels before reading the full names letter by letter.

---
### 📖 Formula & What Each Symbol Means
Main rolling hash idea:

`Hash(next window) = (d*(Hash(current window - leading char*h)) + trailing char) mod q`

**Where:**
- `d` = number base, usually size of character set
- `q` = prime number used for modulo
- `h = d^(m-1) mod q`
- `m` = pattern length
- `n` = text length

---
### 🔢 Step-by-Step Solution

#### 1) Working idea
1. Compute hash of pattern.
2. Compute hash of first window of text of same length.
3. Compare the hashes.
4. If hashes match, compare characters one by one to confirm.
5. Slide the text window by one position and update hash quickly.

#### 2) Pseudo code
```text
RABIN-KARP(T, P, d, q)
1. n = length(T)
2. m = length(P)
3. compute hash(P)
4. compute hash(T[1..m])
5. for s = 0 to n-m do
6.      if hash(P) = hash(T[s+1 .. s+m]) then
7.           compare characters one by one
8.           if all match then report valid match
9.      if s < n-m then
10.          compute next window hash using rolling hash
```

#### 3) Important point: spurious hit
Sometimes:
- pattern hash = window hash
- but actual strings are different

This is called a **spurious hit**.
That is why actual character comparison is still needed after a hash match.

#### 4) Time complexity
- Expected runtime = **O(n + m)**
- Worst-case runtime = **O(nm)**

Worst case happens when many spurious hits occur.

---
### 📊 Visualization
```text
Text:    A B C D A B C D
Pattern:     B C D

Window 1: A B C   -> compare hash
Window 2: B C D   -> hash match -> verify chars -> MATCH
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Rabin-Karp is a string matching algorithm that    ║
║ uses hashing to compare the pattern with each     ║
║ window of the text. If the hash matches, then     ║
║ actual character comparison is done.              ║
║ Expected time = O(n + m)                          ║
║ Worst-case time = O(nm)                           ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
Never forget to mention **spurious hit** and the two complexities: **expected O(n+m)** and **worst-case O(nm)**.

---
<!-- END OF QUESTION 15 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 16 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q8(a)              │
│ ✅ Questions Done: 15           │
│ 📋 Questions Remaining: 110     │
└─────────────────────────────────┘

**## ✏️ Question 16 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 10

---
### ❓ Full Question
Write short notes on the following:
1. Multithreaded matrix multiplication
2. Multithreaded merge sort
3. Distributed breadth first search
4. The Rabin-Karp algorithm

---
### 📌 What is this question about?
This question asks for short notes on four important topics from multithreaded, distributed, and string algorithms. Each note should give the meaning, basic working, and one key benefit.

---
### 📖 Formula & What Each Symbol Means
For matrix multiplication:
- If A is of size `m × n` and B is of size `n × p`, then result C is of size `m × p`
- `C[i][j] = Σ A[i][k] × B[k][j]`

For Rabin-Karp:
- rolling hash is used to compare text windows and pattern quickly.

---
### 🔢 Step-by-Step Solution

#### 1) Multithreaded Matrix Multiplication
In multithreaded matrix multiplication, different threads compute different parts of the result matrix simultaneously.

**Idea:**
- each row or block can be assigned to a separate thread
- since many cells are independent, they can be computed in parallel

**Benefit:**
- faster execution on multicore systems

**Example:**
- thread 1 computes rows 1–2
- thread 2 computes rows 3–4

---
#### 2) Multithreaded Merge Sort
In multithreaded merge sort:
- divide array into halves,
- sort both halves in parallel,
- then merge them.

**Keywords:**
- `spawn` to create parallel tasks
- `sync` to wait for completion

**Benefit:**
- better speed than ordinary merge sort on multiple processors

---
#### 3) Distributed Breadth First Search (DBFS)
Distributed BFS is BFS performed in a distributed system where different processors or machines hold different parts of the graph.

**Working idea:**
- start from source node,
- visit all nodes level by level,
- communicate frontier information across processors.

**Benefit:**
- useful for very large graphs stored across machines

---
#### 4) Rabin-Karp Algorithm
Rabin-Karp is a string matching algorithm based on hashing.

**Working:**
- compute pattern hash,
- compare with hash of each text window,
- if hashes match, do character check.

**Benefit:**
- efficient expected performance,
- useful when matching many patterns or sliding windows

---
### 📊 Visualization
| Topic | Main Idea | Benefit |
|------|-----------|---------|
| Multithreaded Matrix Multiplication | Parallel computation of matrix cells | Faster on multicore CPUs |
| Multithreaded Merge Sort | Sort halves in parallel | Good parallelism |
| Distributed BFS | Level-order graph traversal across machines | Useful for huge graphs |
| Rabin-Karp | Hash-based string matching | Fast expected matching |

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Multithreaded matrix multiplication computes      ║
║ matrix entries in parallel. Multithreaded merge   ║
║ sort sorts subarrays in parallel. Distributed BFS ║
║ performs level-wise graph traversal across        ║
║ multiple machines. Rabin-Karp matches strings     ║
║ using rolling hash values.                        ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
For each short note, write **definition + basic working + one advantage**. This is the safest format for 2 to 3 mark subparts.

---
<!-- END OF QUESTION 16 -->
<!-- =================== -->

---

┌─────────────────────────────────┐
│ 📝 Solving Question 17 of 126   │
│ 📄 Paper: P3147                 │
│ 🔢 Question: Q8(b)              │
│ ✅ Questions Done: 16           │
│ 📋 Questions Remaining: 109     │
└─────────────────────────────────┘

**## ✏️ Question 17 of 126**
**📄 Paper:** [6004]-480A (P3147)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 7

---
### ❓ Full Question
With respect to Multithreaded Algorithms explain Analyzing multithreaded algorithms, Parallel loops, Race conditions.

---
### 📌 What is this question about?
This question asks about three basic ideas used in multithreaded programming: how we measure performance, how loops can run in parallel, and what problems happen when two threads use shared data incorrectly.

These are the fundamentals needed to understand parallel programs safely.

---
### 📖 Formula & What Each Symbol Means
Important measures:
- **Work (T1)** = total running time on one processor
- **Span (T∞)** = longest chain of dependent operations
- **Parallelism** = `T1 / T∞`

---
### 🔢 Step-by-Step Solution

#### 1) Analyzing Multithreaded Algorithms
To study a multithreaded algorithm, we mainly use:

**a) Work (T1)**
- total amount of computation done
- same as running time on one processor

**b) Span (T∞)**
- minimum possible execution time if unlimited processors are available
- also called critical-path length

**c) Parallelism**
- `Parallelism = T1 / T∞`
- larger value means more opportunities for speedup

These measures tell us how well the algorithm can use many processors.

---
#### 2) Parallel Loops
A **parallel loop** is a loop in which different iterations can run simultaneously.

Example:
```text
parallel for i = 1 to n
    C[i] = A[i] + B[i]
```

Why can this run in parallel?
- because each iteration computes a different element,
- one iteration does not depend on another.

**Important condition:**
Loop iterations must be independent.

---
#### 3) Race Conditions
A **race condition** happens when:
- two or more threads access shared data at the same time,
- and at least one thread modifies it,
- causing an unpredictable result.

Example:
```text
x = 0
Thread 1: x = x + 1
Thread 2: x = x + 1
```

Expected result may be 2, but due to overlapping execution, result may become 1.

**How to avoid race conditions?**
- locks / mutexes
- atomic operations
- proper synchronization
- careful design of shared memory access

---
### 📊 Visualization
```text
Work      -> total computation
Span      -> longest dependent path
Parallelism = Work / Span

Parallel loop:
Iteration 1  ||  Iteration 2  ||  Iteration 3

Race condition:
Thread 1 ---- modifies x
Thread 2 ---- modifies x
Result may become wrong if not synchronized
```

---
### ✅ Final Answer
╔════════════════════════════════════════════════════╗
║ FINAL ANSWER:                                     ║
║ Multithreaded algorithms are analyzed using work, ║
║ span, and parallelism. Parallel loops run         ║
║ different independent iterations at the same      ║
║ time. A race condition occurs when multiple       ║
║ threads access shared data unsafely, causing      ║
║ unpredictable results.                            ║
╚════════════════════════════════════════════════════╝

---
### 🎯 Marking Tip
For this answer, write at least one formula: **Parallelism = T1 / T∞**. Also give one simple example of a race condition, because examples make the answer much stronger.

---
<!-- END OF QUESTION 17 -->
<!-- =================== -->

┌──────────────────────────────────────────────┐
│ ✅ Paper 1 Complete                          │
│ 📎 Answers appended to answer1.md            │
│ 📚 File now contains 17 solved entries       │
│ ⏭️  Next planned: Paper 2 - Q1(a)            │
└──────────────────────────────────────────────┘
