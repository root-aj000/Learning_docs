═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 5 Answers — [6181]-101 (P-6551)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 5**.
> New answers will always be added at the bottom.

---

## ✏️ Question 71 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 9

### ❓ Full Question
Write high-level description of job sequencing algorithm. Let n=5, profit vector P={20,15,10,5,1}, deadline vector D={2,2,1,3,3}. Find feasible solutions. What is the optimal solution and maximum profit?

### ✅ Answer
Let jobs be `J1, J2, J3, J4, J5`.

| Job | Profit | Deadline |
|-----|--------|----------|
| J1 | 20 | 2 |
| J2 | 15 | 2 |
| J3 | 10 | 1 |
| J4 | 5 | 3 |
| J5 | 1 | 3 |

### High-Level Description
1. Sort jobs in decreasing order of profit.
2. Find maximum deadline = number of slots.
3. Place each job in the latest free slot before its deadline.
4. Continue until all jobs are checked.

### Sorted Order
Already sorted by profit:
**J1, J2, J3, J4, J5**

Maximum deadline = 3, so slots are 1, 2, 3.

### Some Feasible 3-job Solutions
- `{J1, J2, J4}`
- `{J1, J2, J5}`
- `{J1, J3, J4}`
- `{J2, J3, J4}`
- `{J3, J4, J5}`

### Greedy Scheduling
- J1 goes to slot 2
- J2 goes to slot 1
- J3 cannot be scheduled (slot 1 already full)
- J4 goes to slot 3
- J5 cannot improve profit and latest free slot not available after greedy placement

### Optimal Schedule
Slot-wise schedule:
- Slot 1 → **J2**
- Slot 2 → **J1**
- Slot 3 → **J4**

### Maximum Profit
`15 + 20 + 5 = 40`

### Final Answer
Optimal solution = **{J1, J2, J4}** with maximum profit **40**.

### 🎯 Marking Tip
Mention both: **feasible schedules exist**, but **greedy gives optimal schedule J2, J1, J4**.

---

## ✏️ Question 72 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 9

### ❓ Full Question
Solve the knapsack problem using dynamic programming for items:
1(2,12), 2(1,10), 3(3,20), 4(2,15), capacity = 5.

### ✅ Answer
The DP table gives maximum profit **37**.

### Selected Items
- Item 1 → weight 2, profit 12
- Item 2 → weight 1, profit 10
- Item 4 → weight 2, profit 15

### Total
- Total weight = `2 + 1 + 2 = 5`
- Total profit = `12 + 10 + 15 = 37`

### Final Answer
Optimal set = **{1, 2, 4}**, maximum profit = **37**.

### 🎯 Marking Tip
Write the final selected items after the DP table; otherwise the answer looks incomplete.

---

## ✏️ Question 73 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 12

### ❓ Full Question
What is Job scheduling algorithm? How can it be solved using Greedy approach for:
J1(2,60), J2(1,100), J3(3,20), J4(2,40), J5(1,20)

### ✅ Answer
### Definition
Job scheduling arranges jobs in available slots such that deadlines are satisfied and total profit is maximum.

### Greedy Rule
- Sort jobs by decreasing profit.
- Place each job in the latest possible free slot before its deadline.

### Sorted by Profit
| Job | Deadline | Profit |
|-----|----------|--------|
| J2 | 1 | 100 |
| J1 | 2 | 60 |
| J4 | 2 | 40 |
| J3 | 3 | 20 |
| J5 | 1 | 20 |

Maximum deadline = 3 → slots 1, 2, 3.

### Scheduling
- J2 → slot 1
- J1 → slot 2
- J4 → cannot fit
- J3 → slot 3
- J5 → cannot fit

### Final Sequence
**J2, J1, J3**

### Total Profit
`100 + 60 + 20 = 180`

### Greedy Principle
A high-profit job is chosen first, and placing it as late as possible preserves earlier slots for other jobs.

### Time Complexity
Standard implementation: **O(n²)**

### 🎯 Marking Tip
Write three things: **sorted table**, **slot assignment**, and **final profit 180**.

---

## ✏️ Question 74 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 6

### ❓ Full Question
Write steps for Greedy approach for Job sequencing.

### ✅ Answer
### Steps
1. List all jobs with deadlines and profits.
2. Sort jobs in decreasing order of profit.
3. Find maximum deadline.
4. Create empty slots from 1 to maximum deadline.
5. Pick jobs one by one in sorted order.
6. Place each job in the latest free slot before its deadline.
7. If no slot is free, reject that job.
8. Scheduled jobs give the final answer.

### 🎯 Marking Tip
Number the steps clearly. This type of question is best answered in algorithm steps, not paragraph form.

---

## ✏️ Question 75 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 9

### ❓ Full Question
What is branch and bound algorithmic strategy? Apply branch and bound strategy to solve traveling salesman problem.

### ✅ Answer
### Branch and Bound Strategy
Branch and Bound is an optimization method that:
- branches into possible choices,
- calculates a bound for each partial solution,
- prunes branches that cannot lead to a better result.

### Applying it to TSP
1. Start from a city.
2. Build a state-space tree of tours.
3. For each partial tour, compute a lower bound on total cost.
4. Expand the node with least bound first.
5. Prune nodes whose bound is greater than current best tour.
6. Continue until complete minimum-cost tour is found.

### Why Useful?
TSP has many possible tours. Branch and bound avoids checking all of them by pruning poor branches.

### 🎯 Marking Tip
Mention **state-space tree**, **lower bound**, and **pruning**. These are the three most important terms.

---

## ✏️ Question 76 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 8

### ❓ Full Question
Explain with suitable example Backtracking: Principle, control abstraction, time analysis of control abstraction.

### ✅ Answer
Backtracking tries a solution step by step and stops exploring a branch as soon as it becomes invalid.

### Principle
- choose
- test promising condition
- continue or backtrack

### Control Abstraction
```text
BACKTRACK(k)
1. if complete solution then print
2. else
3.      generate candidates
4.      for each candidate do
5.           if promising then recurse
```

### Example
Graph coloring or sum of subsets.

### Time Complexity
Worst-case is exponential, usually `O(2^n)` or `O(m^n)` depending on the problem.

### 🎯 Marking Tip
Always include the word **promising** in backtracking answers.

---

## ✏️ Question 77 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 9

### ❓ Full Question
Use branch and bound to solve 0/1 knapsack of capacity 15 with items A(9,18), B(4,10), C(6,12), D(2,10).

### ✅ Answer
Check best feasible sets:
- A+B+D → weight 15, profit 38 ✅
- B+C+D → weight 12, profit 32
- A+C → weight 15, profit 30
- A+D → weight 11, profit 28

### Optimal Set
**A, B and D**

### Final Result
- Weight = **15**
- Profit = **38**

### 🎯 Marking Tip
State the final combination in one clear line: **“Best feasible solution is A+B+D with profit 38.”**

---

## ✏️ Question 78 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 8

### ❓ Full Question
Solve sum of subset problem for set {2,3,5,6,8,10}, sum = 10 using backtracking.

### ✅ Answer
Valid subsets whose sum is 10 are:
- `{2,3,5}`
- `{2,8}`
- `{10}`

### Backtracking Idea
At each element, either include it or exclude it.
Prune any branch when current sum becomes greater than 10.

### 🎯 Marking Tip
Write all valid subsets, not only one subset.

---

## ✏️ Question 79 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 9

### ❓ Full Question
What is amortized analysis? Explain the aggregate method with example.

### ✅ Answer
Amortized analysis studies the average cost per operation over a sequence of operations.

### Aggregate Method
- Find total cost of all operations together.
- Divide by number of operations.

### Example: Stack
If there are `n` pushes, total pops over the whole sequence can be at most `n`.
So total cost ≤ `2n`.
Hence amortized cost per operation is `2n/n = O(1)`.

### 🎯 Marking Tip
Use the exact sentence: **“Each element is pushed once and popped at most once.”**

---

## ✏️ Question 80 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 9

### ❓ Full Question
Explain potential function method of amortized analysis. Find amortized cost of PUSH, POP and MULTIPOP.

### ✅ Answer
Choose potential function:
`Φ = number of elements in stack`

Then:
- PUSH: actual 1, ΔΦ = +1 → amortized = 2
- POP: actual 1, ΔΦ = -1 → amortized = 0
- MULTIPOP removing `t` items: actual `t`, ΔΦ = `-t` → amortized = 0

### Result
- PUSH = **2**
- POP = **0**
- MULTIPOP = **0**

All are **O(1)** amortized.

### 🎯 Marking Tip
Write the formula too: `Amortized Cost = Actual Cost + ΔΦ`.

---

## ✏️ Question 81 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 6

### ❓ Full Question
What are special needs of embedded algorithm? Which sorting algorithm is best for embedded systems? Why?

### ✅ Answer
### Special Needs of Embedded Algorithms
- low memory usage
- low power consumption
- predictable execution time
- simple code
- high reliability

### Best Sorting Algorithm
For many embedded applications, **Insertion Sort** is a good choice.

### Why?
- in-place
- simple to implement
- good for small datasets
- works well for nearly sorted data
- very low overhead

### 🎯 Marking Tip
Use the words **memory**, **power**, and **predictable timing**.

---

## ✏️ Question 82 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 4

### ❓ Full Question
Explain Randomized and Approximate algorithms.

### ✅ Answer
### Randomized Algorithm
Uses random choices during execution.
Example: randomized quick sort.

### Approximation Algorithm
Gives a near-optimal solution in polynomial time for hard optimization problems.
Example: vertex cover approximation.

### 🎯 Marking Tip
Two definitions with one example each are enough here.

---

## ✏️ Question 83 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q6(c)
**⭐ Marks:** 8

### ❓ Full Question
What is randomized algorithm? Give one example. Also explain random variable, binomial random variable and mathematics for randomized algorithm.

### ✅ Answer
### Randomized Algorithm
An algorithm that uses random choices while running.

### Example
**Randomized Quick Sort** selects pivot randomly.

### Random Variable
A random variable is a variable whose value depends on the outcome of a random process.
Example: number of heads in coin tosses.

### Binomial Random Variable
If an experiment has:
- fixed number of trials `n`
- each trial has success probability `p`
then the number of successes follows binomial distribution.

Notation:
`X ~ Bin(n, p)`

### Mathematics in Randomized Algorithms
We often use:
- probability
- expectation
- linearity of expectation

These help in analyzing expected running time and expected correctness.

### 🎯 Marking Tip
Write one line with notation: **X ~ Bin(n,p)**.

---

## ✏️ Question 84 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q7(a)(i)
**⭐ Marks:** 5

### ❓ Full Question
Explain an algorithm for Distributed Minimum Spanning Tree.

### ✅ Answer
In a Distributed MST algorithm:
1. Each node starts as a separate fragment.
2. Every fragment finds its minimum outgoing edge (MOE).
3. Fragments connected by MOE merge.
4. The process repeats until only one fragment remains.

This final fragment is the minimum spanning tree.

### Keyword
This idea is used in the **GHS algorithm**.

### 🎯 Marking Tip
Use the words **fragment**, **MOE**, and **merge**.

---

## ✏️ Question 85 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q7(a)(ii)
**⭐ Marks:** 5

### ❓ Full Question
Write and explain Rabin-Karp algorithm for string matching.

### ✅ Answer
Rabin-Karp compares the pattern with each text window using hash values.

### Steps
1. Compute hash of pattern.
2. Compute hash of first text window.
3. Compare hashes.
4. If hashes match, compare characters.
5. Roll the hash to next window.

### Complexity
- Expected = `O(n+m)`
- Worst case = `O(nm)`

### 🎯 Marking Tip
Do not forget to mention **rolling hash**.

---

## ✏️ Question 86 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 7

### ❓ Full Question
With respect to multithreaded algorithms explain analyzing multithreaded algorithms, parallel loops, race conditions.

### ✅ Answer
- **Analyzing multithreaded algorithms** uses work `T1`, span `T∞`, and parallelism `T1/T∞`.
- **Parallel loops** run independent loop iterations at the same time.
- **Race conditions** occur when multiple threads access shared data unsafely.

### Example Race Condition
Two threads updating the same variable `x` may produce wrong output if not synchronized.

### 🎯 Marking Tip
Write the formula `Parallelism = T1/T∞`.

---

## ✏️ Question 87 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 9

### ❓ Full Question
Write and explain pseudo code for multithreaded merge sort algorithm. How parallel merging gives a significant parallelism advantage over merge sort?

### ✅ Answer
### Pseudo Code
```text
P-MERGE-SORT(A, p, r)
1. if p >= r return
2. q = floor((p+r)/2)
3. spawn P-MERGE-SORT(A, p, q)
4. spawn P-MERGE-SORT(A, q+1, r)
5. sync
6. P-MERGE(A, p, q, r)
```

### Advantage of Parallel Merging
If merge is sequential, it becomes a bottleneck.
If merge is parallel too, then:
- span decreases,
- speedup improves,
- more processors can work simultaneously.

### 🎯 Marking Tip
Write: **“Parallel merging reduces the sequential bottleneck.”**

---

## ✏️ Question 88 of 126
**📄 Paper:** [6181]-101 (P-6551)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 8

### ❓ Full Question
For Rabin-Karp matcher with q = 11, text T = 31415926535 and pattern P = 26, how many spurious hits occur? Find exact match of P mod q.

### ✅ Answer
Pattern length = 2
Pattern = `26`

### Pattern Hash
`26 mod 11 = 4`

### Text Windows of Length 2
| Window | Value | mod 11 |
|--------|-------|--------|
| 31 | 31 | 9 |
| 14 | 14 | 3 |
| 41 | 41 | 8 |
| 15 | 15 | 4 |
| 59 | 59 | 4 |
| 92 | 92 | 4 |
| 26 | 26 | 4 |
| 65 | 65 | 10 |
| 53 | 53 | 9 |
| 35 | 35 | 2 |

### Hash Matches
Hash = 4 occurs for:
- 15
- 59
- 92
- 26

Only **26** is the true match.

### Therefore
- **Exact match** = window `26`
- **Number of spurious hits** = `4 - 1 = 3`

### Match Position
`26` starts at the **7th digit** of the text (1-based position).

### 🎯 Marking Tip
Write both: **hash matches = 4** and **exact match = 1**. Then spurious hits = 3.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 5 Complete                          │
│ 📎 Answers appended to answer5.md            │
│ 📚 Includes Questions 71 to 88               │
└──────────────────────────────────────────────┘
