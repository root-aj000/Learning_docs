═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 6 Answers — [6354]-485 (PC-2368)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 6**.
> New answers will always be added at the bottom.

---

## ✏️ Question 89 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 8

### ❓ Full Question
Design a greedy algorithm to schedule tasks with deadlines and penalties so that total penalty is minimized. Prove its correctness.

### ✅ Answer
### Idea
Minimizing total penalty is the same as **maximizing total saved penalty**.
So we treat penalty like profit and use job sequencing.

### Greedy Algorithm
1. Sort tasks in decreasing order of penalty.
2. Create slots up to maximum deadline.
3. Place each task in the latest free slot before its deadline.
4. Tasks that cannot be placed are left late, and their penalties are paid.

### Why It Works
- A task with high penalty should be finished on time if possible.
- Putting it in the latest free slot keeps earlier slots open for other tasks.
- By exchange argument, if a smaller-penalty task is placed instead of a bigger-penalty task, swapping them cannot increase saved penalty.

### Conclusion
Thus the greedy choice is correct and minimizes total penalty.

### 🎯 Marking Tip
Write: **“Minimizing penalty = maximizing saved penalty.”** That is the key transformation.

---

## ✏️ Question 90 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 8

### ❓ Full Question
Knapsack capacity = 15. Objects: O1(8,10), O2(6,8), O3(4,3), O4(2,4). Use greedy approach to maximize total value.

### ✅ Answer
This is solved as **fractional knapsack**.

### Profit/Weight Ratios
| Object | Weight | Value | Ratio |
|--------|--------|-------|-------|
| O1 | 8 | 10 | 1.25 |
| O2 | 6 | 8 | 1.33 |
| O3 | 4 | 3 | 0.75 |
| O4 | 2 | 4 | 2.00 |

### Sorted Order
**O4, O2, O1, O3**

### Fill the Knapsack
- Take O4 fully → weight 2, value 4, remaining = 13
- Take O2 fully → weight 6, value 8, remaining = 7
- O1 weight = 8, remaining = 7 → take `7/8` of O1
  - value added = `10 × 7/8 = 8.75`

### Total Value
`4 + 8 + 8.75 = 20.75`

### Final Answer
Maximum greedy value = **20.75**

### 🎯 Marking Tip
If you use greedy knapsack, always compute the **ratio table** first.

---

## ✏️ Question 91 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q1(c)
**⭐ Marks:** 2

### ❓ Full Question
With respect to dynamic programming, what do you understand by optimal substructure?

### ✅ Answer
A problem has **optimal substructure** if its optimal solution can be built from optimal solutions of smaller subproblems.

### Example
In matrix chain multiplication, the best way to multiply the whole chain depends on the best way to multiply smaller parts of the chain.

### 🎯 Marking Tip
Write one sentence definition and one small example.

---

## ✏️ Question 92 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 10

### ❓ Full Question
For matrices of sizes 4×10, 10×3, 3×12, 12×20, 20×7, solve chain matrix multiplication using dynamic programming.

### ✅ Answer
Dimension array:
`p = [4, 10, 3, 12, 20, 7]`

Using dynamic programming, minimum cost is:

**1344 scalar multiplications**

### Optimal Parenthesization
**((A1A2)((A3A4)A5))**

### Key Costs
- `m[1,2] = 120`
- `m[3,4] = 720`
- `m[3,5] = 1140`
- `m[1,5] = 1344`

### 🎯 Marking Tip
Write both the **minimum cost = 1344** and the **optimal parenthesization**.

---

## ✏️ Question 93 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 4

### ❓ Full Question
Under what situation might dynamic programming for knapsack struggle to find the optimal solution? Briefly explain.

### ✅ Answer
Dynamic programming for knapsack may struggle when:
1. **Capacity is very large** – table size becomes huge.
2. **Weights/profits are large integers** – memory and time increase a lot.
3. **Weights are non-integer / real values** – standard table-based DP is not directly suitable.
4. The method becomes **pseudo-polynomial**, so it may be slow for big inputs.

### 🎯 Marking Tip
Write the phrase **“DP knapsack is pseudo-polynomial”**.

---

## ✏️ Question 94 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q2(c)
**⭐ Marks:** 4

### ❓ Full Question
Enlist the uses of writing control abstraction for any algorithmic strategies.

### ✅ Answer
Uses of control abstraction:
1. Shows the general structure of the strategy.
2. Separates logic from problem-specific details.
3. Makes algorithm design easier.
4. Helps compare different strategies.
5. Useful for teaching and understanding recursion/selection steps.
6. Makes time analysis easier.

### 🎯 Marking Tip
For this question, list points directly. No long explanation is needed.

---

## ✏️ Question 95 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 8

### ❓ Full Question
Use recursive backtracking algorithm to color the given graph with three colours R, G, B.

### ✅ Answer
A valid coloring is:
- `A = R`
- `B = G`
- `C = B`
- `D = R`
- `E = R`
- `F = R`
- `G = R`

### Why valid?
- A is adjacent to B and C, so B and C must differ from A.
- D and E are adjacent only to B, so they may both be R.
- F and G are adjacent only to C, so they may both be R.

### Backtracking Idea
Color vertices one by one and reject any color that matches an adjacent colored vertex.

### 🎯 Marking Tip
It is enough to give **one valid coloring** and mention the backtracking rule.

---

## ✏️ Question 96 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 9

### ❓ Full Question
Solve 0/1 knapsack using LC branch and bound for O1(10,12), O2(8,10), O3(6,8), capacity = 14.

### ✅ Answer
Feasible combinations:
- O1 → 12
- O2 → 10
- O3 → 8
- O1+O2 → overweight
- O1+O3 → overweight
- O2+O3 → weight 14, profit 18 ✅
- O1+O2+O3 → overweight

### Optimal Solution
**O2 + O3**

Weight = `8 + 6 = 14`
Profit = `10 + 8 = 18`

### 🎯 Marking Tip
Show that the combinations with O1 become overweight. Then the best set is obvious.

---

## ✏️ Question 97 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 8

### ❓ Full Question
A salesman must visit A, B, C, D and return to A. Distances: AB=10, AC=15, AD=20, BC=35, BD=25, CD=30. Find shortest route using branch and bound. Start at A.

### ✅ Answer
Possible main tours from A:
- A → B → C → D → A = `10 + 35 + 30 + 20 = 95`
- A → B → D → C → A = `10 + 25 + 30 + 15 = 80`
- A → C → B → D → A = `15 + 35 + 25 + 20 = 95`
- A → C → D → B → A = `15 + 30 + 25 + 10 = 80`
- A → D → B → C → A = `20 + 25 + 35 + 15 = 95`
- A → D → C → B → A = `20 + 30 + 35 + 10 = 95`

### Minimum Cost Tour
**A → B → D → C → A**
(or equivalently **A → C → D → B → A**)

### Minimum Cost
**80**

### 🎯 Marking Tip
Write at least 2–3 full tour costs and then circle the minimum one.

---

## ✏️ Question 98 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 5

### ❓ Full Question
Write a short note on LC branch and bound method.

### ✅ Answer
LC branch and bound means **Least Cost Branch and Bound**.

### Main Idea
- Store live nodes in a priority queue.
- Always expand the live node with the **least cost** (or best promising value).
- Compute bounds for child nodes.
- Prune nodes that cannot improve the current best solution.

### Use
Used in optimization problems like:
- 0/1 knapsack
- TSP
- assignment problem

### 🎯 Marking Tip
Use the phrase **“priority queue of live nodes”**.

---

## ✏️ Question 99 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q4(c)
**⭐ Marks:** 4

### ❓ Full Question
What are the drawbacks of branch and bound method?

### ✅ Answer
Drawbacks:
- may require large memory
- worst-case exponential time
- performance depends on quality of bound
- implementation is relatively complex

### 🎯 Marking Tip
Four short bullet points are enough.

---

## ✏️ Question 100 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 8

### ❓ Full Question
What are the advantages and disadvantages of Aggregate Analysis and Accounting Method?

### ✅ Answer
### Aggregate Analysis
**Advantages:**
- simple
- easy to explain
- good for overall average cost

**Disadvantages:**
- does not explain cost of each operation separately
- less flexible for complex structures

### Accounting Method
**Advantages:**
- intuitive credit system
- explains costly and cheap operations nicely

**Disadvantages:**
- choosing correct charge is not always easy
- credit assignment may become tricky

### 🎯 Marking Tip
Write advantages and disadvantages in a table or separate bullets.

---

## ✏️ Question 101 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 9

### ❓ Full Question
What are approximation algorithms? Based on the approximation ratio, classify them.

### ✅ Answer
Approximation algorithms give near-optimal answers in polynomial time for hard optimization problems.

### Classification by Ratio
- **Exact**: ratio = 1
- **Constant-factor**: ratio = constant
- **Logarithmic-factor**: ratio = `O(log n)`
- **PTAS**: `(1+ε)` approximation for any ε>0
- **FPTAS**: stronger practical version of PTAS

### Example
Vertex cover has a 2-approximation algorithm.

### 🎯 Marking Tip
Do not skip the word **ratio**. That is the central idea of the answer.

---

## ✏️ Question 102 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 8

### ❓ Full Question
Why potential function method cannot be used for analysing binary counter? Explain.

### ✅ Answer
Strictly speaking, this statement is **not correct**.
The potential method **can** be used for binary counter analysis.

### Correct Explanation
Choose the potential function:

`Φ = number of 1s in the counter`

When the counter is incremented:
- one 0 becomes 1,
- some trailing 1s become 0.

If `t` bits flip, then:
- actual cost = `t`
- potential decreases by `t-2` overall
- amortized cost becomes at most **2**

### Conclusion
So binary counter **can be analyzed** by potential method, and its amortized increment cost is **O(1)**.

### 🎯 Marking Tip
Write politely: **“The statement is not correct; potential method can be applied by choosing Φ as number of 1-bits.”**

---

## ✏️ Question 103 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 9

### ❓ Full Question
Comment on the statements:
1. The knapsack problem is NP-hard
2. SAT is NP-complete
3. Minimum spanning tree is tractable

### ✅ Answer
1. **Knapsack is NP-hard** – True for the optimization version. No polynomial-time exact algorithm is known.
2. **SAT is NP-complete** – True. SAT was the first NP-complete problem.
3. **Minimum spanning tree is tractable** – True. Algorithms like Prim’s and Kruskal’s solve it in polynomial time.

### 🎯 Marking Tip
For each statement, write **True/False + one line justification**.

---

## ✏️ Question 104 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 10

### ❓ Full Question
Write a Rabin-Karp string matching algorithm. Let input be text `t` length `n` and pattern `p` length `m`. What is expected runtime and worst-case runtime?

### ✅ Answer
### Rabin-Karp Steps
1. Compute hash of pattern.
2. Compute hash of first text window of length `m`.
3. Slide the window through the text.
4. If hash matches, compare characters.
5. Update hash by rolling hash formula.

### Complexity
- Expected runtime = **O(n + m)**
- Worst-case runtime = **O(nm)**

### Reason for Worst Case
Many spurious hits may force repeated character comparisons.

### 🎯 Marking Tip
Use the keyword **rolling hash** and define **spurious hit**.

---

## ✏️ Question 105 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 8

### ❓ Full Question
Briefly explain performance measures — speedup, efficiency, throughput, contention, and latency of multithreaded algorithms.

### ✅ Answer
### Speedup
`Speedup = T1 / Tp`
How much faster the program becomes using `p` processors.

### Efficiency
`Efficiency = Speedup / p`
Shows processor utilization.

### Throughput
Amount of work completed per unit time.

### Contention
Competition among threads for shared resources like memory or locks.

### Latency
Delay between request and response / time for one operation to complete.

### 🎯 Marking Tip
Writing formulas for **speedup** and **efficiency** gives easy marks.

---

## ✏️ Question 106 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 10

### ❓ Full Question
Show stepwise process how the distributed breadth first search algorithm works on the given graph.

### ✅ Answer
Assume source node = **A**.

### Level 0
`{A}`

### Level 1
Neighbors of A → `{B, C}`

### Level 2
Neighbors of B and C:
- B gives `{D, E}`
- C gives `{F, G}`

So level 2 = `{D, E, F, G}`

### BFS Tree
```text
        A
      /   \
     B     C
    / \   / \
   D   E F   G
```

### Distributed View
Different processors can expand B and C at the same time, which speeds up traversal.

### 🎯 Marking Tip
Always write the answer level-wise: `{A}`, `{B,C}`, `{D,E,F,G}`.

---

## ✏️ Question 107 of 126
**📄 Paper:** [6354]-485 (PC-2368)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 8

### ❓ Full Question
If matrices are of order m×n and n×p, what is the time complexity in conventional approach and in multithreaded approach? Discuss.

### ✅ Answer
### Conventional Matrix Multiplication
For each of `m×p` entries, we do `n` multiply-add operations.
So time complexity is:

**O(mnp)**

### Multithreaded Approach
If work is divided among `P` processors, then ideal running time becomes approximately:

**O(mnp / P)**

ignoring thread overhead and communication.

### Discussion
- Total work remains `O(mnp)`.
- Parallel execution reduces wall-clock time.
- Actual speedup depends on processor count, overhead, cache behavior, and load balancing.

### 🎯 Marking Tip
Write both: **work remains O(mnp)** and **parallel time ideally becomes O(mnp/P)**.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 6 Complete                          │
│ 📎 Answers appended to answer6.md            │
│ 📚 Includes Questions 89 to 107              │
└──────────────────────────────────────────────┘
