═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 4 Answers — [5927]-342 (PA-912)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 4**.
> New answers will always be added at the bottom.

---

## ✏️ Question 55 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 10

### ❓ Full Question
Solve the matrix chain multiplication for the following 6-matrix problem using dynamic programming.

A1=10×20, A2=20×5, A3=5×15, A4=15×50, A5=50×10, A6=10×15

### ✅ Answer
Dimension array:
`p = [10, 20, 5, 15, 50, 10, 15]`

Using dynamic programming, the minimum multiplication cost for the full chain is:

**m[1,6] = 8750**

### Important intermediate costs
- `m[1,2] = 1000`
- `m[1,3] = 1750`
- `m[1,4] = 7250`
- `m[1,5] = 7750`
- `m[1,6] = 8750`

### Optimal Parenthesization
From the split table, the optimal order is:

**((A1A2)(((A3A4)A5)A6))**

### Final Answer
Minimum scalar multiplications = **8750**

### 🎯 Marking Tip
For full marks, write both the **minimum cost** and the **optimal parenthesization**.

---

## ✏️ Question 56 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 8

### ❓ Full Question
Explain Greedy strategy: Principle, control abstraction, time analysis of control abstraction with suitable example.

### ✅ Answer
### Greedy Principle
Greedy strategy chooses the **best available option at the current step** and never changes it later.

### Control Abstraction
```text
GREEDY(A)
1. S = empty solution
2. while not complete do
3.      x = best available candidate
4.      if x is feasible then
5.            add x to S
6. return S
```

### Time Analysis
Time depends on:
- selecting best candidate,
- feasibility check,
- number of candidates.

Usually:
- with sorting → **O(n log n)**
- with nested slot checking → sometimes **O(n²)**

### Example
**Activity Selection**
- Choose activity with earliest finish time.
- Then choose next compatible activity.

### 🎯 Marking Tip
Write the key sentence: **“Greedy makes a locally optimal choice at each step.”**

---

## ✏️ Question 57 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 10

### ❓ Full Question
Explain dynamic programming for creating an optimal binary search tree for a set of n keys. Use it for keys A, B, C, D with probabilities 0.1, 0.2, 0.4, 0.3.

### ✅ Answer
### OBST Idea
An **Optimal Binary Search Tree** places more frequently searched keys closer to the root so that average search cost becomes minimum.

### Probabilities
| Key | A | B | C | D |
|-----|---|---|---|---|
| Probability | 0.1 | 0.2 | 0.4 | 0.3 |

### DP Result
Using OBST recurrence:
`cost[i,j] = min(cost[i,r-1] + cost[r+1,j] + sum(i,j))`

The minimum cost is:

**1.7**

### Optimal Tree
The best root is **C**.
Left subtree root is **B**, and A becomes left child of B.
Right subtree root is **D**.

### Final OBST
```text
        C
       / \
      B   D
     /
    A
```

### 🎯 Marking Tip
Write the recurrence formula and draw the final tree. Those are the main scoring parts.

---

## ✏️ Question 58 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 8

### ❓ Full Question
Explain Dynamic Programming: Principle, control abstraction, time analysis of control abstraction with suitable example.

### ✅ Answer
### Principle
Dynamic programming solves a problem by:
- dividing it into smaller overlapping subproblems,
- solving each only once,
- storing results for reuse.

### Control Abstraction
```text
DP()
1. identify subproblems
2. define recurrence
3. initialize base values
4. solve smaller problems first
5. store all answers in table
6. combine them to get final answer
```

### Time Analysis
General form:

**Time = number of states × time per state**

Examples:
- 0/1 Knapsack → `O(nW)`
- Matrix Chain Multiplication → `O(n³)`

### Example
In 0/1 knapsack, `V[i][w]` stores best profit using first `i` items and capacity `w`.

### 🎯 Marking Tip
Use the words **overlapping subproblems** and **optimal substructure**.

---

## ✏️ Question 59 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 9

### ❓ Full Question
Explain the branch and bound approach. Write branch and bound algorithm for 0/1 knapsack and solve capacity 15 kg for items:
A(9,18), B(4,10), C(6,12), D(2,10)

### ✅ Answer
### Branch and Bound Idea
- Build a state-space tree.
- Compute an upper bound for each node.
- Prune nodes that cannot beat the current best solution.

### Knapsack Result
Check feasible combinations:
- A+B+D = weight `9+4+2 = 15`, profit `18+10+10 = 38` ✅
- B+C+D = weight 12, profit 32
- A+C = weight 15, profit 30
- A+D = 11, profit 28
- others are smaller or overweight

### Optimal Solution
**Select A, B and D**

Total weight = **15 kg**
Total profit = **38**

### Generic B&B Knapsack Algorithm
```text
1. Start from root node
2. Compute bound of root
3. Expand promising node
4. Create include/exclude children
5. Update best profit
6. Prune nodes whose bound < current best profit
```

### 🎯 Marking Tip
In knapsack B&B answers, always mention **bound**, **pruning**, and the final best set.

---

## ✏️ Question 60 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 8

### ❓ Full Question
Explain with suitable example Backtracking: Principle, control abstraction, time analysis of control abstraction.

### ✅ Answer
Backtracking is a method where we build the solution step by step and reject a partial solution as soon as it becomes invalid.

### Principle
- Try a choice.
- If it is promising, continue.
- If it fails, go back and try another choice.

### Control Abstraction
```text
BACKTRACK(k)
1. if solution complete then output it
2. else
3.      generate candidates
4.      for each candidate do
5.           if promising then
6.                include it
7.                BACKTRACK(k+1)
8.                remove it
```

### Example
**N-Queens** or **Graph Coloring**

### Time Complexity
Worst-case is usually **exponential**, often written as `O(b^d)` or `O(2^n)` depending on the problem.

### 🎯 Marking Tip
Write one small example along with the generic control abstraction.

---

## ✏️ Question 61 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 9

### ❓ Full Question
What is Branch and Bound method? Write control abstraction for Least Cost search.

### ✅ Answer
Branch and Bound is an optimization technique that systematically explores a state space tree and prunes nodes that cannot lead to a better answer.

### Least Cost Search Control Abstraction
```text
LC-SEARCH(root)
1. put root in priority queue
2. while queue is not empty do
3.      E = remove least cost live node
4.      if E is solution then return E
5.      generate children of E
6.      compute cost/bound of children
7.      insert promising children into queue
```

### Key Terms
- **Live node** = generated but not expanded
- **E-node** = node chosen for expansion
- **Dead node** = expanded or pruned node

### 🎯 Marking Tip
Use the phrase **“priority queue of live nodes”** in the answer.

---

## ✏️ Question 62 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 8

### ❓ Full Question
Explain backtracking with graph coloring problem. Find solution for the given graph C1…C5.

### ✅ Answer
### Backtracking in Graph Coloring
Assign colors to vertices one by one.
If a color clashes with an adjacent vertex, reject it and try the next color.

### One Valid Coloring
A valid 3-coloring is:
- `C1 = Red`
- `C2 = Green`
- `C4 = Green`
- `C3 = Red`
- `C5 = Blue`

### Why 3 colors are needed
Vertices `C3, C4, C5` form a triangle-like conflict set, so at least 3 colors are needed.

### 🎯 Marking Tip
Write the final coloring clearly as a list. That gets easy marks.

---

## ✏️ Question 63 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 10

### ❓ Full Question
Write short notes on Aggregate Analysis, Accounting Method, Potential Function Method, Tractable and Non-tractable Problems.

### ✅ Answer
### Aggregate Analysis
Total cost of a sequence is divided by number of operations.
Example: stack operations.

### Accounting Method
Charge extra to some operations and save credit for future costly operations.
Example: PUSH charged as 2.

### Potential Function Method
Use formula:
`Amortized Cost = Actual Cost + (Φ_after - Φ_before)`
Example: `Φ = stack size`.

### Tractable Problems
Problems solvable in polynomial time.
Examples: binary search, MST.

### Non-tractable Problems
No known polynomial-time algorithms.
Examples: TSP, SAT.

### 🎯 Marking Tip
For short notes, use small headings and one example for each point.

---

## ✏️ Question 64 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 8

### ❓ Full Question
Write short notes with suitable example of each:
1. Randomized algorithm
2. Approximation algorithm

### ✅ Answer
### Randomized Algorithm
Uses random choices while running.
Example: **Randomized Quick Sort**.
Benefit: avoids bad fixed input patterns.

### Approximation Algorithm
Gives a near-optimal solution in polynomial time for hard optimization problems.
Example: **2-approximation for Vertex Cover**.
Benefit: fast and practically useful.

### 🎯 Marking Tip
Definition + example + one advantage is enough for each short note.

---

## ✏️ Question 65 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 9

### ❓ Full Question
What is Potential function method of amortized analysis? Find amortized cost of PUSH, POP and MULTIPOP stack operations.

### ✅ Answer
Choose potential function:

`Φ = number of elements in the stack`

### Formula
`Amortized Cost = Actual Cost + (Φ_after - Φ_before)`

### PUSH
- Actual cost = 1
- ΔΦ = +1
- Amortized cost = `1 + 1 = 2`

### POP
- Actual cost = 1
- ΔΦ = -1
- Amortized cost = `1 - 1 = 0`

### MULTIPOP(k)
If `t` elements are removed:
- Actual cost = `t`
- ΔΦ = `-t`
- Amortized cost = `t - t = 0`

### Final Result
- PUSH = **2**
- POP = **0**
- MULTIPOP = **0**

All have **O(1)** amortized cost.

### 🎯 Marking Tip
Do not forget to write the chosen potential function before calculations.

---

## ✏️ Question 66 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 9

### ❓ Full Question
What is embedded algorithm? Explain embedded system scheduling using power optimized scheduling algorithm.

### ✅ Answer
### Embedded Algorithm
An **embedded algorithm** is an algorithm designed for embedded systems such as sensors, medical devices, washing machines, automotive controllers, etc.

### Features of Embedded Algorithms
- low memory usage
- low power consumption
- predictable timing
- simple implementation

### Power Optimized Scheduling
The goal is to complete tasks while using minimum energy.

### Basic Idea
- schedule tasks according to deadlines
- when system load is low, reduce processor speed/voltage
- execute tasks more efficiently and save battery power

### Example Approach
Dynamic Voltage and Frequency Scaling (DVFS):
- high speed when urgent work exists
- lower speed when tasks have slack time

### Benefit
- less power consumption
- longer battery life
- still meets deadlines

### 🎯 Marking Tip
Mention **deadline**, **power saving**, and **DVFS** if possible.

---

## ✏️ Question 67 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 10

### ❓ Full Question
Write short notes on:
1. Multithreaded matrix multiplication
2. Multithreaded merge sort
3. Distributed breadth first search
4. The Rabin-Karp algorithm

### ✅ Answer
### 1) Multithreaded Matrix Multiplication
Different threads compute different rows/blocks/cells of the result matrix in parallel.

### 2) Multithreaded Merge Sort
Left and right halves are sorted in parallel using `spawn`, then joined using `sync`.

### 3) Distributed BFS
Graph traversal is done level by level across multiple machines/processors.

### 4) Rabin-Karp Algorithm
Pattern matching is done using rolling hash values; character comparison is done only after hash match.

### 🎯 Marking Tip
Write 3 short lines under each heading instead of one long paragraph.

---

## ✏️ Question 68 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 7

### ❓ Full Question
With respect to multithreaded algorithms explain analyzing multithreaded algorithms, parallel loops, race conditions.

### ✅ Answer
### Analyzing Multithreaded Algorithms
Use:
- **Work (T1)** = time on one processor
- **Span (T∞)** = longest dependency chain
- **Parallelism = T1/T∞**

### Parallel Loops
A loop is parallel if different iterations are independent.
Example:
```text
parallel for i = 1 to n
    C[i] = A[i] + B[i]
```

### Race Conditions
A race condition occurs when multiple threads access shared data unsafely and results depend on timing.

### 🎯 Marking Tip
Include one simple loop example and one race-condition example.

---

## ✏️ Question 69 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 9

### ❓ Full Question
Write and explain pseudo code for multithreaded merge sort. How parallel merging gives a significant parallelism advantage over merge sort?

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

### Why Parallel Merging Helps
If merge step is sequential, it becomes the main bottleneck.
If merging is also parallel:
- span decreases,
- more processors work together,
- overall parallelism improves.

### Key Advantage
Parallel merge removes the large sequential part of ordinary merge sort.

### 🎯 Marking Tip
The line **“parallel merge removes the sequential bottleneck”** is very important.

---

## ✏️ Question 70 of 126
**📄 Paper:** [5927]-342 (PA-912)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 8

### ❓ Full Question
Write pseudo code for naïve string matching and Rabin-Karp string matching and analyze them.

### ✅ Answer
### Naïve String Matching
```text
NAIVE(T, P)
1. for s = 0 to n-m do
2.      compare P with T[s+1 ... s+m]
3.      if all characters match, report occurrence
```

### Complexity
- Best case: good if mismatch occurs early
- Worst case: **O(nm)**

### Rabin-Karp
```text
RABIN-KARP(T, P)
1. compute pattern hash
2. compute first text window hash
3. compare hashes window by window
4. on hash match, verify characters
5. update rolling hash
```

### Complexity
- Expected time: **O(n + m)**
- Worst case: **O(nm)**

### Comparison
- Naïve directly compares characters every time.
- Rabin-Karp first compares hash values, so it is faster on average.

### 🎯 Marking Tip
Write both complexities clearly for both algorithms.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 4 Complete                          │
│ 📎 Answers appended to answer4.md            │
│ 📚 Includes Questions 55 to 70               │
└──────────────────────────────────────────────┘
