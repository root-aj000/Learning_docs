═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 2 Answers — [6263]-81 (PB2243)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 2**.
> New answers will always be added at the bottom.

---

## ✏️ Question 18 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 6

### ❓ Full Question
Write a control abstraction for greedy method. Comment on the time complexity of this abstraction.

### 📌 Concept
A **greedy method** builds the answer step by step. At each step it picks the **best available local choice** and never changes that choice later.

### ✅ Answer
### Control Abstraction of Greedy Method
```text
GREEDY(A)
1. S = empty solution
2. while solution not complete do
3.      x = select best available candidate
4.      if x is feasible with S then
5.            S = S ∪ {x}
6. return S
```

### Explanation
- `select best available candidate` means choose the item that looks best now.
- `feasible` means adding it should not break the problem constraints.
- The algorithm continues until the solution is complete.

### Time Complexity
The time depends on:
1. **Selecting the best candidate**
2. **Checking feasibility**
3. Number of candidates

If sorting is needed first, complexity is often **O(n log n)**.
If feasibility is checked inside nested loops, it may become **O(n²)**.
So the control abstraction itself is generic; its exact time complexity depends on the problem.

### 🎯 Marking Tip
Write the words **selection function**, **feasibility function**, and say that greedy makes a **locally optimal choice** at each step.

---

## ✏️ Question 19 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 8

### ❓ Full Question
Find an optimal solution for the following knapsack instance using greedy method. Number of objects n = 5. Capacity of knapsack m = 100.

| Object | Weight | Profit |
|--------|--------|--------|
| O1 | 20 | 10 |
| O2 | 30 | 20 |
| O3 | 66 | 30 |
| O4 | 40 | 40 |
| O5 | 60 | 50 |

### 📌 Concept
This is solved using the **fractional knapsack greedy rule**: choose items in decreasing order of **profit/weight ratio**.

### ✅ Answer
### Step 1: Compute profit/weight ratio

| Object | Weight | Profit | Profit/Weight |
|--------|--------|--------|---------------|
| O1 | 20 | 10 | 10/20 = 0.50 |
| O2 | 30 | 20 | 20/30 = 0.67 |
| O3 | 66 | 30 | 30/66 ≈ 0.45 |
| O4 | 40 | 40 | 40/40 = 1.00 |
| O5 | 60 | 50 | 50/60 ≈ 0.83 |

### Step 2: Sort by ratio
Order: **O4, O5, O2, O1, O3**

### Step 3: Fill knapsack
- Take **O4** fully: weight = 40, profit = 40, remaining capacity = 60
- Take **O5** fully: weight = 60, profit = 50, remaining capacity = 0

Total profit = **40 + 50 = 90**

### Final Selection
- Selected objects: **O4 and O5**
- Total weight = **100**
- Total profit = **90**

### 🎯 Marking Tip
Always show the **ratio table**. In greedy knapsack, most marks are given for correct sorting by **profit/weight**.

---

## ✏️ Question 20 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q1(c)
**⭐ Marks:** 4

### ❓ Full Question
Comment on the statement: “Problem which does not satisfy the principle of optimality cannot be solved by dynamic programming”.

### ✅ Answer
The statement is **true**.

Dynamic programming works only when the problem has **optimal substructure**, which means:
- the optimal solution of the whole problem
- can be built from optimal solutions of its smaller subproblems.

This is called the **principle of optimality**.

If a problem does **not** satisfy this property, then solving subproblems optimally does not guarantee the whole solution will be optimal. In that case, dynamic programming is not suitable.

### Example
- **Matrix Chain Multiplication** satisfies the principle of optimality, so DP works.
- Problems without optimal substructure cannot be solved correctly by DP.

### 🎯 Marking Tip
Write one line clearly: **“Dynamic programming requires optimal substructure; without it, DP cannot guarantee an optimal answer.”**

---

## ✏️ Question 21 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 8

### ❓ Full Question
Write a control abstraction for dynamic programming strategy. Comment on the time complexity of this abstraction.

### ✅ Answer
### Control Abstraction of Dynamic Programming
```text
DYNAMIC-PROGRAMMING()
1. Identify subproblems
2. Define recurrence relation
3. Initialize base cases
4. Solve subproblems in proper order
5. Store each answer in a table
6. Use stored values to build final answer
7. Return final optimal solution
```

### Explanation
Dynamic programming solves a problem by:
- breaking it into smaller overlapping subproblems,
- storing results in a table,
- reusing stored results instead of solving the same thing again.

### Time Complexity
The time complexity depends on:
- number of states/subproblems,
- time taken to compute each state.

So in general:

**Time Complexity = Number of subproblems × Time per subproblem**

Examples:
- 0/1 Knapsack: **O(nW)**
- Matrix Chain Multiplication: **O(n³)**
- Floyd Warshall: **O(n³)**

### 🎯 Marking Tip
In DP theory answers, mention these two words: **optimal substructure** and **overlapping subproblems**.

---

## ✏️ Question 22 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 10

### ❓ Full Question
Consider 4 matrices A1, A2, A3 and A4. The orders are:
A1 = 3×5, A2 = 5×4, A3 = 4×2, A4 = 2×4.
Find the optimal sequence of chain matrix multiplication using dynamic programming.

### ✅ Answer
### Step 1: Dimension array
`p = [3, 5, 4, 2, 4]`

### Step 2: Cost of length 2 chains
- `m[1,2] = 3×5×4 = 60`
- `m[2,3] = 5×4×2 = 40`
- `m[3,4] = 4×2×4 = 32`

### Step 3: Cost of length 3 chains
For `m[1,3]`:
- k=1: `0 + 40 + 3×5×2 = 70`
- k=2: `60 + 0 + 3×4×2 = 84`
- So `m[1,3] = 70`

For `m[2,4]`:
- k=2: `0 + 32 + 5×4×4 = 112`
- k=3: `40 + 0 + 5×2×4 = 80`
- So `m[2,4] = 80`

### Step 4: Cost of full chain `m[1,4]`
- k=1: `0 + 80 + 3×5×4 = 140`
- k=2: `60 + 32 + 3×4×4 = 140`
- k=3: `70 + 0 + 3×2×4 = 94`

So minimum cost is:
**m[1,4] = 94**

### Optimal Parenthesization
- First compute `(A2A3)`
- Then compute `A1(A2A3)`
- Then multiply with `A4`

So optimal order is:

**((A1(A2A3))A4)**

### Cost Table
| i\j | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| 1 | 0 | 60 | 70 | 94 |
| 2 | - | 0 | 40 | 80 |
| 3 | - | - | 0 | 32 |
| 4 | - | - | - | 0 |

### 🎯 Marking Tip
Do not write only the final order. Show all `k` values tried for each main cell like `m[1,3]`, `m[2,4]`, and `m[1,4]`.

---

## ✏️ Question 23 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 8

### ❓ Full Question
Assume a graph with n vertices is represented by an adjacency matrix G. Let there be m colours available. Write a recursive backtracking algorithm to colour all vertices. What is the time complexity?

### ✅ Answer
### Recursive Backtracking Algorithm
```text
mColoring(k)
1. if k > n then
2.      print color[1..n]
3.      return
4. for c = 1 to m do
5.      if isSafe(k, c) then
6.           color[k] = c
7.           mColoring(k + 1)
8.           color[k] = 0
```

### Safety Test
```text
isSafe(k, c)
1. for i = 1 to n do
2.      if G[k][i] = 1 and color[i] = c then
3.            return false
4. return true
```

### Idea
- Color vertices one by one.
- For each vertex, try all `m` colors.
- If a chosen color conflicts with an adjacent vertex, reject it.
- If all vertices get valid colors, print the solution.

### Time Complexity
For each of the `n` vertices, we may try up to `m` colors.
So worst-case complexity is:

**O(m^n)**

If adjacency checking is included, it may be written as **O(n·m^n)**, but the standard answer is **O(m^n)**.

### 🎯 Marking Tip
Write both parts separately: **main recursive function** and **promising/safety function**.

---

## ✏️ Question 24 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 9

### ❓ Full Question
Three items are given:
O1(Weight=5, Value=6), O2(Weight=4, Value=5), O3(Weight=3, Value=4), capacity = 7.
Solve 0/1 knapsack using LC branch and bound method.

### ✅ Answer
### Step 1: Profit/Weight ratios
| Object | Weight | Value | Ratio |
|--------|--------|-------|-------|
| O1 | 5 | 6 | 1.20 |
| O2 | 4 | 5 | 1.25 |
| O3 | 3 | 4 | 1.33 |

Sort by ratio: **O3, O2, O1**

### Step 2: Check feasible combinations
- O1 alone → profit 6, weight 5
- O2 alone → profit 5, weight 4
- O3 alone → profit 4, weight 3
- O1 + O2 → weight 9 ❌
- O1 + O3 → weight 8 ❌
- O2 + O3 → weight 7 ✅ profit 9
- O1 + O2 + O3 → weight 12 ❌

### Best Solution
The best feasible combination is:
- **O2 + O3**
- Total weight = `4 + 3 = 7`
- Total profit = `5 + 4 = 9`

### Branch and Bound Note
In LC branch and bound, nodes with low cost / high bound are expanded first. Branches whose upper bound is below the current best profit are pruned.

### 🎯 Marking Tip
Even if you show the state-space briefly, always highlight the final feasible best set: **{O2, O3} with profit 9**.

---

## ✏️ Question 25 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 8

### ❓ Full Question
Compare backtracking with branch and bound with respect to search technique, exploration of state space tree and kind of problems solved.

### ✅ Answer
| Basis | Backtracking | Branch and Bound |
|------|--------------|------------------|
| **Search technique** | Depth-first search is commonly used | Best-first / BFS / LC search often used |
| **Main goal** | Find feasible solution(s) | Find optimal solution |
| **Pruning rule** | Prune non-promising node if it cannot lead to a valid solution | Prune node if its bound shows it cannot beat current best answer |
| **State-space exploration** | Goes deep along one path, then backtracks | Keeps many live nodes and chooses next using bound/cost |
| **Problems solved** | N-Queens, graph coloring, sum of subsets | 0/1 knapsack, TSP, assignment problem |

### Simple Difference
- **Backtracking** = feasibility oriented
- **Branch and Bound** = optimization oriented

### 🎯 Marking Tip
Write one sentence exactly like this: **“Backtracking is mainly for constraint satisfaction, while branch and bound is mainly for optimization.”**

---

## ✏️ Question 26 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 9

### ❓ Full Question
For A = {5, 10, 15, 20, 25}, find the first subset whose sum is 30 using backtracking. Show state space tree.

### ✅ Answer
We try elements in the given order.

### Backtracking Steps
- Start with `{}` sum = 0
- Include 5 → sum = 5
- Include 10 → sum = 15
- Include 15 → sum = 30 ✅

So the **first solution** found is:

**{5, 10, 15}**

### State Space Tree
```text
                 {}
               /    \
             {5}    {}
            /   \
      {5,10}    {5}
        /   \
{5,10,15}   {5,10}
    ✓
```

### Final Answer
First subset with sum 30 = **{5, 10, 15}**

### 🎯 Marking Tip
Since the question asks for **first solution**, mention the search order clearly: “elements are tried in the order 5, 10, 15, 20, 25.”

---

## ✏️ Question 27 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 8

### ❓ Full Question
What are randomized algorithms? Enlist and explain in brief the primary reasons for using randomized algorithms.

### ✅ Answer
A **randomized algorithm** uses random numbers or random choices during execution.

### Reasons for Using Randomized Algorithms
1. **Simplicity** – sometimes easier to design than deterministic algorithms.
2. **Good expected performance** – often fast on average.
3. **Avoid worst-case patterns** – random choice prevents specially bad input arrangements.
4. **Useful for large problems** – gives practical solutions quickly.
5. **Helpful in distributed systems and hashing** – randomness reduces collisions and conflicts.

### Example
**Randomized Quick Sort** chooses pivot randomly and usually performs in **O(n log n)** expected time.

### 🎯 Marking Tip
Definition + 4 reasons + 1 example is enough for full marks in such theory questions.

---

## ✏️ Question 28 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 9

### ❓ Full Question
What are approximation algorithms? Based on the approximation ratio, classify the approximation algorithms.

### ✅ Answer
An **approximation algorithm** gives a solution close to the optimal one in polynomial time for hard optimization problems.

### Approximation Ratio
If `A(I)` is the algorithm answer and `OPT(I)` is the optimal answer:
- For minimization: `A(I) / OPT(I) ≤ ρ(n)`
- For maximization: `OPT(I) / A(I) ≤ ρ(n)`

Here `ρ(n)` is the approximation ratio.

### Classification
1. **Exact algorithm**: ratio = 1
2. **Constant-factor approximation**: ratio = constant like 2, 3, etc.
3. **Logarithmic approximation**: ratio = `O(log n)`
4. **PTAS** (Polynomial Time Approximation Scheme): for any `ε > 0`, gives `(1+ε)` close answer
5. **FPTAS**: PTAS with better dependence on `1/ε`

### Example
- Vertex Cover 2-approximation
- Knapsack has approximation schemes

### 🎯 Marking Tip
Do not skip the formula for ratio. It is the main marking point in approximation questions.

---

## ✏️ Question 29 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 8

### ❓ Full Question
Explain the methods of amortized analysis. Give suitable example.

### ✅ Answer
There are **three methods** of amortized analysis:

### 1) Aggregate Method
Find total cost of a sequence of operations and divide by number of operations.
- Example: in a stack, each element is pushed once and popped at most once.

### 2) Accounting Method
Charge extra cost to cheap operations and store the extra as credit.
- Example: charge PUSH as 2 instead of 1; saved credit pays for POP later.

### 3) Potential Method
Use a potential function `Φ`.
- Amortized cost = Actual cost + `(Φ_after - Φ_before)`
- Example: for stack, choose `Φ = number of elements`

### Suitable Example: Stack
- PUSH: O(1)
- POP: O(1)
- MULTIPOP: actual cost may be high, but amortized cost is O(1)

### 🎯 Marking Tip
In this question, write all **three method names as headings**. That makes the answer neat and complete.

---

## ✏️ Question 30 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 9

### ❓ Full Question
For an embedded medical device collecting time-stamped sensor data in real time, suggest a suitable sorting algorithm and justify.

### ✅ Answer
A suitable sorting algorithm is **Insertion Sort**.

### Why Insertion Sort is suitable here
1. **Small data batches** – embedded devices often process small chunks of incoming data.
2. **Nearly sorted input** – sensor data usually arrives almost in time order.
3. **Low memory use** – insertion sort is **in-place** and needs very little extra memory.
4. **Simple implementation** – important for embedded systems.
5. **Good for online updates** – new reading can be inserted quickly into an already sorted list.

### Why not heavy algorithms?
- Merge sort needs extra memory.
- Quick sort has bad worst case.
- Heap sort is more complex and less cache friendly for tiny batches.

### Conclusion
For real-time embedded monitoring with small, nearly ordered data, **Insertion Sort** is a practical and safe choice.

### 🎯 Marking Tip
Mention these exact phrases: **low memory**, **simple implementation**, and **nearly sorted real-time data**.

---

## ✏️ Question 31 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 10

### ❓ Full Question
Write a Rabin-Karp string matching algorithm. What is the expected runtime and worst-case runtime?

### ✅ Answer
### Rabin-Karp Algorithm
```text
RABIN-KARP(T, P, d, q)
1. n = length(T), m = length(P)
2. Compute hash of pattern P
3. Compute hash of first window of T of length m
4. for s = 0 to n-m do
5.      if pattern hash = current window hash then
6.           compare characters one by one
7.           if all match, report occurrence
8.      if s < n-m then
9.           update hash using rolling hash
```

### Working Idea
- Compare hash values first.
- If hashes match, then verify by actual character comparison.
- This avoids unnecessary full comparisons most of the time.

### Time Complexity
- **Expected runtime:** `O(n + m)`
- **Worst-case runtime:** `O(nm)`

Worst case happens when many windows have same hash as the pattern but actual strings differ.

### 🎯 Marking Tip
Always mention **spurious hit** while explaining worst case.

---

## ✏️ Question 32 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 8

### ❓ Full Question
Write multithreaded merge sort algorithm. Briefly discuss how it differs from conventional merge sort.

### ✅ Answer
### Pseudo Code
```text
P-MERGE-SORT(A, p, r)
1. if p >= r then return
2. q = floor((p + r)/2)
3. spawn P-MERGE-SORT(A, p, q)
4. spawn P-MERGE-SORT(A, q+1, r)
5. sync
6. MERGE(A, p, q, r)
```

### Difference from Conventional Merge Sort
| Conventional Merge Sort | Multithreaded Merge Sort |
|-------------------------|--------------------------|
| Runs on one processor | Uses multiple threads/processors |
| Left and right halves sorted one after another | Left and right halves sorted in parallel |
| Lower hardware use | Better use of multicore CPU |
| Simpler | Faster on large data with parallel hardware |

### Key Idea
Multithreaded merge sort reduces execution time by doing independent recursive calls simultaneously.

### 🎯 Marking Tip
Write the keywords **spawn** and **sync**. These are the most important terms in multithreaded answers.

---

## ✏️ Question 33 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 10

### ❓ Full Question
Show stepwise process of distributed breadth first search on the given graph.

### ✅ Answer
Assume source node = **A**.

### Graph Levels
From the adjacency matrix:
- A is connected to B, C
- B is connected to D, E
- C is connected to F, G

### DBFS Steps
**Level 0**
- Start at A
- Visited: {A}

**Level 1**
- Explore neighbors of A
- New nodes found: {B, C}
- Visited: {A, B, C}

**Level 2**
- Explore neighbors of B and C
- From B → D, E
- From C → F, G
- New nodes found: {D, E, F, G}
- Visited: {A, B, C, D, E, F, G}

No more unvisited nodes remain.

### BFS Tree
```text
        A
      /   \
     B     C
    / \   / \
   D   E F   G
```

### Distributed View
- Processor holding A sends frontier to processors holding B and C.
- Then processors for B and C expand in parallel.
- This is why DBFS is efficient for distributed systems.

### 🎯 Marking Tip
The examiner usually wants the **level-wise traversal**. Write clearly: `Level 0 = {A}, Level 1 = {B,C}, Level 2 = {D,E,F,G}`.

---

## ✏️ Question 34 of 126
**📄 Paper:** [6263]-81 (PB2243)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 8

### ❓ Full Question
What do you understand by spawn and sync keywords used in multithreaded programming? Explain with suitable example.

### ✅ Answer
### Spawn
`spawn` means create a new parallel task.
The parent task and spawned task can run at the same time.

### Sync
`sync` means wait until all previously spawned child tasks are completed.

### Example
```text
P-FIB(n)
1. if n <= 1 return n
2. x = spawn P-FIB(n-1)
3. y = P-FIB(n-2)
4. sync
5. return x + y
```

### Explanation
- `P-FIB(n-1)` starts in parallel.
- Meanwhile, the parent computes `P-FIB(n-2)`.
- `sync` ensures both answers are ready before addition.

### Simple Meaning
- **spawn = start parallel work**
- **sync = wait for parallel work to finish**

### 🎯 Marking Tip
Give one short example with both keywords in code. That makes the explanation complete.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 2 Complete                          │
│ 📎 Answers appended to answer2.md            │
│ 📚 Includes Questions 18 to 34               │
└──────────────────────────────────────────────┘
