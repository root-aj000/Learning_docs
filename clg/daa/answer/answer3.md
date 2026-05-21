═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 3 Answers — [6404]-81 (PD4576)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 3**.
> New answers will always be added at the bottom.

---

## ✏️ Question 35 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 8

### ❓ Full Question
Give a mathematical formulation for:
1. Fractional Knapsack problem
2. 0/1 Knapsack problem

### ✅ Answer
## 1) Fractional Knapsack
Let:
- `n` = number of items
- `p_i` = profit of item `i`
- `w_i` = weight of item `i`
- `x_i` = fraction of item `i` taken
- `m` = knapsack capacity

### Formulation
Maximize:
`Z = Σ (p_i x_i)` for `i = 1 to n`

Subject to:
`Σ (w_i x_i) ≤ m`

and
`0 ≤ x_i ≤ 1`

Here an item may be taken partially.

## 2) 0/1 Knapsack
Same notation, but now each item is either fully taken or not taken.

### Formulation
Maximize:
`Z = Σ (p_i x_i)`

Subject to:
`Σ (w_i x_i) ≤ m`

and
`x_i ∈ {0,1}`

### Difference
- Fractional knapsack: `x_i` can be any value between 0 and 1
- 0/1 knapsack: `x_i` can only be 0 or 1

### 🎯 Marking Tip
Write the constraint on `x_i` carefully. That single line is the main difference between the two formulations.

---

## ✏️ Question 36 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 9

### ❓ Full Question
Use greedy algorithmic strategy to compute an execution schedule having maximum number of non-conflicting activities.

| Activity | Start | Finish |
|----------|-------|--------|
| A | 1 | 3 |
| B | 3 | 4 |
| C | 2 | 5 |
| D | 0 | 7 |
| E | 5 | 9 |
| F | 8 | 10 |
| G | 11 | 12 |

### ✅ Answer
### Greedy Rule
Select activities in increasing order of **finish time**.

The activities are already almost in finish-time order:
A(1,3), B(3,4), C(2,5), D(0,7), E(5,9), F(8,10), G(11,12)

### Selection Steps
- Select **A** (finishes at 3)
- Next activity starting at or after 3 is **B** → select
- Next activity starting at or after 4 is **E** → select
- Next activity starting at or after 9 is **G** → select

### Final Schedule
**A, B, E, G**

### Why not others?
- C conflicts with A and B
- D conflicts with many
- F starts at 8, but E finishes at 9, so F conflicts with E

### Maximum set of non-conflicting activities
**{A, B, E, G}**

### 🎯 Marking Tip
Write the greedy rule clearly: **“Choose the next activity with the earliest finish time.”**

---

## ✏️ Question 37 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 7

### ❓ Full Question
With respect to dynamic programming, what is the principle of optimality? Give a mathematical representation.

### ✅ Answer
### Principle of Optimality
The **principle of optimality** says:

> An optimal solution to a problem contains optimal solutions to its subproblems.

That means if the whole answer is best, then the smaller parts inside it must also be best.

### Mathematical Representation
If `OPT(i, j)` denotes the optimal solution for subproblem `(i, j)`, then:

`OPT(i, j) = best combination of OPT(smaller subproblems)`

For example in matrix chain multiplication:

`m[i,j] = min { m[i,k] + m[k+1,j] + p(i-1)p(k)p(j) }`

This equation shows that the optimal value for `(i,j)` is built from optimal values of smaller parts.

### 🎯 Marking Tip
Write both: **definition in words** and **one recurrence relation**. That combination gets full marks.

---

## ✏️ Question 38 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 10

### ❓ Full Question
Solve the 0/1 knapsack problem with capacity 7 using dynamic programming.

| Item | Weight | Profit |
|------|--------|--------|
| A | 1 | 1 |
| B | 3 | 4 |
| C | 4 | 5 |
| D | 5 | 7 |

### ✅ Answer
### DP Table
Let `V[i][w]` = best profit using first `i` items and capacity `w`.

Final table:

| Item\Cap | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|----------|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| B | 0 | 1 | 1 | 4 | 5 | 5 | 5 | 5 |
| C | 0 | 1 | 1 | 4 | 5 | 6 | 6 | 9 |
| D | 0 | 1 | 1 | 4 | 5 | 7 | 8 | 9 |

### Maximum Profit
`V[4][7] = 9`

### Backtracking
At capacity 7:
- D not selected because value remains 9 from above row
- C selected
- remaining capacity = 7 - 4 = 3
- B selected
- remaining capacity = 0

### Optimal Set
**B and C**

Total weight = `3 + 4 = 7`
Total profit = `4 + 5 = 9`

### 🎯 Marking Tip
In DP knapsack, always show the **final table** and the **selected items** separately.

---

## ✏️ Question 39 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 10

### ❓ Full Question
Colour the given graph using Red and Black by backtracking and show the process.

### ✅ Answer
From the matrix, edges are:
- A–B, A–C, B–D, C–D

This is a bipartite graph, so 2-coloring is possible.

### Stepwise Coloring
- Color **A = Red**
- B is adjacent to A, so **B = Black**
- C is adjacent to A, so **C = Black**
- D is adjacent to B and C, so **D = Red**

### Final Coloring
- **A = Red**
- **B = Black**
- **C = Black**
- **D = Red**

### State-Space Idea
```text
A=R
├── B=B
│   └── C=B
│       └── D=R  ✓
```

### 🎯 Marking Tip
Mention why each color is chosen: **“adjacent vertices cannot have the same color.”**

---

## ✏️ Question 40 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 4

### ❓ Full Question
Prove that the full state space tree of sum of subsets of n elements using backtracking has `2^n - 1` nodes excluding leaf nodes.

### ✅ Answer
For each element, there are two choices:
1. include it
2. exclude it

So the state space tree is a **binary tree** of depth `n`.

Number of internal nodes in a full binary decision tree of depth `n` is:

`1 + 2 + 4 + ... + 2^(n-1)`

This is a geometric series:

`= 2^n - 1`

Hence, excluding the leaf nodes, the total number of nodes is:

**`2^n - 1`**

### 🎯 Marking Tip
Write the geometric series expansion. That is the proof.

---

## ✏️ Question 41 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q3(c)
**⭐ Marks:** 4

### ❓ Full Question
What are the major drawbacks of branch and bound method?

### ✅ Answer
Major drawbacks of Branch and Bound:
1. **High memory usage** – many live nodes may need to be stored.
2. **Worst-case exponential time** – may still explore a very large tree.
3. **Bound calculation can be costly** – computing good bounds takes extra time.
4. **Performance depends on bound quality** – weak bounds cause poor pruning.
5. **Implementation is more complex** than simple recursion.

### 🎯 Marking Tip
Write at least 4 points. “High memory” and “worst-case exponential time” are must-have points.

---

## ✏️ Question 42 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 8

### ❓ Full Question
State the sum of subsets problem. Write an algorithm using backtracking and comment on time complexity.

### ✅ Answer
### Statement
Given a set of positive integers and a target sum `M`, find all subsets whose sum is exactly `M`.

### Backtracking Algorithm
```text
SUMSUB(k, s)
1. if s = M then
2.      print current subset
3.      return
4. if k > n or s > M then return
5. include x[k]
6. SUMSUB(k+1, s + x[k])
7. exclude x[k]
8. SUMSUB(k+1, s)
```

### Idea
At each element, we make two choices:
- take it
- leave it

If current sum becomes greater than target, that branch is stopped.

### Time Complexity
Worst case explores all subsets:

**O(2^n)**

### 🎯 Marking Tip
Write “two choices for each element” and then conclude worst case `O(2^n)`.

---

## ✏️ Question 43 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 10

### ❓ Full Question
Solve the 0/1 knapsack problem using LC branch and bound for:
O1(5,6), O2(4,5), O3(3,4), capacity=7.

### ✅ Answer
Feasible combinations are:
- O1 → profit 6
- O2 → profit 5
- O3 → profit 4
- O1+O2 → overweight
- O1+O3 → overweight
- O2+O3 → weight 7, profit 9 ✅
- O1+O2+O3 → overweight

### Optimal Solution
**O2 + O3**

Weight = `4 + 3 = 7`
Profit = `5 + 4 = 9`

### LC Branch and Bound Note
The algorithm expands the live node with the best bound first and prunes branches whose bound is less than the current best profit.

### 🎯 Marking Tip
Show that the two combinations with O1 become overweight. That is why `{O2,O3}` is the best feasible set.

---

## ✏️ Question 44 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 6

### ❓ Full Question
Give an amortized analysis of a k-bit binary counter using aggregate method.

### ✅ Answer
A binary counter increments by flipping bits.

### Observation
- Bit 0 flips every increment.
- Bit 1 flips every 2 increments.
- Bit 2 flips every 4 increments.
- ...
- Bit `i` flips every `2^i` increments.

For `n` increments, total number of bit flips is:

`n + n/2 + n/4 + n/8 + ... < 2n`

So total cost of `n` increments is less than `2n`.

### Amortized Cost per Increment
`< 2n / n = 2 = O(1)`

Hence amortized cost per increment is:

**O(1)**

### 🎯 Marking Tip
Write the series `n + n/2 + n/4 + ... < 2n`. That is the core of the aggregate proof.

---

## ✏️ Question 45 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 7

### ❓ Full Question
What are tractable and non-tractable problems? Give examples.

### ✅ Answer
### Tractable Problems
Problems that can be solved in **polynomial time** are called tractable.
Examples of polynomial time:
- `O(n)`
- `O(n log n)`
- `O(n²)`

**Examples:**
- Binary Search
- Minimum Spanning Tree
- Shortest Path

### Non-tractable Problems
Problems for which no polynomial-time algorithm is known are called non-tractable or intractable.
They often need exponential or factorial time.

**Examples:**
- Traveling Salesman Problem
- SAT
- 0/1 Knapsack (general form)

### 🎯 Marking Tip
Write one line: **“Tractable = polynomial-time solvable; Non-tractable = no known polynomial-time solution.”**

---

## ✏️ Question 46 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q5(c)
**⭐ Marks:** 4

### ❓ Full Question
Does randomized algorithm for quick sort improve the average case time complexity?

### ✅ Answer
Randomized quick sort does **not change the asymptotic average-case complexity**.

- Ordinary quick sort average case = **O(n log n)**
- Randomized quick sort expected time = **O(n log n)**

### What it improves
It improves the **chance of avoiding bad pivot choices**, so it reduces the chance of worst-case behavior on special inputs.

So the correct comment is:
- **Average-case asymptotic order remains O(n log n)**
- **Practical robustness improves**

### 🎯 Marking Tip
Use the phrase: **“It improves expected behavior, not the asymptotic average-case order.”**

---

## ✏️ Question 47 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 6

### ❓ Full Question
Explain with example the methods of amortized analysis.

### ✅ Answer
There are three methods:

1. **Aggregate Method**
   - Total cost of sequence / number of operations
   - Example: stack operations

2. **Accounting Method**
   - Charge extra to some operations and save credit
   - Example: charge PUSH = 2 to pay for future POP

3. **Potential Method**
   - Use `Amortized Cost = Actual Cost + ΔΦ`
   - Example: choose `Φ = stack size`

### Example: Stack
Even though MULTIPOP may remove many elements, each element is removed only once. Hence amortized cost per operation is O(1).

### 🎯 Marking Tip
Name all three methods first, then explain them briefly with the same example.

---

## ✏️ Question 48 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 7

### ❓ Full Question
What is an approximation algorithm? How are performance ratios useful?

### ✅ Answer
An **approximation algorithm** is a polynomial-time algorithm that gives a solution close to the optimal solution for hard optimization problems.

### Performance Ratio
The performance ratio tells us how close the answer is to the optimal answer.

If the ratio is small and close to 1, the approximation is good.

### Why useful?
1. Measures solution quality.
2. Helps compare two approximation algorithms.
3. Gives a guaranteed bound on how bad the answer can be.
4. Makes near-optimal algorithms trustworthy.

### Example
A 2-approximation algorithm for vertex cover gives a solution at most twice the optimal size.

### 🎯 Marking Tip
Write: **“Performance ratio gives a guaranteed bound on solution quality.”**

---

## ✏️ Question 49 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q6(c)
**⭐ Marks:** 4

### ❓ Full Question
What are randomized algorithms? Enlist few reasons to use randomized algorithms.

### ✅ Answer
A **randomized algorithm** uses random choices while running.

### Reasons to use them
- simple to design
- good expected performance
- avoids bad fixed input cases
- useful in hashing, quick sort, load balancing

### Example
Randomized quick sort selects pivot randomly.

### 🎯 Marking Tip
Definition + 3 reasons + 1 example is enough.

---

## ✏️ Question 50 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 8

### ❓ Full Question
Write a simple multithreaded matrix multiplication algorithm based on parallelizing relevant loops.

### ✅ Answer
For matrices `A(m×n)` and `B(n×p)`, result `C(m×p)` is:

`C[i][j] = Σ A[i][k] × B[k][j]`

### Multithreaded Algorithm
```text
P-MATRIX-MULTIPLY(A, B, C)
parallel for i = 1 to m
    parallel for j = 1 to p
        C[i][j] = 0
        for k = 1 to n
            C[i][j] = C[i][j] + A[i][k] * B[k][j]
```

### Idea
Each element `C[i][j]` can be computed independently, so different threads can compute different cells in parallel.

### 🎯 Marking Tip
Mention that **each output cell is independent**, so parallelization is safe.

---

## ✏️ Question 51 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 4

### ❓ Full Question
Explain in brief race condition in multithreaded algorithms.

### ✅ Answer
A **race condition** occurs when:
- two or more threads access the same shared data at the same time,
- and at least one thread modifies it,
- leading to unpredictable results.

### Example
```text
x = 0
Thread 1: x = x + 1
Thread 2: x = x + 1
```
Final result may incorrectly become 1 instead of 2.

### Prevention
- locks
- mutexes
- semaphores
- atomic operations

### 🎯 Marking Tip
Always include a small shared-variable example.

---

## ✏️ Question 52 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q7(c)
**⭐ Marks:** 6

### ❓ Full Question
What do you understand by spawn and sync keywords used in multithreaded programming?

### ✅ Answer
### Spawn
`spawn` starts a child task that can execute in parallel with the parent.

### Sync
`sync` makes the parent wait until all spawned child tasks are completed.

### Example
```text
x = spawn computeLeft()
y = computeRight()
sync
answer = x + y
```

### Meaning
- `computeLeft()` and `computeRight()` run in parallel.
- `sync` ensures both results are ready before final addition.

### 🎯 Marking Tip
The simplest correct sentence is: **“spawn creates parallelism; sync joins it.”**

---

## ✏️ Question 53 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 8

### ❓ Full Question
Write distributed breadth search algorithm. What is its advantage over the conventional approach?

### ✅ Answer
### Distributed BFS Algorithm
```text
DBFS(source)
1. mark source as visited
2. frontier = {source}
3. while frontier is not empty do
4.      each processor expands its local frontier nodes
5.      collect all unvisited neighbors
6.      mark them visited
7.      form next frontier
8. repeat
```

### Advantage over Conventional BFS
1. Works on very large graphs distributed across machines.
2. Different processors expand different nodes simultaneously.
3. Faster traversal for huge graphs.
4. Better memory distribution.

### 🎯 Marking Tip
Mention **level-by-level traversal in parallel**. That is the key idea of distributed BFS.

---

## ✏️ Question 54 of 126
**📄 Paper:** [6404]-81 (PD4576)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 10

### ❓ Full Question
Write a Rabin-Karp string matching algorithm. Input: text `t` of length `n`, pattern `p` of length `m`. What is expected runtime and worst-case runtime?

### ✅ Answer
### Algorithm
```text
RABIN-KARP(T, P)
1. Compute hash of P
2. Compute hash of first window of T of size m
3. for each shift s = 0 to n-m do
4.      if hashes match then
5.           compare characters of P with current window
6.           if equal, report match
7.      compute next window hash
```

### Working
- Hash matching is fast.
- Character comparison is done only when hashes match.
- If a hash match is false, it is called a **spurious hit**.

### Time Complexity
- Expected runtime = **O(n + m)**
- Worst-case runtime = **O(nm)**

### 🎯 Marking Tip
Do not forget to define **spurious hit**. Many papers ask this idea indirectly.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 3 Complete                          │
│ 📎 Answers appended to answer3.md            │
│ 📚 Includes Questions 35 to 54               │
└──────────────────────────────────────────────┘
