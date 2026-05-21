═══════════════════════════════════════════════════════
# 📚 Design and Analysis of Algorithms
# 📝 Complete Answer Guide — All 7 Papers
# 📄 Paper 7 Answers — [6584]-91 (PE2192)
═══════════════════════════════════════════════════════

> This is an **append-only** study file for **Paper 7**.
> New answers will always be added at the bottom.

---

## ✏️ Question 108 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q1(a)
**⭐ Marks:** 6

### ❓ Full Question
Compare greedy algorithmic strategy and dynamic programming strategy.

### ✅ Answer
| Basis | Greedy Strategy | Dynamic Programming |
|------|-----------------|--------------------|
| Idea | Best local choice at each step | Solve and store smaller subproblems |
| Revision of choices | Does not revise earlier choices | Combines many subproblem answers |
| Requirement | Greedy-choice property | Optimal substructure + overlapping subproblems |
| Speed | Usually faster | Usually slower but more systematic |
| Optimality | Not always optimal | Gives optimal solution when DP properties hold |
| Example | Activity selection | 0/1 knapsack |

### 🎯 Marking Tip
A table format is best for comparison questions.

---

## ✏️ Question 109 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q1(b)
**⭐ Marks:** 8

### ❓ Full Question
Jobs: A(2,75), B(1,40), C(1,50), D(2,35). Every job takes one unit time. Maximize profit using greedy strategy.

### ✅ Answer
### Sort by Profit
| Job | Deadline | Profit |
|-----|----------|--------|
| A | 2 | 75 |
| C | 1 | 50 |
| B | 1 | 40 |
| D | 2 | 35 |

Maximum deadline = 2 → slots 1, 2

### Schedule
- A → slot 2
- C → slot 1
- B cannot fit
- D cannot fit

### Final Sequence
**C, A**

### Maximum Profit
`50 + 75 = 125`

### 🎯 Marking Tip
Show slot assignment clearly. That is where most marks come from.

---

## ✏️ Question 110 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q1(c)
**⭐ Marks:** 3

### ❓ Full Question
Comment on the statement “Greedy method always provides the optimal solution”.

### ✅ Answer
The statement is **false**.

Greedy method gives optimal solution **only for problems that satisfy the greedy-choice property**.
For some problems it works, but for others it may fail.

### Example
- Works for: activity selection, fractional knapsack
- Does not always work for: 0/1 knapsack

### 🎯 Marking Tip
Write the word **false** first, then justify with one example.

---

## ✏️ Question 111 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q2(a)
**⭐ Marks:** 7

### ❓ Full Question
Write a control abstraction for dynamic programming strategy. Comment on time complexity.

### ✅ Answer
### Control Abstraction
```text
DP()
1. identify subproblems
2. write recurrence relation
3. initialize base cases
4. solve subproblems in correct order
5. store answers in table
6. return final answer
```

### Time Complexity
General form:

**Time = number of subproblems × work per subproblem**

Examples:
- Knapsack: `O(nW)`
- Matrix chain multiplication: `O(n³)`

### 🎯 Marking Tip
Mention the table storage idea. That distinguishes DP from plain recursion.

---

## ✏️ Question 112 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q2(b)
**⭐ Marks:** 10

### ❓ Full Question
Find the binomial coefficients of `(x+y)^5` using dynamic programming.

### ✅ Answer
Using Pascal’s triangle / DP recurrence:

`C(n,r) = C(n-1,r-1) + C(n-1,r)`

with base cases:
- `C(n,0) = 1`
- `C(n,n) = 1`

### Pascal Triangle up to n = 5
- Row 0: `1`
- Row 1: `1 1`
- Row 2: `1 2 1`
- Row 3: `1 3 3 1`
- Row 4: `1 4 6 4 1`
- Row 5: `1 5 10 10 5 1`

### Therefore
`(x+y)^5 = 1x^5 + 5x^4y + 10x^3y^2 + 10x^2y^3 + 5xy^4 + 1y^5`

### Binomial Coefficients
**1, 5, 10, 10, 5, 1**

### 🎯 Marking Tip
Draw Pascal’s triangle row by row. That is the dynamic programming construction.

---

## ✏️ Question 113 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q3(a)
**⭐ Marks:** 10

### ❓ Full Question
For A = {5,10,15,20,25}, find the first subset whose sum is 30 using backtracking. Show space tree.

### ✅ Answer
### Backtracking Steps
- Start with `{}` sum = 0
- Include 5 → sum = 5
- Include 10 → sum = 15
- Include 15 → sum = 30 ✅

### First Solution
**{5, 10, 15}**

### State Space Tree
```text
                 {}
                /
              {5}
              /
          {5,10}
            /
      {5,10,15} ✓
```

### 🎯 Marking Tip
Since the question asks for the **first** solution, stop after reaching `{5,10,15}`.

---

## ✏️ Question 114 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q3(b)
**⭐ Marks:** 8

### ❓ Full Question
Write a control abstraction for branch and bound strategy. Enlist few applications of branch and bound.

### ✅ Answer
### Control Abstraction
```text
BAND-B(root)
1. put root in list of live nodes
2. while live nodes exist do
3.      choose best live node E
4.      if E is solution then update best answer
5.      else generate children of E
6.      compute bound for each child
7.      keep only promising children as live nodes
```

### Applications
- 0/1 knapsack
- Traveling Salesman Problem
- Assignment problem
- Job sequencing with constraints

### 🎯 Marking Tip
The words **live node**, **bound**, and **pruning** must appear.

---

## ✏️ Question 115 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q4(a)
**⭐ Marks:** 10

### ❓ Full Question
Traveling salesperson problem for cities P, Q, R. Cost matrix:
P→Q=4, P→R=2, Q→P=3, Q→R=4, R→P=1, R→Q=8. Tour starts from P.

### ✅ Answer
Possible tours from P:
1. `P → Q → R → P`
   - Cost = `4 + 4 + 1 = 9`
2. `P → R → Q → P`
   - Cost = `2 + 8 + 3 = 13`

### Minimum Cost Tour
**P → Q → R → P**

### Minimum Cost
**9**

### 🎯 Marking Tip
For 3 cities, evaluate both possible tours fully and choose the smaller one.

---

## ✏️ Question 116 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q4(b)
**⭐ Marks:** 8

### ❓ Full Question
Write an algorithm for graph colouring problem using backtracking. Let graph have n nodes and m colours. What is the time complexity?

### ✅ Answer
### Algorithm
```text
mColor(k)
1. if k > n then print coloring
2. else
3.      for c = 1 to m do
4.           if safe(k, c) then
5.                color[k] = c
6.                mColor(k+1)
7.                color[k] = 0
```

### Safety Function
A color is safe if no adjacent vertex already has the same color.

### Time Complexity
Worst case:

**O(m^n)**

### 🎯 Marking Tip
State clearly that the algorithm tries up to `m` colors for each of `n` vertices.

---

## ✏️ Question 117 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q5(a)
**⭐ Marks:** 5

### ❓ Full Question
Prove that amortized cost per operation in a k-bit binary counter is O(1).

### ✅ Answer
For `n` increments:
- bit 0 flips `n` times
- bit 1 flips `n/2` times
- bit 2 flips `n/4` times
- ...

Total flips:
`n + n/2 + n/4 + ... < 2n`

So total cost of `n` increment operations is less than `2n`.
Hence amortized cost per increment is:

`2n / n = 2 = O(1)`

### 🎯 Marking Tip
Write the geometric series and conclude `O(1)`.

---

## ✏️ Question 118 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q5(b)
**⭐ Marks:** 6

### ❓ Full Question
What are intractable problems? Does traveling salesperson problem belong to this class? Justify.

### ✅ Answer
### Intractable Problems
Problems for which no polynomial-time algorithm is known are called intractable or non-tractable.

### TSP
Yes, the **Traveling Salesperson Problem** belongs to this class.
- Decision version of TSP is **NP-complete**.
- Optimization version is **NP-hard**.
- Number of tours grows very fast with number of cities.

### 🎯 Marking Tip
Use both terms: **NP-complete** and **NP-hard**.

---

## ✏️ Question 119 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q5(c)
**⭐ Marks:** 6

### ❓ Full Question
Briefly explain any two methods of amortized analysis.

### ✅ Answer
### 1) Aggregate Method
Find total cost of all operations and divide by number of operations.
Example: stack operations.

### 2) Potential Method
Use a potential function `Φ`.
Amortized cost = actual cost + change in potential.
Example: stack with `Φ = number of elements`.

(You may also write Accounting Method instead of one of these.)

### 🎯 Marking Tip
Name the methods clearly and give one example with each.

---

## ✏️ Question 120 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q6(a)
**⭐ Marks:** 9

### ❓ Full Question
Explain amortized analysis using stack example and single push, single pop and multiple pop operations. Give amortized costs.

### ✅ Answer
Choose potential function:
`Φ = number of elements in stack`

### Costs
- **PUSH**: actual = 1, ΔΦ = +1 → amortized = 2
- **POP**: actual = 1, ΔΦ = -1 → amortized = 0
- **MULTIPOP(t)**: actual = t, ΔΦ = -t → amortized = 0

### Conclusion
All stack operations have **O(1)** amortized cost.

### 🎯 Marking Tip
Write the cost formula and the final three amortized values.

---

## ✏️ Question 121 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q6(b)
**⭐ Marks:** 4

### ❓ Full Question
Does randomized algorithm for quick sort improve the average case time complexity? Discuss.

### ✅ Answer
No, it does not improve the asymptotic average-case order.

- Average case of quick sort = `O(n log n)`
- Expected time of randomized quick sort = `O(n log n)`

What improves is the chance of avoiding bad pivot choices.
So practical behavior becomes more reliable.

### 🎯 Marking Tip
Write: **“expected O(n log n), but asymptotic average order remains same.”**

---

## ✏️ Question 122 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q6(c)
**⭐ Marks:** 4

### ❓ Full Question
Comment on the statement “Searching an ordered list or searching an unordered list belongs to the class of tractable problems”.

### ✅ Answer
The statement is **true**.

### Reason
- Searching an **unordered list** takes linear time `O(n)`.
- Searching an **ordered list** can be done by binary search in `O(log n)`.

Both are polynomial-time algorithms, so both problems are **tractable**.

### 🎯 Marking Tip
Mention both complexities: `O(n)` and `O(log n)`.

---

## ✏️ Question 123 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q7(a)
**⭐ Marks:** 10

### ❓ Full Question
Let text hash code be “4133124” and pattern be “124”. Use modulo arithmetic with prime 13 and apply Rabin-Karp algorithm. Show stepwise process. What is the time complexity?

### ✅ Answer
Pattern length = 3
Pattern = `124`

### Pattern Hash
`124 mod 13 = 7`

### Text Windows of Length 3
| Window | Value | mod 13 | Result |
|--------|-------|--------|--------|
| 413 | 413 | 10 | No match |
| 133 | 133 | 3 | No match |
| 331 | 331 | 6 | No match |
| 312 | 312 | 0 | No match |
| 124 | 124 | 7 | Hash match + exact match |

### Match Found
Pattern occurs at the last window `124`.

### Time Complexity
- Expected = **O(n + m)**
- Worst case = **O(nm)**

### 🎯 Marking Tip
Show the window table. That is the stepwise Rabin-Karp process.

---

## ✏️ Question 124 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q7(b)
**⭐ Marks:** 8

### ❓ Full Question
Compare row-wise and block-wise approaches in multithreaded matrix multiplication with respect to division of work, communication overhead and cache efficiency.

### ✅ Answer
| Basis | Row-wise Approach | Block-wise Approach |
|------|-------------------|--------------------|
| Division of work | Each thread gets one or more rows | Each thread gets one block/submatrix |
| Communication overhead | Usually lower | Sometimes more coordination needed |
| Cache efficiency | Moderate | Better cache locality |
| Load balancing | May become uneven | Usually better balanced |
| Best use | Simple parallelization | High-performance implementations |

### Conclusion
Row-wise is simpler, but block-wise is usually better for cache use and large matrix performance.

### 🎯 Marking Tip
Write the comparison as a table.

---

## ✏️ Question 125 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q8(a)
**⭐ Marks:** 10

### ❓ Full Question
Write a naïve string matching algorithm. Show the stepwise working for T = “KITCHEN” and P = “HEN”.

### ✅ Answer
### Naïve Algorithm
```text
NAIVE(T, P)
1. for s = 0 to n-m do
2.      compare P with T[s+1 ... s+m]
3.      if all characters match, report s
```

### Text and Pattern
- Text `T = KITCHEN`
- Pattern `P = HEN`

### Stepwise Matching
Windows of length 3:
1. `KIT` ≠ `HEN`
2. `ITC` ≠ `HEN`
3. `TCH` ≠ `HEN`
4. `CHE` ≠ `HEN`
5. `HEN` = `HEN` ✅

### Match Position
Pattern found starting at position **5** (1-based indexing).

### 🎯 Marking Tip
Show every window checked. That is the whole point of naïve matching.

---

## ✏️ Question 126 of 126
**📄 Paper:** [6584]-91 (PE2192)
**🔢 Original Question:** Q8(b)
**⭐ Marks:** 8

### ❓ Full Question
Write a distributed algorithm to find the minimum spanning tree.

### ✅ Answer
### Distributed MST Algorithm Idea
1. Initially each node is a separate fragment.
2. Each fragment finds its **minimum outgoing edge (MOE)**.
3. Fragments connected by MOE merge.
4. Repeat this process until only one fragment remains.
5. That final fragment is the MST.

### Why it Works
The minimum outgoing edge of a fragment is safe to include because of the MST cut property.

### Uses
Useful in communication networks and distributed systems where no single node stores the whole graph.

### 🎯 Marking Tip
Use the terms **fragment**, **minimum outgoing edge**, and **merge**.

---

┌──────────────────────────────────────────────┐
│ ✅ Paper 7 Complete                          │
│ 📎 Answers appended to answer7.md            │
│ 📚 Includes Questions 108 to 126             │
└──────────────────────────────────────────────┘
