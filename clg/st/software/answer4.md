# 📚 Software Testing and Quality Assurance (410245D) — Paper 4 Answer Guide
# 📝 Paper 4 [5927]-353 (PA-921) — Solved Step by Step
# 👨‍🎓 Simple Language | Maximum Marks | Visual Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
# 📄 PAPER 4: [5927]-353 (PA-921)
---

## ✏️ Paper 4 — Q1(a) [6] | Static Techniques
### 🔢 Answer
**Static techniques** test software WITHOUT executing the code. They examine work products (code, documents, designs) manually or with tools to find defects EARLY.

**Types of Static Techniques:**

**1. Reviews (Manual Static Testing):**
- **Informal Review:** Colleague glances at your work. No formal process.
- **Walkthrough:** Author presents work step-by-step to peers.
- **Technical Review:** Peers independently examine work for technical correctness.
- **Inspection (Fagan):** Most formal — defined roles (Moderator, Author, Reviewer, Scribe), defined process (Plan→Overview→Prep→Meeting→Rework→Follow-up). Finds 60-90% of defects.

**2. Static Analysis (Tool-Based):**
- Tools analyze source code WITHOUT executing it.
- Finds: coding standard violations, potential bugs (null pointers, array overflows), security vulnerabilities, code complexity, dead/unreachable code.
- **Tools:** SonarQube, FindBugs, PMD, Checkstyle, ESLint, Pylint.

**Benefits:** Finds defects EARLY (before testing phase = cheaper). No need to execute software. Improves code quality. Enforces standards.

**Static vs Dynamic:** Static = examine without running. Dynamic = test by running the software.
<!-- END OF QUESTION P4-Q1(a) -->

---

## ✏️ Paper 4 — Q1(b) [6] | Error Guessing and Exploratory Testing
### 🔢 Answer
**Error Guessing:** Tester GUESSES where bugs might be based on experience. Uses knowledge of common developer mistakes: division by zero, null inputs, empty fields, special characters, boundary values, max/min values, negative numbers. No formal technique — relies on tester's intuition. **Example:** Testing age field → try: 0, -1, 999, "abc", blank, 1.5, special chars.

**Exploratory Testing:** Simultaneous test design, execution, and learning — tester EXPLORES the app without pre-written scripts. Uses test charters ("Explore checkout focusing on payment errors") and time-boxes (60-90 min sessions). Tester learns app behavior in real-time and creates tests on-the-fly. Highly effective for finding unexpected bugs. **Example:** Tester freely explores shopping cart: adds 99999 items, applies expired coupons, changes currency mid-checkout — finds crash when 2 coupons applied simultaneously.

**Both are experience-based techniques** — depend on tester's skill, not formal methods.
<!-- END OF QUESTION P4-Q1(b) -->

---

## ✏️ Paper 4 — Q1(c) [6] | System Testing and Acceptance Testing
### 🔢 Answer
*Same as Paper 1 Q2(c).* **System Testing:** Complete integrated system tested against requirements. By independent testing team. Tests functional + non-functional. Black box approach. Third level of testing. **Acceptance Testing:** Final level — end users/clients decide if system meets BUSINESS needs. Types: UAT (users test real scenarios), Alpha (at developer site), Beta (at user site with real users), Contract (against contract terms), Regulatory (compliance). Pass = release. Fail = fix.
<!-- END OF QUESTION P4-Q1(c) -->

---

## ✏️ Paper 4 — Q2(a) [6] | Path Coverage and Conditional Coverage
### 🔢 Answer
Both are **white box testing** techniques:

**Path Coverage:** Tests EVERY possible execution path through the code. A path is a unique sequence of statements from entry to exit. Most thorough but most expensive — number of paths can explode exponentially with nested conditions/loops.
```
if A then X else Y    →  Path 1: A→X
if B then P else Q    →  Path 2: A→X, B→P
                          Path 3: A→X, B→Q
                          Path 4: A→Y, B→P (... etc)
```
**Formula:** Path Coverage = (Paths tested / Total paths) × 100%

**Condition Coverage (Conditional Coverage):** Tests that each INDIVIDUAL CONDITION in a decision evaluates to BOTH TRUE and FALSE at least once. Different from branch coverage — focuses on individual conditions within compound decisions.
```
if (A > 0 AND B < 10):    // Compound condition with 2 sub-conditions
   // Test 1: A > 0 = TRUE,  B < 10 = TRUE   → Overall TRUE
   // Test 2: A > 0 = FALSE, B < 10 = FALSE   → Overall FALSE
   // Both conditions tested for TRUE and FALSE ✓
```
**Formula:** Condition Coverage = (Condition outcomes tested / Total condition outcomes) × 100%

**Hierarchy:** Path Coverage > Condition Coverage > Branch Coverage > Statement Coverage (Path is strongest)
<!-- END OF QUESTION P4-Q2(a) -->

---

## ✏️ Paper 4 — Q2(b) [6] | Regression Testing
### 🔢 Answer
**Regression Testing** is re-testing of previously working features after code changes (bug fixes, new features, refactoring) to ensure the changes did NOT break existing functionality.

**Why it's important:**
1. Code changes can have unintended side effects — fixing bug A might break feature B
2. Ensures software quality does not degrade over time
3. Catches "regression bugs" — features that USED to work but now don't
4. Essential in Agile/CI-CD environments where code changes daily

**When to perform:** After every bug fix, after adding new features, after code refactoring, after configuration changes, after library/dependency updates.

**Types:** 1.**Complete Regression** — rerun ALL test cases (time-consuming but thorough) 2.**Selective Regression** — rerun only tests related to the changed area 3.**Progressive Regression** — create new tests for new functionality alongside existing regression tests.

**Best Practice:** AUTOMATE regression tests — they run frequently and must be repeatable. Manual regression is impractical for large test suites.

**Tools:** Selenium (web), Appium (mobile), JUnit/TestNG (unit), Jenkins/GitHub Actions (CI/CD integration).

**Example:** Developer fixes a login bug. Regression testing verifies: login works (fix verified) + registration still works + password reset still works + user profile still works + shopping cart still works (no side effects).
<!-- END OF QUESTION P4-Q2(b) -->

---

## ✏️ Paper 4 — Q2(c) [6] | Performance Testing with Example
### 🔢 Answer
*Same as Paper 1 Q1(c) Part B.* Performance testing evaluates speed, response time, stability, throughput under load. Types: Load (expected), Stress (beyond max), Spike (sudden), Endurance (long), Volume (large data). Tools: JMeter, LoadRunner. **Example:** E-commerce site tested with 10,000 concurrent users during festival sale. Results: Homepage 2.1s (PASS), Search 1.8s (PASS), Payment 8.2s (FAIL — needs optimization), Error rate 3.5% (FAIL), CPU 95% (FAIL — need more servers).
<!-- END OF QUESTION P4-Q2(c) -->

---

## ✏️ Paper 4 — Q3(a) [6] | Why Software Has Defects
### 🔢 Answer
*Same as Paper 3 Q4(c).* Reasons: 1.Human error 2.Complex/ambiguous requirements 3.Time pressure 4.Changing requirements 5.Communication gaps 6.Poor design 7.Technology complexity 8.Inadequate testing 9.Lack of code reviews 10.Third-party dependencies 11.Environmental differences.
<!-- END OF QUESTION P4-Q3(a) -->

---

## ✏️ Paper 4 — Q3(b) [6] | QA vs QC
### 🔢 Answer

| Aspect | Quality Assurance (QA) | Quality Control (QC) |
|--------|----------------------|---------------------|
| **Focus** | PROCESS (how we build) | PRODUCT (what we built) |
| **Approach** | PROACTIVE — prevent defects | REACTIVE — find defects |
| **Activities** | Process audits, standards, training, reviews | Testing, inspections, walkthroughs |
| **Goal** | Ensure correct processes are followed | Ensure product meets requirements |
| **When** | Throughout SDLC | After product is built/during testing |
| **Responsibility** | QA team, process owners | Testing team, QC inspectors |
| **Output** | Process improvements, audit reports | Test reports, defect logs |
| **Analogy** | Teaching students correct study methods | Checking exam papers for errors |
| **Example** | "Are developers following coding standards?" | "Does the login function work correctly?" |
| **Orientation** | Prevention-oriented | Detection-oriented |

**Key:** QA prevents bugs by improving PROCESSES. QC finds bugs by testing PRODUCTS. Both are needed — QA reduces the number of bugs created, QC catches the ones that slip through.
<!-- END OF QUESTION P4-Q3(b) -->

---

## ✏️ Paper 4 — Q3(c) [5] | Quality Management System
### 🔢 Answer
*Same as Paper 3 Q3(a).* QMS is a formalized system documenting processes, procedures, and responsibilities for quality. Pillars: Quality Planning, QA (prevent), QC (detect), Quality Improvement (PDCA). Key elements: management commitment, customer focus, process approach, documentation, audits, corrective actions, training, continuous improvement. Standards: ISO 9001, CMMI, Six Sigma, TQM.
<!-- END OF QUESTION P4-Q3(c) -->

---

## ✏️ Paper 4 — Q4(a) [6] | ISO 9001
### 🔢 Answer
*Same as Paper 1 Q3(c) and Paper 3 Q4(a).*
<!-- END OF QUESTION P4-Q4(a) -->

## ✏️ Paper 4 — Q4(b) [6] | Selenium IDE
### 🔢 Answer
*Same as Paper 1 Q5(b) and Paper 3 Q3(b).*
<!-- END OF QUESTION P4-Q4(b) -->

## ✏️ Paper 4 — Q4(c) [5] | CMM Levels
### 🔢 Answer
*Same as Paper 3 Q3(c).* Level 1: Initial (chaotic). Level 2: Repeatable (basic PM). Level 3: Defined (standardized). Level 4: Quantitatively Managed (measured). Level 5: Optimizing (continuous improvement).
<!-- END OF QUESTION P4-Q4(c) -->

---

## ✏️ Paper 4 — Q5(a) [6] | Selenium IDE
### 🔢 Answer
*Same as Paper 1 Q5(b).* Record-and-playback browser extension. No coding. Selenese commands. Export to code. Chrome/Firefox only.
<!-- END OF QUESTION P4-Q5(a) -->

## ✏️ Paper 4 — Q5(b) [6] | RPA
### 🔢 Answer
*Same as Paper 1 Q6(c).* Software bots automating repetitive tasks. No coding. Non-invasive. Tools: UiPath, Automation Anywhere, Blue Prism.
<!-- END OF QUESTION P4-Q5(b) -->

## ✏️ Paper 4 — Q5(c) [6] | Automated Testing Process
### 🔢 Answer
*Same as Paper 1 Q6(b).* 1.Planning 2.Tool Selection 3.Environment Setup 4.Script Development 5.Execution 6.Analysis & Reporting 7.Maintenance.
<!-- END OF QUESTION P4-Q5(c) -->

---

## ✏️ Paper 4 — Q6(a) [6] | Selenium Tool Suite
### 🔢 Answer
*Same as Paper 3 Q5(b).* 4 components: Selenium IDE (record/playback), WebDriver (code-based, most powerful), Grid (parallel execution, Hub+Nodes), RC (deprecated).
<!-- END OF QUESTION P4-Q6(a) -->

## ✏️ Paper 4 — Q6(b) [6] | Selenium WebDriver
### 🔢 Answer
*Same as Paper 1 Q5(c).* Programming API, direct browser communication, supports Java/Python/C#/Ruby/JS, handles dynamic content, wait mechanisms.
<!-- END OF QUESTION P4-Q6(b) -->

## ✏️ Paper 4 — Q6(c) [6] | Benefits of Automation Testing
### 🔢 Answer
*Same as Paper 1 Q6(a).* 1.Faster 2.Reusable 3.Better coverage 4.24/7 5.Consistent 6.CI/CD 7.Cost-effective 8.Regression 9.Parallel 10.Early detection 11.Reporting 12.Data-driven.
<!-- END OF QUESTION P4-Q6(c) -->

---

## ✏️ Paper 4 — Q7(a) [6] | Ishikawa Flowchart vs Histogram
### 🔢 Answer
*Same as Paper 1 Q7(b).* Flowchart = process diagram (sequence, decisions, bottlenecks). Histogram = bar chart (data distribution, frequency, patterns). Flowchart = PROCESS visualization. Histogram = DATA visualization.
<!-- END OF QUESTION P4-Q7(a) -->

## ✏️ Paper 4 — Q7(b) [6] | Six Sigma Characteristics
### 🔢 Answer
*Same as Paper 1 Q7(a).* 3.4 DPMO. DMAIC. Customer focus. Data-driven. Proactive. Teamwork (Belt roles). Variation reduction. Continuous improvement.
<!-- END OF QUESTION P4-Q7(b) -->

## ✏️ Paper 4 — Q7(c) [5] | How to Maintain SQA
### 🔢 Answer
*Same as Paper 1 Q8(a).* Standards, audits, reviews, metrics, PDCA, training, config management, defect prevention, customer feedback, management commitment.
<!-- END OF QUESTION P4-Q7(c) -->

---

## ✏️ Paper 4 — Q8(a) [6] | Total Quality Management
### 🔢 Answer
*Same as Paper 3 Q8(a).* TQM: All members participate in quality improvement. Principles: customer focus, total employee involvement, process-centered, integrated system, continuous improvement, fact-based decisions, communication. Tools: PDCA, Ishikawa, benchmarking, 5 Whys.
<!-- END OF QUESTION P4-Q8(a) -->

## ✏️ Paper 4 — Q8(b) [6] | Run Charts vs Control Charts
### 🔢 Answer
*Same as Paper 3 Q8(b).* Run chart = data over time (trends, patterns, no control limits). Control chart = data over time WITH UCL/LCL/CL (process stability, out-of-control detection, ±3σ limits).
<!-- END OF QUESTION P4-Q8(b) -->

## ✏️ Paper 4 — Q8(c) [5] | Task, Goal, Metric in SQA
### 🔢 Answer
*Same as Paper 1 Q8(b).* TGM: Task (what), Goal (why), Metric (how to measure). Key tasks: reviews, testing, defect tracking, audits, config management, training. Key metrics: defect density, DRE, coverage, defect leakage, CSAT, compliance %.
<!-- END OF QUESTION P4-Q8(c) -->

---
---

