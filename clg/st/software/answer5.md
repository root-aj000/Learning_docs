# 📚 Software Testing and Quality Assurance (410245D) — Paper 5 Answer Guide
# 📝 Paper 5 [6181]-112 (P-6562) — Solved Step by Step
# 👨‍🎓 Simple Language | Maximum Marks | Visual Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
# 📄 PAPER 5: [6181]-112 (P-6562)
---

## ✏️ Paper 5 — Q1(a) [6] | Static Test Case Design Techniques
### 🔢 Answer
**Static Test Case Design Techniques** help design tests by examining documents/code WITHOUT executing the software.

**1. Requirement-Based Test Design** — Read requirements document. For each requirement, create test cases that verify the requirement is met. Ensures traceability (every requirement has test cases).

**2. Review-Based Test Design** — During reviews (informal, walkthrough, inspection), reviewers identify potential test scenarios. Review findings become test cases. **Example:** During design review, a reviewer asks "What happens if the user enters a negative quantity?" → this becomes a test case.

**3. Checklist-Based Test Design** — Use checklists from past projects/experience to design tests. Covers commonly missed scenarios. **Example:** Security checklist: SQL injection test, XSS test, CSRF test, authentication bypass test.

**4. Risk-Based Test Design** — Identify high-risk areas and design MORE test cases for those areas. Risk = Probability × Impact. High-risk modules get intensive testing.

**Usefulness:** Catch defects EARLY (before coding). Low cost (no execution needed). Improve requirements/design quality. Complement dynamic testing techniques.
<!-- END OF QUESTION P5-Q1(a) -->

---

## ✏️ Paper 5 — Q1(b) [6] | Dynamic Test Design Techniques
### 🔢 Answer
**Dynamic Test Design Techniques** design tests that are executed by RUNNING the software.

**Black Box (Specification-Based):**
1. **Equivalence Class Partitioning** — Group inputs into partitions; test one from each
2. **Boundary Value Analysis** — Test at boundary edges (min, min+1, max-1, max)
3. **Decision Table Testing** — Test all combinations of conditions/actions
4. **State Transition Testing** — Test state changes in the system
5. **Use Case Testing** — Test based on user scenarios

**White Box (Structure-Based):**
1. **Statement Coverage** — Execute every code statement at least once
2. **Branch Coverage** — Execute every decision outcome (TRUE/FALSE)
3. **Path Coverage** — Execute every possible path through the code
4. **Condition Coverage** — Each condition evaluates to TRUE and FALSE
5. **Loop Testing** — Test loop boundaries (0, 1, n, max iterations)

**Experience-Based:**
1. **Error Guessing** — Guess where bugs are based on experience
2. **Exploratory Testing** — Explore app without pre-written tests
3. **Checklist-Based** — Use experience-based checklists
<!-- END OF QUESTION P5-Q1(b) -->

---

## ✏️ Paper 5 — Q1(c) [6] | Static Techniques
### 🔢 Answer
*Same as Paper 4 Q1(a).* Static techniques test WITHOUT executing code. Types: Reviews (informal, walkthrough, technical review, inspection) and Static Analysis (SonarQube, FindBugs, PMD — find bugs, standards violations, complexity). Benefits: early defect detection, cheaper fixes, improves code quality. Static = examine. Dynamic = execute.
<!-- END OF QUESTION P5-Q1(c) -->

---

## ✏️ Paper 5 — Q2(a) [6] | Ad-Hoc Testing
### 🔢 Answer
**Ad-hoc Testing** is an INFORMAL testing approach performed WITHOUT any formal test plan, test cases, or documentation. The tester randomly tests the application based on their knowledge and intuition.

**Characteristics:** No planning or documentation. No formal test design. Relies on tester's domain knowledge. Done when there is limited time. No expected results defined beforehand. Difficult to reproduce found bugs.

**Types:** 1.**Buddy Testing** — Developer + Tester work together. Developer explains feature, tester tests it immediately. 2.**Pair Testing** — Two testers test the same feature simultaneously, sharing ideas. 3.**Monkey Testing** — Random inputs/actions without any logic — like a monkey randomly pressing buttons.

**Example:** A new feature is deployed. With no time to write test cases, the tester opens the application and randomly tries different inputs: enters special characters in the name field, uploads a 500MB file to a 10MB limit field, clicks the submit button 50 times rapidly, navigates back/forward randomly. Finds: "App crashes when submit clicked 50 times rapidly."

**Advantages:** Quick, no preparation, finds unexpected bugs, useful when time is limited.
**Disadvantages:** No documentation, not reproducible, depends entirely on tester skill, no coverage measurement, bugs may be hard to reproduce.
<!-- END OF QUESTION P5-Q2(a) -->

---

## ✏️ Paper 5 — Q2(b) [6] | Unit Testing and Integration Testing
### 🔢 Answer
*Same as Paper 1 Q1(b).* **Unit Testing:** Test individual functions in isolation. First level. By developers. Uses stubs/drivers. Frameworks: JUnit, pytest, NUnit. **Integration Testing:** Test combined modules together. Tests interfaces. Approaches: Big Bang (all at once), Top-Down (stubs), Bottom-Up (drivers), Sandwich (hybrid).
<!-- END OF QUESTION P5-Q2(b) -->

---

## ✏️ Paper 5 — Q2(c) [6] | Regression Testing Importance
### 🔢 Answer
*Same as Paper 4 Q2(b).* Regression testing re-tests existing features after code changes to ensure nothing broke. Important because: changes cause side effects, maintains quality over time, catches regression bugs, essential in Agile/CI-CD. Types: Complete, Selective, Progressive. Best automated. Tools: Selenium, JUnit, Jenkins.
<!-- END OF QUESTION P5-Q2(c) -->

---

## ✏️ Paper 5 — Q3(a) [6] | Software Characteristics
### 🔢 Answer
**Key Characteristics/Quality Attributes of Software (ISO 25010):**
1. **Functionality** — Does what it should (completeness, correctness, appropriateness)
2. **Reliability** — Performs consistently without failure (maturity, availability, fault tolerance, recoverability)
3. **Usability** — Easy to learn, use, and understand (learnability, operability, user interface aesthetics, accessibility)
4. **Efficiency/Performance** — Fast response, optimal resource usage (time behavior, resource utilization, capacity)
5. **Maintainability** — Easy to modify, fix, improve (modularity, reusability, analyzability, modifiability, testability)
6. **Portability** — Works across platforms (adaptability, installability, replaceability)
7. **Security** — Protects data and resists attacks (confidentiality, integrity, non-repudiation, accountability, authenticity)
8. **Compatibility** — Works with other systems (co-existence, interoperability)
<!-- END OF QUESTION P5-Q3(a) -->

---

## ✏️ Paper 5 — Q3(b) [5] | CMM Levels
### 🔢 Answer
*Same as Paper 3 Q3(c).* 5 Levels: 1.Initial (chaotic) 2.Repeatable (basic PM) 3.Defined (standardized) 4.Quantitatively Managed (measured) 5.Optimizing (continuous improvement).
<!-- END OF QUESTION P5-Q3(b) -->

## ✏️ Paper 5 — Q3(c) [6] | Why Software Has Defects
### 🔢 Answer
*Same as Paper 3 Q4(c).* Human error, complex requirements, time pressure, changing requirements, communication gaps, poor design, technology complexity, inadequate testing, lack of reviews, dependencies, environmental differences.
<!-- END OF QUESTION P5-Q3(c) -->

---

## ✏️ Paper 5 — Q4(a) [6] | Steps in Software Development Process
### 🔢 Answer
**Steps in SDLC:**
1. **Requirement Gathering & Analysis** — Understand what the customer needs. Document in SRS.
2. **System Design** — Architecture, database, UI, module design. Document in SDD.
3. **Implementation (Coding)** — Developers write code based on design.
4. **Testing** — Verify software works correctly. Unit → Integration → System → Acceptance.
5. **Deployment** — Release to production environment. Install, configure, train users.
6. **Maintenance** — Fix bugs, add features, improve performance. Ongoing phase.

**Models:** Waterfall (sequential), V-Model (testing parallel to development), Agile (iterative sprints), Spiral (risk-driven iterations), DevOps (continuous delivery).
<!-- END OF QUESTION P5-Q4(a) -->

## ✏️ Paper 5 — Q4(b) [6] | QA vs QC
### 🔢 Answer
*Same as Paper 4 Q3(b).* QA = process-focused, proactive, prevention. QC = product-focused, reactive, detection. QA ensures processes are followed. QC ensures product meets requirements. Both essential.
<!-- END OF QUESTION P5-Q4(b) -->

## ✏️ Paper 5 — Q4(c) [5] | Pillars of QMS
### 🔢 Answer
**Four Pillars of QMS:** 1.**Quality Planning** — Define objectives, standards, metrics, resources 2.**Quality Assurance** — Prevent defects through process improvement (proactive) 3.**Quality Control** — Detect defects through testing/inspection (reactive) 4.**Quality Improvement** — Continuous betterment using PDCA, Six Sigma, root cause analysis. **Supporting Elements:** Customer focus, management commitment, documentation, training, audits, data-driven decisions.
<!-- END OF QUESTION P5-Q4(c) -->

---

## ✏️ Paper 5 — Q5(a) [6] | How to Choose Automation Tool
### 🔢 Answer
*Same as Paper 3 Q6(b).* Factors: App type, team skills, budget, browser/OS support, CI/CD integration, reporting, community, learning curve, maintenance, scalability, record/playback, framework support.
<!-- END OF QUESTION P5-Q5(a) -->

## ✏️ Paper 5 — Q5(b) [6] | Selenium Tool Suite
### 🔢 Answer
*Same as Paper 3 Q5(b).* 4 components: IDE (record/playback), WebDriver (code-based API), Grid (parallel, Hub+Nodes), RC (deprecated).
<!-- END OF QUESTION P5-Q5(b) -->

## ✏️ Paper 5 — Q5(c) [6] | RPA
### 🔢 Answer
*Same as Paper 1 Q6(c).* Software bots automate repetitive tasks. No coding. Non-invasive. Types: Attended, Unattended, Hybrid. Tools: UiPath, Automation Anywhere, Blue Prism, Power Automate.
<!-- END OF QUESTION P5-Q5(c) -->

---

## ✏️ Paper 5 — Q6(a) [6] | Automated Testing Process
### 🔢 Answer
*Same as Paper 1 Q6(b).* 1.Planning 2.Tool Selection 3.Environment Setup 4.Script Development (POM) 5.Execution 6.Analysis & Reporting 7.Maintenance.
<!-- END OF QUESTION P5-Q6(a) -->

## ✏️ Paper 5 — Q6(b) [6] | Selenium RC
### 🔢 Answer
**Selenium RC (Remote Control)** was the second component of Selenium (before WebDriver). It uses a **Selenium Server** as a PROXY between the test script and the browser. **Architecture:** Test Script → HTTP → Selenium RC Server → Browser. The RC server injects JavaScript (Selenium Core) into the browser to control it. **Limitations:** Slower than WebDriver (extra server layer), complex setup, JavaScript security restrictions (same-origin policy), deprecated since Selenium 2.0 (replaced by WebDriver). **Why deprecated:** WebDriver communicates DIRECTLY with the browser (no intermediate server), is faster, more reliable, and uses native browser APIs. **Note:** Selenium RC is now OBSOLETE. All new projects should use WebDriver.
<!-- END OF QUESTION P5-Q6(b) -->

## ✏️ Paper 5 — Q6(c) [6] | Selenium Grid
### 🔢 Answer
*Same as Paper 3 Q6(c).* Enables parallel test execution across multiple machines/browsers/OS. Architecture: Hub (central controller) + Nodes (test machines). Benefits: parallel execution (10x faster), cross-browser, cross-platform, scalable. Grid 4 supports Docker/Kubernetes.
<!-- END OF QUESTION P5-Q6(c) -->

---

## ✏️ Paper 5 — Q7(a) [6] | Total Quality Management
### 🔢 Answer
*Same as Paper 3 Q8(a).* TQM: All members participate in quality. Principles: customer focus, total involvement, process-centered, integrated system, continuous improvement (Kaizen), fact-based decisions, communication. Tools: PDCA, Ishikawa, benchmarking, 5 Whys.
<!-- END OF QUESTION P5-Q7(a) -->

## ✏️ Paper 5 — Q7(b) [6] | Flowchart, Run Charts, Control Charts
### 🔢 Answer
**Flowchart:** Process diagram (boxes, diamonds, arrows) showing sequence of steps, decisions, bottlenecks. **Run Chart:** Line graph plotting data over time. Shows trends, shifts, patterns. No statistical limits. **Control Chart:** Line graph WITH UCL (Upper Control Limit), LCL (Lower Control Limit), and CL (Center Line). Determines process stability. Points within limits = stable. Points outside = out of control → investigate. UCL/LCL typically at ±3 standard deviations from mean.
<!-- END OF QUESTION P5-Q7(b) -->

## ✏️ Paper 5 — Q7(c) [5] | Software Maintenance
### 🔢 Answer
**Software Maintenance** is the process of modifying software AFTER delivery to correct faults, improve performance, or adapt to a changed environment.

**Types:** 1.**Corrective** — Fix bugs/defects found after release. 2.**Adaptive** — Modify software to work with changed environment (new OS, new database, new hardware). 3.**Perfective** — Improve performance, usability, add enhancements based on user feedback. 4.**Preventive** — Refactor code, update documentation, restructure to prevent future problems.

**Maintenance Challenges:** Understanding legacy code, lack of documentation, regression risk, testing complexity, resource allocation (maintenance vs new development). **Maintenance consumes 60-80% of total software costs** over its lifetime.

**Key Activities:** Bug fixing, performance tuning, feature enhancements, security patches, compatibility updates, documentation updates, database maintenance, technology migration.
<!-- END OF QUESTION P5-Q7(c) -->

---

## ✏️ Paper 5 — Q8(a) [6] | Activities to Achieve High Software Quality
### 🔢 Answer
*Same as Paper 2 Q7(a).* Activities: 1.Requirement reviews 2.Design reviews 3.Code reviews/pair programming 4.Static analysis (SonarQube) 5.Comprehensive testing (all levels) 6.CI/CD (automated builds+tests) 7.Defect tracking & root cause analysis 8.Standards (ISO, CMMI) 9.Metrics (defect density, DRE, coverage) 10.Training 11.Configuration management (Git) 12.Customer feedback 13.Risk management 14.Post-release monitoring.
<!-- END OF QUESTION P5-Q8(a) -->

## ✏️ Paper 5 — Q8(b) [6] | Ishikawa's 7 Basic Tools
### 🔢 Answer
*Same as Paper 2 Q8(c).* 1.**Cause-and-Effect (Fishbone)** — Root cause analysis (6M: Man, Machine, Method, Material, Measurement, Environment) 2.**Check Sheet** — Data collection form 3.**Control Chart** — Process stability (UCL/LCL/CL) 4.**Histogram** — Data distribution bar chart 5.**Pareto Chart** — 80/20 rule (sorted bar + cumulative line) 6.**Scatter Diagram** — Correlation between 2 variables 7.**Flowchart** — Process visualization.
<!-- END OF QUESTION P5-Q8(b) -->

## ✏️ Paper 5 — Q8(c) [5] | Task, Goal, Metric in SQA
### 🔢 Answer
*Same as Paper 1 Q8(b).* TGM: Task (what to do), Goal (why), Metric (how to measure). Key tasks: reviews, testing, defect tracking, audits, config management. Metrics: defect density, DRE, coverage, leakage, CSAT.
<!-- END OF QUESTION P5-Q8(c) -->

---
---

