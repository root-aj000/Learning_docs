# 📚 Software Testing and Quality Assurance (410245D) — Paper 6 Answer Guide
# 📝 Paper 6 [6354]-496 (PC2379) — Solved Step by Step
# 👨‍🎓 Simple Language | Maximum Marks | Visual Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
# 📄 PAPER 6: [6354]-496 (PC2379)
---

## ✏️ Paper 6 — Q1(a) [6] | Regression Testing Importance
### 🔢 Answer
*Same as Paper 4 Q2(b).* Re-testing after code changes. Ensures changes didn't break existing features. Important: side effects, quality maintenance, catches regression bugs, CI/CD essential. Types: Complete, Selective, Progressive. Automate with Selenium/JUnit/Jenkins.
<!-- END OF QUESTION P6-Q1(a) -->

## ✏️ Paper 6 — Q1(b) [6] | Two Non-Functional Testing Types
### 🔢 Answer
**1. Performance Testing:** Tests speed, response time, stability under load. Types: Load, Stress, Spike, Endurance, Volume. Metrics: response time, throughput, CPU/memory, error rate. Tools: JMeter, LoadRunner. **Example:** 10,000 users → homepage 2.1s (pass), payment 8.2s (fail).

**2. Security Testing:** Tests resistance to attacks. Checks: SQL injection, XSS, CSRF, authentication bypass, encryption, access control. Methods: vulnerability scanning, penetration testing, security auditing. Tools: OWASP ZAP, Burp Suite, Nessus. **Example:** Enter `' OR 1=1 --` in login → if bypassed = SQL injection vulnerability.
<!-- END OF QUESTION P6-Q1(b) -->

## ✏️ Paper 6 — Q1(c) [6] | Statement and Branch Coverage
### 🔢 Answer
*Same as Paper 1 Q2(b).* Statement Coverage = (executed statements / total) × 100%. Branch Coverage = (executed branches / total) × 100%. Branch is STRONGER — 100% branch → 100% statement, not vice versa. Hierarchy: Path > Branch > Statement.
<!-- END OF QUESTION P6-Q1(c) -->

---

## ✏️ Paper 6 — Q2(a) [6] | Two Functional Testing Types
### 🔢 Answer
**1. Smoke Testing (Build Verification Testing):** Quick, basic testing to verify the MOST CRITICAL functions of a new build work. Determines if the build is stable enough for further testing. Like "lighting a smoke" to see if the basic circuitry works. **Example:** After a new build, test: Can app launch? Can user login? Can main page load? If any fails → build is REJECTED, no further testing. Takes 15-30 minutes.

**2. Sanity Testing:** Quick, focused testing on a SPECIFIC area after a small code change or bug fix. Verifies the specific change works and hasn't broken closely related functionality. Narrower scope than regression testing. **Example:** A bug was fixed in the password reset feature. Sanity testing verifies: password reset works now (fix verified) + login still works + registration still works. Does NOT test the entire application.

**Smoke = broad and shallow (all major features briefly). Sanity = narrow and deep (one specific area thoroughly).**
<!-- END OF QUESTION P6-Q2(a) -->

## ✏️ Paper 6 — Q2(b) [6] | Performance Testing with Example
### 🔢 Answer
*Same as Paper 1 Q1(c) Part B and Paper 4 Q2(c).* Tests speed/responsiveness/stability under load. Types: Load, Stress, Spike, Endurance, Volume. Tools: JMeter, LoadRunner. Example: E-commerce with 10,000 users during sale.
<!-- END OF QUESTION P6-Q2(b) -->

## ✏️ Paper 6 — Q2(c) [6] | Dynamic Techniques
### 🔢 Answer
**Dynamic Testing Techniques** involve testing by EXECUTING the software. The program runs with specific inputs and actual outputs are compared with expected outputs.

**Categories:** 1.**Black Box (Specification-Based):** ECP, BVA, Decision Table, State Transition, Use Case Testing. No code knowledge. Tests functionality. 2.**White Box (Structure-Based):** Statement/Branch/Path/Condition Coverage, Loop Testing. Full code knowledge. Tests internal logic. 3.**Experience-Based:** Error Guessing, Exploratory Testing, Checklist-Based. Based on tester skill/intuition.

**Dynamic vs Static:** Dynamic = RUN the software. Static = EXAMINE without running. Dynamic finds runtime bugs (crashes, wrong outputs). Static finds code/design issues (standards violations, complexity).
<!-- END OF QUESTION P6-Q2(c) -->

---

## ✏️ Paper 6 — Q3(a) [6] | QA vs QC
### 🔢 Answer
*Same as Paper 4 Q3(b).* QA = process/proactive/prevention. QC = product/reactive/detection. Both needed together.
<!-- END OF QUESTION P6-Q3(a) -->

## ✏️ Paper 6 — Q3(b) [6] | Selenium IDE
### 🔢 Answer
*Same as Paper 1 Q5(b).* Record-and-playback browser extension. No coding. Selenese commands. Export to code. Assertions vs Verifications. Chrome/Firefox. Quick tests.
<!-- END OF QUESTION P6-Q3(b) -->

## ✏️ Paper 6 — Q3(c) [5] | CMM Levels
### 🔢 Answer
*Same as Paper 3 Q3(c).* 5 Levels: Initial → Repeatable → Defined → Quantitatively Managed → Optimizing.
<!-- END OF QUESTION P6-Q3(c) -->

---

## ✏️ Paper 6 — Q4(a) [6] | Why Software Has Defects
### 🔢 Answer
*Same as Paper 3 Q4(c).* Human error, complex requirements, time pressure, changes, communication, poor design, technology, inadequate testing, no reviews, dependencies, environments.
<!-- END OF QUESTION P6-Q4(a) -->

## ✏️ Paper 6 — Q4(b) [6] | Reliability of Quality Process
### 🔢 Answer
**Reliability** in quality processes means the ability of a process to consistently produce quality outcomes over time. A reliable process gives PREDICTABLE, REPEATABLE results.

**Key Aspects:** 1.**Process Consistency** — Same process followed every time → same quality output. Documented SOPs ensure consistency. 2.**Defect Prevention** — Reliable processes prevent defects at the source rather than catching them later. Root cause analysis feeds back into process improvement. 3.**Measurement & Monitoring** — Reliable processes are measured using metrics (defect density, DRE, process compliance %). Control charts track process stability — points within UCL/LCL = reliable. 4.**Continuous Improvement** — PDCA cycle constantly improves process reliability. Lessons learned from failures improve future performance. 5.**Training** — Well-trained teams execute processes reliably. 6.**Automation** — Automated processes (CI/CD, automated testing) are more reliable than manual ones — eliminate human variability.

**How to Achieve:** Follow standards (ISO 9001, CMMI), conduct audits, track metrics, train teams, automate repetitive tasks, implement PDCA, use control charts for monitoring.
<!-- END OF QUESTION P6-Q4(b) -->

## ✏️ Paper 6 — Q4(c) [5] | Important Aspects of Quality Management
### 🔢 Answer
*Same as Paper 1 Q4(a).* Four pillars: Quality Planning, Quality Assurance (prevent), Quality Control (detect), Quality Improvement (PDCA). Plus: customer focus, management commitment, data-driven decisions, documentation, training.
<!-- END OF QUESTION P6-Q4(c) -->

---

## ✏️ Paper 6 — Q5(a) [6] | Selenium Tool Suite
### 🔢 Answer
*Same as Paper 3 Q5(b).* IDE, WebDriver, Grid, RC (deprecated).
<!-- END OF QUESTION P6-Q5(a) -->

## ✏️ Paper 6 — Q5(b) [6] | Automated Testing Process
### 🔢 Answer
*Same as Paper 1 Q6(b).* 1.Planning 2.Tool Selection 3.Environment Setup 4.Script Development 5.Execution 6.Analysis 7.Maintenance.
<!-- END OF QUESTION P6-Q5(b) -->

## ✏️ Paper 6 — Q5(c) [6] | RPA
### 🔢 Answer
*Same as Paper 1 Q6(c).* Bots automate repetitive tasks. No coding. UiPath, Automation Anywhere, Blue Prism. Attended/Unattended/Hybrid.
<!-- END OF QUESTION P6-Q5(c) -->

---

## ✏️ Paper 6 — Q6(a) [6] | Performance Testing
### 🔢 Answer
*Same as Paper 1 Q1(c) Part B.* Speed, responsiveness, stability under load. Types: Load, Stress, Spike, Endurance, Volume. Tools: JMeter, LoadRunner. Metrics: response time, throughput, CPU, error rate.
<!-- END OF QUESTION P6-Q6(a) -->

## ✏️ Paper 6 — Q6(b) [6] | Selenium WebDriver
### 🔢 Answer
*Same as Paper 1 Q5(c).* Code-based API. Direct browser communication. Java/Python/C#. ChromeDriver/GeckoDriver. Dynamic elements, waits, multi-window.
<!-- END OF QUESTION P6-Q6(b) -->

## ✏️ Paper 6 — Q6(c) [6] | Automated Testing Process
### 🔢 Answer
*Same as Paper 1 Q6(b).* (Duplicate question in this paper.) Planning → Tool Selection → Setup → Development → Execution → Analysis → Maintenance.
<!-- END OF QUESTION P6-Q6(c) -->

---

## ✏️ Paper 6 — Q7(a) [6] | How to Maintain SQA
### 🔢 Answer
*Same as Paper 1 Q8(a).* Standards, audits, reviews, metrics, PDCA, training, config management, defect prevention, customer feedback, management commitment.
<!-- END OF QUESTION P6-Q7(a) -->

## ✏️ Paper 6 — Q7(b) [6] | Six Sigma Characteristics
### 🔢 Answer
*Same as Paper 1 Q7(a).* 3.4 DPMO. DMAIC. Customer focus. Data-driven. Proactive. Belt roles. Variation reduction. Continuous improvement.
<!-- END OF QUESTION P6-Q7(b) -->

## ✏️ Paper 6 — Q7(c) [5] | Flowcharts vs Control Charts
### 🔢 Answer
**Flowchart:** Process diagram showing steps, decisions, flow using boxes/diamonds/arrows. Purpose: understand/improve processes, identify bottlenecks. **Control Chart:** Line graph tracking data over time with UCL/LCL/CL. Purpose: determine if process is STABLE (points within limits) or UNSTABLE (points outside limits). Flowchart = HOW the process works. Control chart = IS the process stable/consistent.
<!-- END OF QUESTION P6-Q7(c) -->

---

## ✏️ Paper 6 — Q8(a) [6] | Total Quality Management
### 🔢 Answer
*Same as Paper 3 Q8(a).* All members participate. Customer focus, total involvement, process-centered, continuous improvement (Kaizen), fact-based, communication. Tools: PDCA, Ishikawa, 5 Whys.
<!-- END OF QUESTION P6-Q8(a) -->

## ✏️ Paper 6 — Q8(b) [6] | Run Charts vs Control Charts
### 🔢 Answer
*Same as Paper 3 Q8(b).* Run = data over time, trends, no limits. Control = data over time WITH UCL/LCL/CL, process stability, out-of-control detection.
<!-- END OF QUESTION P6-Q8(b) -->

## ✏️ Paper 6 — Q8(c) [5] | Ishikawa Flowchart vs Histogram
### 🔢 Answer
*Same as Paper 1 Q7(b).* Flowchart = process sequence visualization. Histogram = data distribution bar chart. Flowchart = PROCESS. Histogram = DATA.
<!-- END OF QUESTION P6-Q8(c) -->

---
---

