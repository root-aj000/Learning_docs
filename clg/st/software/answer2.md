# 📚 Software Testing and Quality Assurance (410245D) — Paper 2 Answer Guide
# 📝 Paper 2 [6263]-92 (PB-2254) — Solved Step by Step
# 👨‍🎓 Simple Language | Maximum Marks | Visual Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


---
# 📄 PAPER 2: [6263]-92 (PB-2254)
---

## ✏️ Paper 2 — Q1(a) [6 marks]
**📚 Topic:** White Box Testing and Grey Box Testing

---

### ❓ Full Question
Explain White box testing and Grey box testing in detail. **[6]**

---

### 🔢 Answer

#### **White Box Testing**
White box testing (also called structural/glass-box/clear-box testing) is a testing technique where the tester has FULL KNOWLEDGE of the internal code, structure, logic, and design of the software.

**Key Points:**
- Tester can see all source code, algorithms, data structures
- Tests internal logic — branches, loops, conditions, paths
- Performed mostly by developers during unit testing
- Techniques: Statement Coverage, Branch Coverage, Path Coverage, Condition Coverage, Loop Testing
- **Example:** Testing an `if-else` block — write test cases that execute BOTH the `if` branch AND the `else` branch

**Advantages:** Thorough code-level testing, finds hidden bugs, ensures all code paths are tested, detects dead/unreachable code.
**Disadvantages:** Requires programming knowledge, time-consuming, cannot test missing requirements (only tests what's coded).

#### **Grey Box Testing**
Grey box testing is a testing technique where the tester has PARTIAL knowledge of the internal structure. The tester knows about the internal architecture, database schemas, or algorithms at a high level but does NOT have full access to the source code.

**Key Points:**
- Combines aspects of BOTH black box (external behavior) and white box (internal knowledge)
- Tester knows: database structure, data flow, architecture diagrams, API documentation — but NOT the detailed source code
- Commonly used in: Integration Testing, Web Application Testing, API Testing
- Techniques: Matrix Testing, Regression Testing, Pattern Testing, Orthogonal Array Testing

**Example:** Testing a web application — the tester knows the database schema (which tables store user data) and can write test cases that verify data is correctly stored in the database after a user action, but they do NOT read the application's source code.

**Comparison:**

| Aspect | White Box | Grey Box |
|--------|-----------|----------|
| Code Knowledge | Full | Partial |
| Performed By | Developers | Testers with some technical knowledge |
| Focus | Code logic and structure | Integration, data flow, architecture |
| Testing Level | Unit Testing | Integration, System Testing |
| Complexity | High | Medium |

---
<!-- END OF QUESTION P2-Q1(a) -->

---

## ✏️ Paper 2 — Q1(b) [6 marks]
**📚 Topic:** Boundary Value Analysis (BVA) and Equivalence Class Partitioning (ECP)

---

### ❓ Full Question
Discuss Boundary Value Analysis and Equivalence Class Partition. **[6]**

---

### 🔢 Answer

Both are **black box test design techniques** — they help design test cases WITHOUT looking at the source code.

#### **Equivalence Class Partitioning (ECP)**

**Definition:** ECP divides all possible inputs into groups (partitions/classes) where all values in a group are expected to behave the SAME WAY. You only need to test ONE value from each partition (because if one works, all should work).

**In simpler words:** Instead of testing every possible input (impossible!), group similar inputs together and test just one from each group.

**Example:** An age field accepts values 18-60.

| Partition | Range | Expected Behavior | Test Value |
|-----------|-------|-------------------|------------|
| Invalid (below min) | < 18 | Error message | 10 |
| Valid | 18-60 | Accepted | 35 |
| Invalid (above max) | > 60 | Error message | 75 |

**Only 3 test cases** instead of testing every number from 0 to 100+!

#### **Boundary Value Analysis (BVA)**

**Definition:** BVA focuses on testing at the BOUNDARIES (edges) of input ranges because bugs are most likely to occur at boundaries rather than in the middle of ranges.

**In simpler words:** Bugs love to hide at edges. If a field accepts 18-60, test the exact boundary values: 17, 18, 19, 59, 60, 61.

**Example:** Age field accepts 18-60.

| Boundary | Value | Expected |
|----------|-------|----------|
| Just below minimum | 17 | Invalid ✗ |
| Minimum | 18 | Valid ✓ |
| Just above minimum | 19 | Valid ✓ |
| Just below maximum | 59 | Valid ✓ |
| Maximum | 60 | Valid ✓ |
| Just above maximum | 61 | Invalid ✗ |

**Comparison:**

| Aspect | ECP | BVA |
|--------|-----|-----|
| Focus | Groups of similar inputs | Boundary/edge values |
| Test values from | Middle of each partition | Edges of each partition |
| Number of tests | Fewer (1 per partition) | More (boundaries of each partition) |
| Bug detection | Finds category-level bugs | Finds off-by-one and boundary bugs |
| Best used | Together with BVA | Together with ECP |

**Best Practice:** Use ECP first to identify partitions, then use BVA to test the boundaries of those partitions.

---
<!-- END OF QUESTION P2-Q1(b) -->

---

## ✏️ Paper 2 — Q1(c) [6 marks]
**📚 Topic:** Functional vs Non-Functional Testing

---

### ❓ Full Question
Differentiate between Functional testing and Non-functional testing. **[6]**

---

### 🔢 Answer

| Aspect | Functional Testing | Non-Functional Testing |
|--------|-------------------|----------------------|
| **Definition** | Tests WHAT the system does — verifies functions work as specified | Tests HOW the system performs — verifies quality attributes |
| **Focus** | Business requirements and features | Performance, security, usability, reliability |
| **Question Answered** | "Does the feature work correctly?" | "Does it work WELL?" |
| **Based On** | Functional requirements / specifications | Non-functional requirements / quality attributes |
| **Examples** | Login test, payment processing, search, data validation | Load testing, stress testing, security testing, usability testing |
| **Techniques** | Black box: BVA, ECP, Decision Tables | Performance tools: JMeter, LoadRunner |
| **When Done** | Throughout development (unit → acceptance) | Usually after functional testing passes |
| **Who Does It** | Testers, developers | Performance engineers, security specialists |
| **Tools** | Selenium, JUnit, TestNG | JMeter, LoadRunner, OWASP ZAP, Burp Suite |
| **Result** | Feature works / doesn't work | Meets performance/quality criteria or not |
| **Example Scenario** | "Can user add item to cart?" → YES/NO | "Can 10,000 users add items simultaneously in <2 sec?" |

**Functional Testing Types:** Unit, Integration, System, Regression, Smoke, Sanity, UAT
**Non-Functional Testing Types:** Performance, Load, Stress, Security, Usability, Compatibility, Reliability, Scalability, Recovery, Volume

---
<!-- END OF QUESTION P2-Q1(c) -->

---

## ✏️ Paper 2 — Q2(a) [6 marks]
**📚 Topic:** Test Case Design Techniques — Informal Reviews, Walkthroughs, Inspection

---

### ❓ Full Question
Explain the following test case design techniques: 1. Informal Reviews 2. Walkthroughs 3. Inspection **[6]**

---

### 🔢 Answer

All three are **static testing techniques** — they find defects by examining documents/code WITHOUT executing the software.

#### **1. Informal Reviews**
- Least formal review type — no documented process.
- A colleague simply looks at the work product (code, document) and provides feedback.
- No formal meeting, no defined roles, no documented results.
- Quick and cheap but least effective at finding defects.
- **Example:** Developer asks a peer: "Can you quickly look at my login function and see if anything looks wrong?"

#### **2. Walkthroughs**
- Semi-formal review led by the AUTHOR of the work product.
- The author presents (walks through) their work step-by-step to a group of peers.
- Peers ask questions, suggest alternatives, and identify potential issues.
- Main purpose: knowledge sharing AND defect finding.
- Moderately structured — may or may not have a scribe (note-taker).
- **Example:** A developer walks through their database design with the team, explaining each table, relationship, and query. Team members ask: "What happens if this field is NULL?"

#### **3. Inspection (Fagan Inspection)**
- MOST FORMAL and structured review process.
- Defined roles: **Moderator** (leads), **Author** (created the work), **Reviewer/Inspector** (examines for defects), **Scribe/Reader** (records defects).
- Defined process: Planning → Overview → Preparation → Inspection Meeting → Rework → Follow-up.
- Defects are formally logged, categorized by severity, and tracked to resolution.
- Most effective at finding defects (60-90% detection rate).
- **Example:** A formal code inspection meeting where 3 reviewers examine 200 lines of payment processing code, each having prepared for 1 hour. The moderator leads the discussion, the scribe records 12 defects found. The author fixes them and the moderator verifies the fixes.

**Comparison:**

| Aspect | Informal Review | Walkthrough | Inspection |
|--------|----------------|-------------|------------|
| Formality | Low | Medium | High |
| Led By | No leader | Author | Moderator |
| Preparation | None | Optional | Mandatory |
| Documented | No | Partially | Fully |
| Defect Finding | Low | Medium | High (60-90%) |
| Cost/Effort | Low | Medium | High |

---
<!-- END OF QUESTION P2-Q2(a) -->

---

## ✏️ Paper 2 — Q2(b) [6 marks]
**📚 Topic:** Cookies Testing

---

### ❓ Full Question
What is Cookies testing? Explain Cookies testing with an example. **[6]**

---

### 🔢 Answer

**What are Cookies?**
Cookies are small text files stored on a user's browser by websites. They contain data like: session IDs (to keep users logged in), preferences (language, theme), shopping cart items, tracking information, and authentication tokens.

**What is Cookies Testing?**
Cookies testing is the process of verifying that cookies created by a web application work correctly — they are created properly, store correct data, expire at the right time, are secure, and do not leak personal information.

**Why Test Cookies?**
- Incorrect cookies can cause: login failures, session hijacking (security breach), data loss, privacy violations, incorrect user experience.

**What to Test:**

**1. Cookie Creation** — Is the cookie created when expected? After login, does a session cookie appear?

**2. Cookie Content** — Does the cookie store correct data? Check name, value, domain, path.

**3. Cookie Expiration** — Does it expire at the correct time? Session cookies should expire when browser closes. Persistent cookies should expire on the set date.

**4. Cookie Deletion** — When user logs out, is the session cookie deleted? When user clears browser data, are all cookies removed?

**5. Cookie Security** — Is the `Secure` flag set? (cookie sent only over HTTPS). Is the `HttpOnly` flag set? (prevents JavaScript access — protects against XSS). Is the `SameSite` attribute set? (prevents CSRF attacks).

**6. Cookie Behavior with Disabled Cookies** — What happens if the user disables cookies in their browser? Does the app show an appropriate message? Does it still function (with reduced features)?

**7. Cross-Browser Cookie Testing** — Do cookies work correctly in Chrome, Firefox, Safari, Edge?

**8. Cookie Size and Count** — Cookies should not exceed browser limits (typically 4KB per cookie, 50 cookies per domain).

**Example — Testing Cookies for an E-commerce Website:**

| Test Case | Action | Expected Cookie Behavior | Result |
|-----------|--------|-------------------------|--------|
| TC1 | User logs in | Session cookie `session_id` created with `HttpOnly` and `Secure` flags | ✓ |
| TC2 | User adds item to cart | Cart cookie stores product ID and quantity | ✓ |
| TC3 | User selects "Remember Me" | Persistent cookie created with 30-day expiry | ✓ |
| TC4 | User logs out | Session cookie deleted | ✓ |
| TC5 | User disables cookies | App shows "Cookies required" message | ✓ |
| TC6 | After 30 days | "Remember Me" cookie expires, user must re-login | ✓ |
| TC7 | Check cookie flags | `Secure`, `HttpOnly`, `SameSite=Strict` all set | ✓ |

---
<!-- END OF QUESTION P2-Q2(b) -->

---

## ✏️ Paper 2 — Q2(c) [6 marks]
**📚 Topic:** Loop Coverage Testing

---

### ❓ Full Question
Discuss Loop coverage testing and types of it in detail. **[6]**

---

### 🔢 Answer

**Definition:** Loop coverage testing is a white box testing technique that specifically focuses on testing LOOPS (for, while, do-while) in the source code. Loops are common sources of bugs — off-by-one errors, infinite loops, boundary problems.

**Why Test Loops?**
- Loops execute code repeatedly — errors get multiplied
- Common bugs: executing one too many or one too few times, infinite loops, incorrect loop variables, incorrect exit conditions

**Types of Loop Testing:**

**1. Simple Loop Testing**
A single loop with no nesting.
```
for (i = 0; i < n; i++) { ... }
```
**Test Cases:**
- Skip the loop entirely (n = 0 iterations)
- Execute exactly 1 iteration
- Execute 2 iterations
- Execute m iterations (typical value)
- Execute (n-1), n, and (n+1) iterations (boundary)

**2. Nested Loop Testing**
A loop inside another loop.
```
for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) { ... }
}
```
**Strategy:** Start from the INNERMOST loop. Fix all outer loops at their minimum values. Test the innermost loop with simple loop tests. Then move outward one level at a time, repeating the process.

**3. Concatenated Loop Testing**
Two or more loops in sequence (one after another), NOT nested.
```
for (i = 0; i < m; i++) { ... }    // Loop 1
for (j = 0; j < n; j++) { ... }    // Loop 2
```
**Strategy:** If the loops are independent, test each using simple loop testing separately. If Loop 2 depends on Loop 1's output, treat them as nested.

**4. Unstructured Loop Testing**
Loops with unconventional structures — `goto` statements, `break` in unusual places, multiple exit points.
**Strategy:** Refactor the code into structured loops first (if possible), then apply appropriate testing. If refactoring is not possible, use path-based testing.

**Minimum Test Cases for Any Simple Loop (Executing 0 to N times):**
1. Zero iterations (skip loop)
2. One iteration
3. Two iterations
4. Typical number of iterations
5. Maximum - 1 iterations
6. Maximum iterations
7. Maximum + 1 iterations (should trigger termination)

---
<!-- END OF QUESTION P2-Q2(c) -->

---

## ✏️ Paper 2 — Q3(a) [4 marks]
**📚 Topic:** Customer Satisfaction

---

### ❓ Full Question
Write a note on Customer Satisfaction. **[4]**

---

### 🔢 Answer

**Customer satisfaction** is the measure of how well a product or service meets or exceeds customer expectations. In software, it means the degree to which users are happy with the software's functionality, performance, usability, and support.

**Why It Matters:**
1. Quality is ultimately defined by the customer — not by specifications
2. Satisfied customers = repeat business + referrals
3. Dissatisfied customers = churn + negative reviews + revenue loss
4. ISO 9001 requires monitoring customer satisfaction as part of QMS
5. Acquiring new customers costs 5-7x more than retaining existing ones

**How to Measure:**
- **CSAT (Customer Satisfaction Score):** "How satisfied are you?" (1-5 scale)
- **NPS (Net Promoter Score):** "Would you recommend us?" (0-10 scale). Promoters (9-10) - Detractors (0-6) = NPS
- **CES (Customer Effort Score):** "How easy was it to use?" (1-7 scale)
- Support ticket analysis, app store ratings, social media sentiment, churn rate

**How to Improve:**
- Deliver bug-free, high-performance software
- Provide responsive customer support
- Regularly collect and act on feedback
- Continuously improve based on user needs

---
<!-- END OF QUESTION P2-Q3(a) -->

---

## ✏️ Paper 2 — Q3(b) [6 marks]
**📚 Topic:** Requirements of a Product — Stated/Implied and Present/Future

---

### ❓ Full Question
Explain the following requirements of a product: 1. Stated/Implied requirements 2. Present/Future requirements **[6]**

---

### 🔢 Answer

#### **1. Stated Requirements vs Implied Requirements**

**Stated Requirements:**
- Requirements that are EXPLICITLY documented in the requirements specification, contract, or user stories.
- Written down clearly — no ambiguity about what is needed.
- **Example:** "The system shall allow users to reset their password via email OTP."

**Implied Requirements:**
- Requirements that are NOT explicitly documented but are EXPECTED by common sense, industry standards, or user expectations.
- Everyone assumes they will be met even without writing them down.
- **Example:** Nobody writes "the system should not crash" — but everyone expects it. Nobody writes "the login should respond in under 30 seconds" — but users expect fast response.
- Other implied requirements: security (data should be encrypted), usability (UI should be intuitive), reliability (system should not lose data), accessibility (should work for disabled users).

**Key Difference:** Stated = written down. Implied = assumed/expected but not written.

#### **2. Present Requirements vs Future Requirements**

**Present Requirements:**
- Requirements that must be met RIGHT NOW — for the current release/version.
- Address the CURRENT needs of users and the business.
- **Example:** "The app must support English language interface."

**Future Requirements:**
- Requirements that are NOT needed now but will be needed LATER as the product evolves, user base grows, or technology changes.
- Must be considered during design to avoid costly redesigns later.
- **Example:** "The app may need to support 10 languages in the future" → design the system with internationalization (i18n) support from the start, even though only English is needed now.
- Other examples: Scalability (handle 10x more users in the future), integration with future platforms, compliance with upcoming regulations.

**Key Difference:** Present = needed now. Future = anticipated needs for later.

**Importance:** A quality product satisfies ALL four types — stated + implied + present + future requirements.

---
<!-- END OF QUESTION P2-Q3(b) -->

---

## ✏️ Paper 2 — Q3(c) [8 marks]
**📚 Topic:** Waterfall Model — Diagram and Limitations

---

### ❓ Full Question
With neat diagram discuss waterfall model of software development. Also explain its limitations. **[8]**

---

### 🔢 Answer

**What is the Waterfall Model?**
The Waterfall model is the oldest and simplest software development life cycle (SDLC) model. It follows a LINEAR, SEQUENTIAL approach where each phase must be completed before the next phase begins. There is NO going back to a previous phase (like water flowing down a waterfall — it does not flow upward).

**Phases of Waterfall Model:**

```
┌────────────────────┐
│ 1. REQUIREMENTS    │ ← Gather ALL requirements upfront
│    ANALYSIS        │
└────────┬───────────┘
         ↓
┌────────────────────┐
│ 2. SYSTEM DESIGN   │ ← Design architecture, database, UI
└────────┬───────────┘
         ↓
┌────────────────────┐
│ 3. IMPLEMENTATION  │ ← Write the actual code
│    (CODING)        │
└────────┬───────────┘
         ↓
┌────────────────────┐
│ 4. TESTING         │ ← Test the complete system
└────────┬───────────┘
         ↓
┌────────────────────┐
│ 5. DEPLOYMENT      │ ← Release to production
└────────┬───────────┘
         ↓
┌────────────────────┐
│ 6. MAINTENANCE     │ ← Fix bugs, add features
└────────────────────┘
```

**Phase Details:**
1. **Requirements Analysis:** ALL requirements gathered, documented in SRS (Software Requirements Specification). Customer signs off.
2. **System Design:** Architecture, database schema, UI design, module design documented in SDD (Software Design Document).
3. **Implementation:** Developers write code based on design documents.
4. **Testing:** Complete system is tested — unit, integration, system, acceptance testing.
5. **Deployment:** Software delivered to the customer/production environment.
6. **Maintenance:** Bug fixes, patches, enhancements, updates.

**Limitations of Waterfall Model:**

**1. No Going Back** — Once a phase is complete, it is very difficult and costly to go back. If a requirement error is discovered during testing, the entire process must restart from requirements.

**2. Late Testing** — Testing happens ONLY after coding is complete. Defects are found late when they are expensive to fix.

**3. Rigid Requirements** — ALL requirements must be known upfront before development starts. In real projects, requirements often change during development.

**4. No Working Software Until Late** — The customer does not see working software until the testing/deployment phase. If the product does not meet expectations, significant rework is needed.

**5. Poor Risk Handling** — Risks are not identified or addressed until late in the project.

**6. Not Suitable for Complex/Long Projects** — For projects that take months or years, requirements change so much that the delivered software may be outdated.

**7. Assumes Sequential Nature** — Real software development is iterative — phases overlap and feedback loops are natural.

**8. Customer Involvement Limited** — Customer is involved only at the beginning (requirements) and end (acceptance). No feedback during development.

**When to Use Waterfall:**
- Small, simple projects with well-understood requirements
- Requirements are fixed and unlikely to change
- Technology is well-known and stable
- Short-duration projects

---
<!-- END OF QUESTION P2-Q3(c) -->

---

## ✏️ Paper 2 — Q4(a) [8 marks]
**📚 Topic:** Types of Products Based on Criticality

---

### ❓ Full Question
Give types of products based on the basis of criticality to the user. Explain each type with proper example. **[8]**

---

### 🔢 Answer

Products can be classified based on how critical (important) they are to the user's life, safety, business, or well-being:

**Type 1: Safety-Critical Products**
- Failure can cause LOSS OF LIFE, physical injury, or environmental disaster.
- Highest quality standards required. Extensively tested. Certified by regulatory bodies.
- **Examples:** Medical devices (pacemakers, MRI machines), aviation software (autopilot, air traffic control), nuclear power plant control systems, automotive braking systems (ABS), railway signaling systems.
- **Testing:** Formal verification, extensive simulation, regulatory compliance testing (DO-178C for aviation, IEC 62304 for medical).

**Type 2: Mission-Critical Products**
- Failure does NOT cause physical harm but causes MAJOR BUSINESS/OPERATIONAL DISRUPTION.
- Organization cannot function without these systems. Downtime = significant financial loss.
- **Examples:** Banking transaction systems, stock exchange trading platforms, airline reservation systems, e-commerce platforms (Amazon), telecom billing systems.
- **Testing:** High availability testing, disaster recovery, load testing, security testing. Requires 99.99% uptime.

**Type 3: Business-Critical Products**
- Failure causes INCONVENIENCE and FINANCIAL LOSS but the organization can still function using alternative methods.
- Important but not essential for survival.
- **Examples:** Email systems, CRM (Customer Relationship Management) systems, HR management software, inventory management, internal reporting tools.
- **Testing:** Functional testing, performance testing, usability testing. Less rigorous than mission-critical.

**Type 4: Non-Critical (Consumer/Convenience) Products**
- Failure causes MINOR INCONVENIENCE. Users can easily find alternatives.
- Low criticality — bugs are annoying but not harmful.
- **Examples:** Social media apps (Instagram, TikTok), games, entertainment apps, personal to-do lists, weather apps.
- **Testing:** Basic functional testing, usability testing, compatibility testing. Can tolerate some bugs. Frequent updates fix issues.

| Type | Failure Impact | Testing Rigor | Example |
|------|---------------|---------------|---------|
| Safety-Critical | Death/injury | Highest | Pacemaker |
| Mission-Critical | Major business loss | Very High | Banking system |
| Business-Critical | Financial loss | High | CRM software |
| Non-Critical | Minor inconvenience | Moderate | Social media app |

---
<!-- END OF QUESTION P2-Q4(a) -->

---

## ✏️ Paper 2 — Q4(b) [6 marks]
**📚 Topic:** Problematic Areas in SDLC

---

### ❓ Full Question
Discuss problematic areas in software development life cycle. **[6]**

---

### 🔢 Answer

**Problematic areas in SDLC:**

**1. Incomplete/Ambiguous Requirements**
- Requirements are vague, contradictory, or incomplete.
- Different stakeholders have different expectations.
- Leads to building the wrong software or constant requirement changes.
- **Impact:** Rework, delays, cost overruns, customer dissatisfaction.

**2. Poor Communication**
- Communication gaps between stakeholders, developers, testers, and management.
- Requirements misunderstood, design decisions not communicated, bug reports unclear.
- **Impact:** Wrong features built, duplicated effort, missed deadlines.

**3. Unrealistic Deadlines and Budget**
- Management sets aggressive timelines without consulting the technical team.
- Developers cut corners to meet deadlines — skip testing, skip code reviews, write poor code.
- **Impact:** Low quality software, high defect rate, technical debt.

**4. Insufficient Testing**
- Testing starts too late or gets squeezed when development runs behind schedule.
- Inadequate test coverage — critical scenarios not tested.
- **Impact:** Bugs reach production, customer complaints, costly hotfixes.

**5. Scope Creep**
- New features keep getting added during development without adjusting timelines or budget.
- "Just one more feature" syndrome.
- **Impact:** Delays, burnout, quality compromised to fit extra features.

**6. Lack of Documentation**
- Code not documented, design decisions not recorded, test cases not maintained.
- Knowledge resides only in people's heads — when they leave, knowledge is lost.
- **Impact:** Difficult maintenance, onboarding challenges, repeated mistakes.

**7. Poor Change Management**
- Changes to requirements, design, or code not properly tracked and controlled.
- No version control discipline, no change approval process.
- **Impact:** Conflicting changes, regression bugs, lost work.

**8. Inadequate Risk Management**
- Risks not identified, assessed, or mitigated early.
- Technical risks (new technology), resource risks (key person leaves), schedule risks.
- **Impact:** Surprises that derail the project.

---
<!-- END OF QUESTION P2-Q4(b) -->

---

## ✏️ Paper 2 — Q4(c) [3 marks]
**📚 Topic:** Limitations of CMM

---

### ❓ Full Question
List and explain limitations of Capability Maturity Models (CMM). **[3]**

---

### 🔢 Answer

**Limitations of CMM:**

1. **Focuses on Process, Not Product** — CMM assesses the PROCESS maturity but does not directly measure the QUALITY of the software product. A company at CMM Level 5 can still produce poor software if processes are followed without understanding.

2. **Expensive and Time-Consuming** — Achieving higher CMM levels requires significant investment in documentation, training, audits, and process improvement. Small companies may not afford it.

3. **Document-Heavy / Bureaucratic** — CMM requires extensive documentation. This can slow down development and create a culture of "following the paper trail" rather than actually improving.

4. **Does Not Guarantee Innovation** — CMM focuses on repeatable, controlled processes. This can stifle creativity and innovation, making it harder to adopt new approaches like Agile.

5. **One-Size-Fits-All Problem** — CMM prescribes the same maturity framework regardless of project size, type, or domain. What works for a large defense project may not work for a small mobile app.

6. **Assessment Subjectivity** — CMM assessments involve human judgment, which can be subjective. Different assessors may rate the same organization differently.

---
<!-- END OF QUESTION P2-Q4(c) -->

---

## ✏️ Paper 2 — Q5(a) [6 marks]
**📚 Topic:** Manual vs Automation Testing

---

### ❓ Full Question
Differentiate between Manual Testing and Automation Testing. **[6]**

---

### 🔢 Answer

| Aspect | Manual Testing | Automation Testing |
|--------|---------------|-------------------|
| **Definition** | Humans execute test cases manually without tools | Software tools execute pre-scripted tests automatically |
| **Speed** | Slow — depends on human speed | Fast — scripts run in seconds/minutes |
| **Accuracy** | Prone to human errors (fatigue, distraction) | Highly accurate — executes exactly as scripted |
| **Initial Cost** | Low — no tool investment | High — tool licenses, script development |
| **Long-term Cost** | High — repetitive human effort every cycle | Low — reusable scripts, unattended execution |
| **Reusability** | None — must redo everything each cycle | High — scripts reused across builds/releases |
| **Best For** | Exploratory testing, usability, ad-hoc, UI visual testing | Regression, performance, load, repetitive, data-driven |
| **Skill Needed** | Domain knowledge, testing skills | Programming + testing skills |
| **CI/CD Integration** | Not possible | Fully integrates with CI/CD pipelines |
| **Human Judgment** | Yes — can evaluate look/feel, creativity | No — cannot judge aesthetics or UX |
| **Maintenance** | No script maintenance needed | Scripts need updating when app changes |
| **Tools** | None (just test cases + brain) | Selenium, Appium, JMeter, TestNG |
| **Suitable Scale** | Small projects, infrequent testing | Large projects, frequent testing, many test cases |
| **24/7 Execution** | No — humans need rest | Yes — runs overnight, weekends |

**When to Use Manual:** New features, exploratory testing, usability testing, short-term projects, one-time tests.
**When to Use Automation:** Regression, performance, CI/CD, data-driven, cross-browser, tests run >3 times.

---
<!-- END OF QUESTION P2-Q5(a) -->

---

## ✏️ Paper 2 — Q5(b) [6 marks]
**📚 Topic:** Benefits of Automation Testing

---

### ❓ Full Question
List and explain benefits of Automation testing. **[6]**

---

### 🔢 Answer

1. **Faster Execution** — Tests run in minutes vs hours for manual. A suite of 500 tests completes in 30 min instead of 3 days.
2. **Reusability** — Scripts written once, reused across releases/builds without rewriting.
3. **Better Coverage** — Can execute thousands of test cases covering more scenarios than manual testers could.
4. **24/7 Unattended** — Runs overnight, weekends, holidays without human presence.
5. **Consistency** — Executes identically every time — no human errors, no skipped steps.
6. **CI/CD Integration** — Automated tests trigger on every code commit (Jenkins, GitHub Actions), catching bugs immediately.
7. **Cost-Effective Long-Term** — High initial investment but saves significantly over repeated test cycles.
8. **Regression Testing** — Makes frequent regression testing practical — run all tests after every change.
9. **Parallel Execution** — Tests run simultaneously on multiple browsers/devices (Selenium Grid).
10. **Data-Driven Testing** — Same test with hundreds of data combinations (from CSV/Excel/DB).
11. **Early Bug Detection** — Bugs found within minutes of code commit = cheapest to fix.
12. **Detailed Reporting** — Automatic reports with screenshots, execution time, pass/fail trends.

---
<!-- END OF QUESTION P2-Q5(b) -->

---

## ✏️ Paper 2 — Q5(c) [6 marks]
**📚 Topic:** Performance Testing — Uses

---

### ❓ Full Question
What is Performance testing? Explain the uses of it as well. **[6]**

---

### 🔢 Answer

**Performance Testing** evaluates speed, responsiveness, stability, and resource usage under a particular workload.

**Uses of Performance Testing:**

1. **Verify Response Times** — Ensure pages load within acceptable limits (e.g., <3 seconds). Users abandon slow apps.
2. **Determine Maximum Capacity** — Find the maximum number of concurrent users the system can handle before degradation.
3. **Identify Bottlenecks** — Pinpoint which component (database, network, server CPU, memory) is the performance bottleneck.
4. **Validate SLAs** — Verify the system meets Service Level Agreements (e.g., 99.9% uptime, <2s response).
5. **Stress Testing** — Determine what happens under extreme load — does it crash gracefully with a proper error message or hang?
6. **Endurance/Soak Testing** — Find memory leaks and resource degradation over extended periods (run for 72 hours).
7. **Scalability Assessment** — Determine if the system can scale from 1,000 to 100,000 users by adding resources.
8. **Capacity Planning** — Plan infrastructure (servers, bandwidth) needed for expected growth.
9. **Pre-Release Validation** — Ensure performance meets requirements before releasing to production.
10. **Comparison/Benchmarking** — Compare performance across different configurations, versions, or competitors.

**Tools:** Apache JMeter (free), LoadRunner, Gatling, Locust, k6.

**Metrics:** Response time, throughput (requests/second), concurrent users, CPU/memory utilization, error rate.

---
<!-- END OF QUESTION P2-Q5(c) -->

---

## ✏️ Paper 2 — Q6(a) [6 marks] | **Topic:** Automation Testing with Example
### 🔢 Answer
Automation testing uses software tools to execute pre-scripted tests automatically. **Example:** Using Selenium WebDriver in Python to test a login page — script opens Chrome, navigates to login URL, enters username "admin" and password "pass123", clicks Login, verifies "Welcome" message appears, closes browser. This script runs automatically every night after developers commit code — catching login bugs immediately without any human clicking.
<!-- END OF QUESTION P2-Q6(a) -->

---

## ✏️ Paper 2 — Q6(b) [6 marks] | **Topic:** Automated Testing Process (with diagram)
### 🔢 Answer
**Process:** 1.Planning (what to automate, ROI) → 2.Tool Selection (Selenium/Appium/JMeter) → 3.Environment Setup (install tools, frameworks) → 4.Script Development (write code, POM framework) → 5.Execution (manual/scheduled/CI-CD triggered) → 6.Result Analysis (pass/fail, screenshots, defect logging) → 7.Maintenance (update scripts as app changes).
*Refer to Paper 1 Q6(b) for detailed explanation and diagram.*
<!-- END OF QUESTION P2-Q6(b) -->

---

## ✏️ Paper 2 — Q6(c) [6 marks]
**📚 Topic:** Apache JMeter

---

### ❓ Full Question
Describe Apache JMeter based on: 1. Aim/Purpose 2. Working 3. Advantages **[6]**

---

### 🔢 Answer

#### **1. Aim/Purpose**
Apache JMeter is a free, open-source performance testing tool designed to test the performance, load, and stress of web applications and other services. Originally designed for testing web applications, it now supports: HTTP/HTTPS, SOAP/REST APIs, FTP, JDBC (databases), LDAP, JMS, Mail (SMTP/POP3/IMAP), and more.

#### **2. Working**
1. **Create Test Plan** — Define what to test (which URL, which API, which database query).
2. **Add Thread Group** — Define virtual users (threads): number of users, ramp-up time, loop count. Example: 100 users, ramp up over 60 seconds, each loops 10 times.
3. **Add Samplers** — Define requests: HTTP Request (URL, method, parameters), JDBC Request, FTP Request.
4. **Add Listeners** — Define how to view results: Summary Report, Graph Results, View Results Tree, Aggregate Report.
5. **Run the Test** — JMeter simulates the configured virtual users sending requests to the server.
6. **Analyze Results** — View response times, throughput, error rates, graphs.

```
[Thread Group: 100 Users]
    → [HTTP Request: GET /homepage]
    → [HTTP Request: POST /login]
    → [HTTP Request: GET /dashboard]
    → [Listener: Summary Report]
    → [Listener: Graph Results]
```

#### **3. Advantages**
1. **Free and Open Source** — No licensing cost. Active community support.
2. **Cross-Platform** — Runs on Windows, Mac, Linux (Java-based).
3. **Multi-Protocol** — Supports HTTP, FTP, JDBC, SOAP, REST, LDAP, JMS.
4. **GUI + CLI Modes** — GUI for test creation, CLI (non-GUI) for large-scale execution.
5. **Distributed Testing** — Can run tests from multiple machines simultaneously for massive load generation.
6. **Extensible** — Plugin support for additional protocols and reporting (JMeter Plugins Manager).
7. **Recording Capability** — HTTP(S) Test Script Recorder captures browser actions as test steps.
8. **Rich Reporting** — Multiple built-in listeners + HTML dashboard reports.
9. **CI/CD Integration** — Integrates with Jenkins, Maven, Gradle for automated performance testing in pipelines.
10. **Parameterization** — Supports CSV data files for data-driven load testing.

---
<!-- END OF QUESTION P2-Q6(c) -->

---

## ✏️ Paper 2 — Q7(a) [8 marks] | **Topic:** Activities to Achieve High Software Quality
### 🔢 Answer
**Activities:** 1.Requirement Reviews (catch defects at source) 2.Design Reviews/Inspections 3.Code Reviews/Pair Programming 4.Static Analysis (SonarQube, FindBugs) 5.Comprehensive Testing (all levels: unit→acceptance) 6.Continuous Integration/CD (automated builds+tests) 7.Defect Tracking & Root Cause Analysis (Jira, Ishikawa) 8.Process Standards (ISO 9001, CMMI) 9.Metrics & Measurement (defect density, DRE, coverage) 10.Training & Skill Development 11.Configuration Management (Git, version control) 12.Customer Feedback Integration 13.Risk Management (identify+mitigate early) 14.Post-Release Monitoring (production monitoring, alerting).
<!-- END OF QUESTION P2-Q7(a) -->

---

## ✏️ Paper 2 — Q7(b) [6 marks] | **Topic:** Six Sigma Strategy
### 🔢 Answer
Six Sigma is a data-driven methodology targeting 3.4 DPMO (99.9997% perfection). Uses DMAIC: **D**efine (problem+goals), **M**easure (current performance), **A**nalyze (root causes), **I**mprove (implement solutions), **C**ontrol (sustain gains). Roles: Champion, Master Black Belt, Black Belt, Green Belt, Yellow Belt. Focus: customer (VOC/CTQs), data-driven decisions, variation reduction, proactive management, continuous improvement. *See Paper 1 Q7(a) for full details.*
<!-- END OF QUESTION P2-Q7(b) -->

---

## ✏️ Paper 2 — Q7(c) [3 marks] | **Topic:** Histogram, Flowchart, Control Chart
### 🔢 Answer
**Histogram:** Bar chart showing frequency distribution of data (e.g., bug counts by severity). Identifies patterns, outliers, distribution shape. **Flowchart:** Visual diagram of process steps using boxes, diamonds, arrows. Shows process flow, decisions, bottlenecks. **Control Chart:** Line graph with upper/lower control limits (UCL/LCL) and center line. Tracks process performance over time. Points within limits = stable process. Points outside = out of control, investigate.
<!-- END OF QUESTION P2-Q7(c) -->

---

## ✏️ Paper 2 — Q8(a) [6 marks] | **Topic:** ISO 9000 Standard
### 🔢 Answer
**ISO 9000** is a FAMILY of standards for quality management systems published by ISO. Unlike ISO 9001 (requirements for certification), ISO 9000 provides fundamentals, vocabulary, and guidelines. **Family:** ISO 9000 (concepts & vocabulary), ISO 9001 (requirements — used for certification), ISO 9003 (guidelines for performance improvement), ISO 19011 (auditing guidance). **7 Principles:** Customer focus, Leadership, People engagement, Process approach, Improvement, Evidence-based decisions, Relationship management. **Importance:** Provides internationally recognized framework for quality. Certification increases customer confidence. Promotes continuous improvement culture. Required by many clients/governments for contracts.
<!-- END OF QUESTION P2-Q8(a) -->

---

## ✏️ Paper 2 — Q8(b) [5 marks] | **Topic:** SQA Plan
### 🔢 Answer
**SQA Plan** is a document that defines the quality assurance activities, standards, tools, and responsibilities for a software project. **Contents:** 1.Purpose & scope 2.Referenced standards (ISO, IEEE) 3.Organization & responsibilities 4.Documentation requirements 5.Standards, practices, conventions 6.Reviews & audits schedule 7.Testing strategy 8.Defect reporting & tracking procedures 9.Tools & techniques 10.Configuration management 11.Training plan 12.Risk management 13.Metrics to be collected 14.Corrective action procedures. The SQA Plan ensures ALL team members know what quality activities to perform and when.
<!-- END OF QUESTION P2-Q8(b) -->

---

## ✏️ Paper 2 — Q8(c) [6 marks] | **Topic:** Ishikawa's Basic Tools for Quality Control
### 🔢 Answer
**Ishikawa's 7 Basic QC Tools:** 1.**Cause-and-Effect Diagram (Fishbone/Ishikawa)** — Identifies root causes of a problem. Categories: Man, Machine, Method, Material, Measurement, Environment (6M). 2.**Check Sheet** — Structured form for collecting data systematically. Tally marks for defect types. 3.**Control Chart** — Line graph with UCL/LCL tracking process stability over time. 4.**Histogram** — Bar chart showing data distribution/frequency. 5.**Pareto Chart** — Bar chart sorted by frequency + cumulative line. Shows 80/20 rule (80% of problems from 20% of causes). 6.**Scatter Diagram** — Plots two variables to find correlation (e.g., code complexity vs bug count). 7.**Flowchart** — Visual diagram of process steps showing flow, decisions, bottlenecks.
<!-- END OF QUESTION P2-Q8(c) -->

---
---

