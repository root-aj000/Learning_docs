# Phase 5: Testing and Code Quality

Testing isn't about achieving perfect coverage — it's about giving yourself confidence that your code works and doesn't break unexpectedly. This chapter teaches you how to research and build a testing strategy that makes sense for YOUR project.

---

## 5.1 — Do You Need Tests? (Research the Answer)

Not every project needs extensive testing. The answer depends on your project's risk profile.

### Questions to Research About Your Project

1. **What breaks if the code is wrong?**
   - Personal use, low impact → minimal or no tests
   - Used by others / handles money / handles user data → testing matters
   - Public-facing / customer-facing → testing is critical

2. **How much will you change this code later?**
   - Stable, one-time script → don't test it
   - Core business logic that will evolve → test it
   - Library or API others depend on → test it thoroughly

3. **How confident are you in your own code?**
   - You're new to this domain → tests help you learn and catch mistakes
   - You're experienced in this domain → tests save refactoring time

### The Solo Developer's Testing Decision

Use this quick research method:
- **If your project would cause real harm if it broke** → research testing and add a minimal suite
- **If your project is a personal experiment** → skip testing or add one simple test per component
- **If you're learning a new domain** → add tests as you go; they help you understand the code

---

## 5.2 — Finding Testing Tools for Your Stack

### How to Research Testing Tools

Search pattern: `"[language] + [framework] + testing guide"` or `"[language] + testing best practices 2025"`.

### Testing Types by Effort Level

| Testing Type | What It Tests | Effort | When You Need It |
|---|---|---|---|
| No tests | Nothing | Zero | Personal scripts, throwaway experiments |
| Manual testing | "Does it work?" by using it | Low | Personal projects, prototypes |
| Automated unit tests | Individual functions/methods | Medium | Anything beyond throwaway code |
| Integration tests | Parts working together | Medium-High | Multi-component projects |
| End-to-end tests | Full user workflows | High | Public-facing apps with paid users |

### How to Find Testing Tools for YOUR Stack

1. Search `"[language/framework] + testing tutorial"`
2. Find the one that appears in all top results (the standard choice)
3. Check the 5-point library checklist (from Phase 2)
4. Run the quickstart — get one test passing
5. Add more tests as you build features

### Common Testing Tools by Language Ecosystem

| Ecosystem | Unit Testing | Integration Testing | E2E Testing |
|---|---|---|---|
| JavaScript/Node.js | Jest, Vitest | Supertest | Playwright, Cypress |
| Python | pytest | pytest + fixtures | Playwright |
| Go | built-in testing | built-in testing | external tools |
| Ruby | built-in (RSpec/Minitest) | built-in | Capybara |
| Rust | built-in (cargo test) | built-in | e2e tests vary by domain |

**Research method:** Find which testing tool the official docs of your framework recommend. Start there.

---

## 5.3 — The Pragmatic Testing Strategy for Solo Devs

You don't need all testing types. Here's the research-driven approach:

### Step 1: Start with One Test
Pick the most critical function/method in your project. Write ONE test for it. Research how to write a simple test in your testing framework. Get it passing. Commit it.

### Step 2: Expand as Needed
When you add a feature that could break existing functionality, write a test for it. Don't try to test everything upfront.

### Step 3: Research "What Should I Test?"

The answer varies by project type. Research these patterns:
- `"[your framework] + what to test"` or `"[your framework] + testing best practices"`
- Focus on: business logic, API endpoints, data transformation functions
- Skip testing: UI layout details, third-party integrations (test your integration, not their library), trivial functions

### The "Testing When Stuck" Method

When you find a bug:
1. Write a test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Commit both the test and the fix

This method teaches you testing organically through real problems rather than abstract principles.

---

## 5.4 — Linting and Formatting

Linting catches mistakes before you run code. Formatting keeps your code consistent and readable.

### How to Research Linting and Formatting Tools

Search for: `"[your language] + linter"` and `"[your language] + formatter"`.

The results will usually point to one dominant tool per language. Here are common ones:

| Language | Linter | Formatter |
|---|---|---|
| JavaScript | ESLint | Prettier |
| Python | ruff, pylint | Black |
| Go | built-in (golangci-lint) | gofmt |
| TypeScript | ESLint + typescript-eslint | Prettier |
| Rust | Clippy (built-in) | rustfmt |

### Setup Method

1. Install the linter and formatter for your language
2. Run them — look at the output
3. Fix the issues your existing code has (or configure them to ignore non-critical rules)
4. Add a pre-commit hook or editor integration so they run automatically
5. **Research this:** `"how to set up [your linter] in your editor"`

These take 30 minutes to research and set up, and they save hours of fixing mistakes later.

---

## 5.5 — Type Safety as a Research Tool

Type systems are not just for preventing bugs — they're also a research tool that helps you understand how your code should be structured.

### How to Research Type Safety for Your Stack

Search for: `"[your language] + does it have types"` or `"[your language] + typing guide"`.

- **TypeScript** (JavaScript) → adds types to JavaScript
- **Python** → mypy for optional static typing
- **Go** → types are built into the language
- **Rust** → types are built into the language

### The Solo Developer's Approach to Types

1. Start with dynamic typing (no type annotations) to get things working fast
2. Add types gradually when:
   - You're confused about what a function returns
   - You keep getting type-related errors at runtime
   - You're refactoring and want safety nets
3. Use your editor's type hints to understand unfamiliar libraries faster

**Research method:** Search `"[library name] + TypeScript types"` or `"[library name] + type hints"` to see if it has type support.

---

## 5.6 — Code Review as Self-Research

You don't have a team for code review — you do it yourself. The research method is treating your own code like someone else's.

### The Solo Developer Code Review Checklist

Before you finish a feature or commit, research and check these:

1. **Does this code do what I intended?** (test it manually or with automated tests)
2. **Are there any hard-coded values that should be configuration?**
3. **Are errors handled (not silently ignored)?**
4. **Is the code readable? (Could another developer understand it?)**
5. **Are there unused imports or dead code?**
6. **Could this be simpler? (Am I over-engineering?)**

**Research method for each item:** when unsure, search for `"[topic] + best practices"` for the specific item you want to review.

---

## Research Task for Phase 5

1. Research whether YOUR project needs tests (answer the questions in Section 5.1)
2. Find the standard testing tool for your stack (Section 5.2)
3. Write ONE test for something you already built
4. Set up a linter and formatter (Section 5.4)
5. Run them on your existing code — fix what they flag

The goal isn't a perfect test suite. The goal is building the habit of research-driven quality practices.