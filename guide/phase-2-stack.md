# Phase 2: Choosing Your Technology Stack

Every project needs technology decisions. This chapter does not give you a "best stack for everything" — that doesn't exist. Instead, it teaches you how to evaluate and choose tools for YOUR specific project.

---

## 2.1 — Understanding What "Choosing a Stack" Actually Means

The goal of choosing a stack is not to find the best tool — it's to find **a good-enough tool that you can start building with today**. The best tool you can't use effectively is worse than a mediocre tool you understand well.

### What Decisions Do You Actually Need to Make?

For most projects, you need to answer these questions:

1. **Language:** What language will you write in? (Most decisions flow from this.)
2. **Frontend:** Does your project need a user interface? If so, what kind?
3. **Backend:** Does your project need a server? What kind of server?
4. **Database:** How will you store data?
5. **Deployment target:** Where will it run?

Each of these questions has a reasonable default. Your job is to find the right default for YOUR project, not to discover the objectively best choice.

---

## 2.2 — The Decision Framework: Matching Your Project's Needs

Instead of memorizing stacks, use this decision flow for each component.

### Step 1: Classify Your Project

What type of project are you building? Common categories:

| Project Type | Examples |
|---|---|
| Static website | Blog, portfolio, documentation |
| Web application | CRM, dashboard, SaaS tool |
| API service | Backend for a mobile app, webhook processor |
| CLI tool | Automation scripts, DevOps utilities |
| Desktop application | Desktop app with GUI |
| Mobile application | iOS/Android app |
| Data processing | ETL pipeline, data analysis script |
| Machine learning | Model training, inference, predictions |

### Step 2: Research the Default for That Type

For each project type, there is a commonly accepted starting stack. The goal is to find it, not because it's the best, but because it's the safest starting point.

**How to research this:**
- Search for `"[project type] + get started + tutorial"` and look at what language/framework the tutorials use
- Look at the most popular GitHub repositories for project type X — what language and tools do they use?
- Check curated "get started" pages from the major framework documentation sites

**The key insight:** you're not choosing based on what's "best" — you're choosing the most popular/documented option for your project type because that gives you the best starting documentation and community support. You can change later.

### Step 3: Narrow Within Your Chosen Type

Once you know the general direction (e.g., "web application"), research:

1. What are the 2-3 most popular frameworks for this type?
2. Which ones have documentation that explains things clearly to someone at your level?
3. Which ones have active communities (you can find help)?

**The evaluation filter (same as the 5-point checklist):**
1. Is it actively maintained?
2. Are there good getting-started docs?
3. Can you find a minimal working example quickly?
4. Does it match YOUR project type?
5. Is there a community you can ask questions in?

Pick the one that passes the most checks. Commit to it.

---

## 2.3 — The 5-Point Library Evaluation Checklist

This is the most important framework in this guide. You will use this checklist every time you encounter a new library, tool, or framework.

When someone (or you) finds a library you might want to use, evaluate it with these five questions:

### 1. Maintenance Health
- **When was the last commit?** Check the repository's commit history. If it's been 2+ years with no activity, skip it.
- **Are issues being responded to?** Browse the issues page. Are maintainers answering questions? Are issues being closed or ignored?
- What is the release frequency? Libraries with regular releases are likely maintained.

### 2. Documentation Quality
- Is there a **getting started** guide (not just API reference)?
- Can you find a **complete minimal example** that you can run?
- Is the documentation up to date (do the examples match the current version)?

### 3. Ecosystem Fit
- Does it play well with tools you're already using? (e.g., does it work with your database, your frontend framework, your deployment platform?)
- Are there clear integration examples?

### 4. Community Signal
- How many people are using it? (Stars, downloads, mentions in tutorials and blog posts)
- Are there active community spaces ( Discord, forums, Stack Overflow tags)?
- Does "how do I do X with [library]" return useful results?

### 5. Hello-World Test
- Can you install it, run a minimal example, and see it work in under 15 minutes?
- If the setup is complex or confusing, that's a warning sign. Not impossible, but you'll need more documentation and troubleshooting skills later.

### How to Use This Checklist

- **3 or more checks passed** → the library is likely a good fit. Use it.
- **2 checks passed** → the library might work, but you'll hit friction. Consider alternatives.
- **1 or fewer checks passed** → skip it and find another option.

This checklist applies to EVERY library you encounter — databases, testing frameworks, deployment tools, ML libraries, CSS frameworks, anything.

---

## 2.4 — How to Evaluate New Technologies as They Emerge

New tools appear constantly. You don't need to track them all. But when a new tool is relevant to your current project, here's how to evaluate it efficiently:

1. **Does this solve a problem you have RIGHT NOW?** If not, ignore it. Future-you can evaluate it when the problem comes up.
2. **Search for real-world usage:** "I used [tool] for [project type]" — do those results look positive?
3. **Check the 5-point checklist** (Section 2.3).
4. **Compare it to the tool you'd normally use for that job** — the new tool needs to be meaningfully better (simpler, faster, more relevant to your use case) to justify switching.
5. **If it's better:** try it on a small part of your project before committing fully.

---

## 2.5 — Avoiding Stack Changes (The Commitment Rule)

Solo developers often switch stacks multiple times on one project. This is expensive — each switch resets your progress.

### How to Commit Once You've Chosen

1. **Give yourself a research deadline.** Set a 2-hour timer for choosing. When it rings, pick the best option you've found and start building.
2. **Commit to the choice for at least one sprint** (1-2 weeks of focused work). Don't reconsider during that window.
3. **If you hit a real wall** that the tool can't overcome (not just "this is harder than I expected"), THEN research alternatives.
4. **Track your switching reasons.** If you switch frequently, note what's driving it. Is it legitimate (the tool genuinely doesn't fit) or emotional (you're bored or anxious)?

---

## Research Task for Phase 2

**For your current project (or a project you want to start):**

1. List what type of project you're building (use the categories in Section 2.2).
2. Search for `"[your project type] + best stack for beginners 2025"` — review the top 3 results.
3. Apply the 5-point evaluation checklist to the first library/framework that catches your eye.
4. If the checklist results are unclear, pick the one with the highest community signal and move to Phase 3.

The goal is to make a decision and move forward, not to find the perfect stack.