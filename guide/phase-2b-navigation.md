# Phase 2b: Navigating Unfamiliar Domains

Every solo developer eventually needs to work in a domain they have zero experience in — machine learning, DevOps, game development, data engineering, etc. This chapter gives you a reusable framework for entering any unfamiliar technical domain efficiently, without getting lost in the overwhelming amount of information available.

The example in this chapter uses machine learning, but the framework applies to any domain.

---

## 2b.1 — Start by Classifying the Problem

The first mistake solo developers make: jumping to tools before understanding what kind of problem they're solving. Different problems need different approaches.

### Your Problem Classification Checklist

Look at your project and answer these questions in plain language:

1. **What am I trying to produce?**
   - A category/label? (e.g., "is this email spam or not?")
   - A number/quantity? (e.g., "what will the price be tomorrow?")
   - A group of similar items? (e.g., "group these customers together")
   - A sequence of outputs? (e.g., "what text comes next?")
   - An unusual detection? (e.g., "is this transaction suspicious?")
   - A recommendation? (e.g., "what product should this user see next?")

2. **What does the input data look like?**
   - Structured tabular data (rows and columns)
   - Text (sentences, documents)
   - Images (photos, diagrams)
   - Audio (voice, sound)
   - Time series (data points over time)
   - Graphs (networks of relationships)

3. **What are my constraints?**
   - How much data do I have?
   - How fast does it need to run?
   - Does it need to run on a phone/server/desktop?
   - Does the result need to be explainable to a human?

Write down your answers. They tell you which domain bucket you're in.

### The Problem-Bucket Map

| Your Answers Point To | Domain |
|---|---|
| "predict a number" or "predict a category" with tabular data | Traditional machine learning |
| "understand or generate text" | Natural language processing (NLP) |
| "understand or generate images" | Computer vision |
| "find unusual patterns" | Anomaly detection |
| "find the best option among many" | Optimization |
| "model data over time" | Time series forecasting |
| "recommend things to users" | Recommendation systems |

This map doesn't give you an algorithm — it gives you a starting direction. The next step is finding tools.

---

## 2b.2 — The Tool Lookup (By Problem Bucket, Not By Algorithm)

Don't memorize algorithms. Use this lookup to find the right starting tool for your problem bucket.

### The Lookup Table

| Problem Bucket | Start With | Why This One | When It's Not Enough | Next Step |
|---|---|---|---|---|
| Tabular prediction | scikit-learn | Covers most classification, regression, and clustering tasks with simple APIs | Large datasets or custom model architectures | XGBoost, then PyTorch |
| Text processing (simple) | scikit-learn (with TF-IDF) | Fast baseline for text classification and similarity | You need deep language understanding | HuggingFace Transformers |
| Text generation/understanding | HuggingFace Transformers | Pre-trained models for NLP tasks | You need specialized models or large-scale training | PyTorch, fine-tuning libraries |
| Image classification | HuggingFace (pre-trained models) | Start with what others have trained | You need a custom image model | PyTorch, computer vision libraries |
| Image generation | Diffusers (HuggingFace) | State-of-the-art diffusion models for generating images | N/A for most solo projects | Custom training |
| Data wrangling (small-medium) | pandas | The standard for tabular data manipulation | Data is too large for memory | Polars, DuckDB, or Dask |
| Data wrangling (large) | DuckDB | Handles large datasets without moving to Spark | Need distributed processing | Apache Spark |
| Simple optimization | scipy.optimize | Built-in optimization algorithms for math problems | Non-convex, black-box, or complex optimization | Optuna, evolutionary algorithms |
| Database | Your language's ORM or SQL tool | Works for 90% of projects | Need specialized queries or massive scale | Raw SQL, specialized tools |

**How to use this table:**
1. Identify your problem bucket from Section 2b.1
2. Find the "Start With" tool for that bucket
3. Go to that tool's official documentation and follow the getting-started guide
4. If you hit a wall, check "When It's Not Enough" for the next step

**You do not need to decide the path in advance.** You follow the arrows as you discover what your project actually needs.

---

## 2b.3 — The "Google It Right" Method for Domain Research

Searching effectively in a technical domain is a skill. Most solo developers search poorly and end up on outdated or overly academic content.

### Search Patterns That Work

| Instead of... | Search This... | Why It Works |
|---|---|---|
| "classification algorithms for tabular data" | "best library for classifying tabular data in Python" | Targets tools, not theory |
| "what is the best ML framework" | "scikit-learn vs PyTorch for beginners 2025" | Finds comparison guides written for actual use |
| "how do neural networks work" | "how neural networks work in 5 minutes Python example" | Finds practical explanations with code |
| "image recognition machine learning" | "image recognition example with pretrained model" | Finds ready-to-use solutions |
| "how to deploy ML model" | "how to serve a sklearn model with Flask" | Targets your exact stack |

### Search Filters for Quality

When you search, apply these filters to the results:

1. **Prefer official documentation** — it's the most current source
2. **Prefer tutorials with working code** — you can copy and run them
3. **Prefer recent content** — look for posts from the last 2 years
4. **Prefer community forums over academic papers** — you want practical usage, not theory
5. **Avoid results that are 5+ years old** unless you're learning foundational concepts

### Trusted Research Sources in Order of Reliability

1. **Official documentation** of the tool/library you're using
2. **Curated lists** with recent updates (e.g., maintained GitHub repos like "awesome-[domain]")
3. **Tutorials from well-known platforms** (Real Python, Towards Data Science, docs.python.info, etc.)
4. **Blog posts from working developers** — check the author's credentials
5. **Stack Overflow** — search for your exact error or problem
6. **Research papers** — generally not useful for solo devs doing practical work. See Section 2b.5.

---

## 2b.4 — The 10-Minute Library Evaluation

This is the same 5-point checklist from Phase 2, applied specifically when you've found a new library while exploring an unfamiliar domain.

Can you evaluate this library in 10 minutes?

1. **Getting-started guide exists?** — Open the docs. Do they have a "quickstart" or "getting started"? If not, skip.
2. **Minimal code example?** — Can you find a code snippet that does something useful in < 10 lines? If not, skip.
3. **Recently maintained?** — Check the repository's last commit. Less than 6 months ago = healthy. More than 2 years = probably dead.
4. **Ecosystem fit?** — Does it accept/return formats your project already uses? (e.g., works with pandas DataFrames, returns standard Python types)
5. **Hello-world success?** — Install it, run the minimal example. Did it work in under 15 minutes?

If the library passes 3 or more — it's viable. Try it on a small test case in your project. If it passes fewer — find another option using the same search methods.

---

## 2b.5 — Understanding Research Papers Without Reading Them

Solo developers doing practical work almost never need to read academic papers. You need to know what exists and when to use it.

### The "Abstract and Code" Method

When a paper or concept is mentioned and you want to know if it's relevant to your project:

1. Read only the **abstract** — this is the summary of what the paper claims
2. Read the **conclusions** — this tells you if it actually works well
3. Find the **GitHub repository** linked in the paper (check the paper itself and the abstract)
4. Check if the repo is active (see Section 2b.4 checklist)
5. If the repo is active and has a demo — it might be useful. If not — skip it

### Using Survey Papers and Reviews

Instead of reading individual papers, start with **survey papers** that summarize a field:

- Search for `"survey of [domain] 2024"` or `"review of [domain]"` — these give overviews of the field
- They tell you what approaches exist, what's popular, and what's emerging — without requiring you to read every individual paper

### When You DO Need Deep Paper Reading (Rare)

The only when you should read a full paper:
- You're building an original research project (not a typical solo dev project)
- You've exhausted all practical resources and you still can't solve your problem
- Your project is specifically about reproducing or extending academic work

For 99% of solo development projects, practical tutorials and library documentation are enough.

---

## 2b.6 — The "Project-First" Learning Approach

The fastest way to learn a new domain is to build something small with it immediately.

### The Method

1. **Install the top-recommended tool for your problem bucket** (from Section 2b.2)
2. **Run the official "Hello World"** example exactly as documented
3. **Modify it to use your own data** — even if it's just a dummy CSV file
4. **Hit a wall?** — Search for exactly your error or your specific stuck point
5. **Learn just enough to unblock yourself** — then continue modifying your project
6. **When you're truly stuck** — research the underlying concept at that point

### Why This Works for Solo Developers

- **Theory without practice is forgettable.** You learn concepts when you have an actual problem to apply them to.
- **You avoid the "I read 5 tutorials and can't build anything" trap.** The project gives you context and motivation.
- **Research becomes targeted.** Instead of reading broadly, you search for exactly what you need right now.

### The "Conceptual Gap" Journal

Keep a list of concepts you don't understand but need to. Every time you hit a wall where the fix requires learning a new concept, add it to the list. Then research that specific concept — just enough to unblock your project.

This turns "I don't understand ML" (overwhelming) into "I need to understand cross-validation and overfitting to fix my model" (actionable).

---

## 2b.7 — When to Use Managed Services Before DIY

Not every tool needs to be built from scratch. Many domains have managed services that handle the complexity for you.

| DIY Approach | Managed Alternative | When to Use the Managed One |
|---|---|---|
| Building a model from scratch | Using a pre-trained model via an API | When you need a baseline or the task is common |
| Training custom ML model | Using an AutoML service | When you're learning or the dataset is small |
| Hosting inference servers | Using a model-as-a-service API | When you don't need custom model architectures |
| Setting up databases from scratch | Using managed database services | When you don't need custom configuration |
| Building CI/CD pipeline from scratch | Using platform's built-in deployment | When you just need automated deploys |

**Research method:** Search for `"[task] + managed service"` or `"[task] + API"` and compare the managed option to the DIY option on ease-of-use, cost, and control.

For solo developers, managed services should be the default starting point. Switch to DIY only when the managed service genuinely doesn't meet your needs.

---

## Research Task for Phase 2b

**Practice the navigation framework with a specific domain:**

1. Pick an unfamiliar domain you've heard about but never used (ML, DevOps, data engineering, etc.)
2. Classify your project's place in that domain using the Problem Classification Checklist (2b.1)
3. Find the start-here tool from the Lookup Table (2b.2)
4. Run its official getting-started guide and get a minimal example working
5. Write down what concepts you didn't understand when you hit a wall (your "Conceptual Gap" list)

This gives you a concrete experience of the framework before you need it for your actual project.