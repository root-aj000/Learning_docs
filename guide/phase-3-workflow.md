# Phase 3: Project Setup and Development Workflow

Before you write application code, you need a foundation. This chapter teaches you how to set up a development workflow by researching what each tool does and finding the right setup for your project — not by giving you a pre-built template.

---

## 3.1 — Setting Up Project Structure

### What Is Project Structure?

Project structure is how files and folders are organized. Good structure makes it easy to find things, understand the project at a glance, and add new code without confusion.

### How to Research the Right Structure

Different languages and frameworks have common conventions. Here's how to find them:

1. **Search for `"[language/framework] + project structure"`** on GitHub or in the official docs. Look at the structure of the getting-started example project.
2. **Search for `"[language/framework] + folder structure best practices"`** — you'll find articles that explain the reasoning behind each folder.
3. **Copy the structure from a well-maintained open-source project** in your chosen stack. Don't invent your own — use conventions that other developers understand.

### The Minimum Viable Structure

You don't need a complex project layout for a solo project. Here's enough to start:

```
project/
├── README.md        ← what this project is and how to run it
├── .gitignore       ← files Git should ignore
├── src/             ← your source code
├── tests/           ← tests (can be empty at first)
└── docs/            ← documentation (optional at first)
```

**Research method:** Every time you add a new component (database, frontend, CI), research what folder it belongs in. Search `"[tool] + where to put files in [framework]"`.

---

## 3.2 — Git and Version Control

Version control is the safety net that lets you experiment without fear. Every solo developer should use Git, even if they're publishing code only once.

### What to Learn First

You do not need to know all of Git. For solo development, focus on these commands and concepts in order:

1. `git init` — start tracking your project
2. `git add` — stage changes for commit
3. `git commit -m "message"` — save a snapshot with a message describing what changed
4. `git log` — view your history (to see what changed when)
5. `git branch` — create a branch for experiments
6. `git checkout <branch>` — switch branches
7. `git merge <branch>` — integrate completed work
8. `git remote add origin <url>` — connect to a remote repository (GitHub, GitLab)
9. `git push` — upload your work to the remote
10. `git pull` — download changes from remote

### How to Research Git Problems

When you hit a Git issue (and you will), learn to search for it effectively:

- **Good search:** `"git how to undo last commit"`
- **Good search:** `"git error fatal not a git repository"`
- **Search the official Git docs first** — they have a comprehensive documentation section at git-scm.com/doc
- **Use Stack Overflow** — search `[git] your error message`
- **Search with exact error messages** — copy-paste error text into your search

### The "Meaningful Commits" Practice

Your commit messages are your project's changelog. Write them so you can understand what happened later:

- Bad: `fixed stuff`
- Bad: `asdf`
- Good: `add user registration endpoint`
- Good: `fix login redirect after logout`
- Good: `update README with setup instructions`

**Research this yourself:** Search for `"git commit message best practices"` and read the top results. Adopt the pattern you find most useful.

---

## 3.3 — Development Environment Setup

### Editor or IDE

Choose one and stick with it for at least your first project. Switching editors constantly slows you down more than any missing feature helps.

**How to research which editor to use:**
1. Search for `"best code editor for [your language]"`
2. Read the top 3 results — what do they say about why it's good?
3. Check if the top choice has extensions/plugins for tools you already plan to use (linters, formatters, debuggers)
4. Download it, install a few basic extensions, and start using it

Popular choices (as of 2025): **VS Code**, **JetBrains IDEs** (PyCharm for Python, IntelliJ for JVM languages), **Neovim** (advanced users).

### Essential Extensions to Research

For your chosen editor, research and install:

1. **Language support** — syntax highlighting, autocomplete, error detection
2. **Linter** — catches common mistakes automatically
3. **Formatter** — keeps your code style consistent
4. **Git integration** — view diffs, commit from the editor

**Research method:** Search `"[editor name] + extensions for [your language]"` — the official extension marketplace and documentation usually list the recommended ones.

### Virtual Environments and Dependency Management

Most languages have a way to isolate project dependencies from your system-wide tools. This prevents version conflicts and makes projects reproducible.

Search for: `"[your language] + how to create virtual environment"` and follow the official documentation. This is one of the first things to set up.

---

## 3.4 — Dependency Management: How to Research Before You Install

Every library you install is a dependency that you now need to maintain, update, and understand. Before installing anything, run it through the 5-point checklist from Phase 2.

### The Dependency Philosophy for Solo Devs

- **One tool per job.** Don't install a library that does 10 things. Choose the one that does the one thing you need best.
- **Start with zero dependencies.** Can you solve the problem with built-in tools? If yes, do that first.
- **Add dependencies reluctantly.** Each one is future maintenance work for you.
- **When you add a dependency, document why.** Add a comment or a section in your README explaining why you chose it.

### How to Evaluate a New Dependency

When you find a library you want to use:

1. Run it through the 5-point checklist (see Phase 2)
2. **Check if it's the standard choice for your stack.** Search `"[language] + standard [library type]"` — if there's a clearly recommended one, start there.
3. **Search for known issues.** "[library name] + problems" or "[library name] + alternatives"
4. **Install it in a test project first.** Don't add to your main project until you've verified it works for your use case.

---

## 3.5 — Essential Command-Line Tools (Research-Based)

You don't need to master the terminal, but you need enough to be productive. Research these one at a time:

1. **File navigation** — `cd`, `ls`, `pwd`, `mkdir`, `rm` — learn these with your OS's official docs
2. **File searching** — `grep` or `find` — search for "how to grep recursively" and follow a tutorial
3. **Process management** — how to see running programs, kill processes, check ports
4. **Package manager** — the tool your language uses (npm, pip, cargo, etc.) — its official docs cover the basics

**Research method for CLI tools:**
- Don't try to learn the full manual for any tool you'll use.
- Search for `"[command] + [your specific problem]"` — learn the command by doing.

---

## Research Task for Phase 3

**Set up your project foundation right now:**

1. Initialize a Git repository for your project
2. Create a minimal project structure (README.md, .gitignore, src/)
3. Research and install the basic editor extensions for your language
4. Set up a virtual environment for your project
5. Commit these initial setup files with a meaningful message: "Initial project setup"

The goal is not a perfect setup. The goal is a working foundation you can build on.