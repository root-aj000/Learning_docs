# Phase 4: Building the Project

This is where research meets execution. During building, you'll constantly need to assemble different components — frontend, backend, database, authentication, third-party services. This chapter teaches you how to research and integrate each one efficiently.

---

## 4.1 — Frontend Development

### Do You Actually Need a Frontend Framework?

Before reaching for a React/Vue/Angular project, ask yourself:

1. **Does this project need user interaction beyond simple forms?** If it's a CRUD app or a tool with form-based input, plain HTML/CSS/JavaScript might be enough.
2. **Do I need client-side state management?** If the UI needs to react dynamically to user input without page reloads, yes — you need a frontend framework.
3. **Is this a full web app or just a page?** Static sites (blogs, portfolios) don't need frameworks.

### How to Research Your Frontend Approach

**Search pattern:** `"[type of app] + best frontend approach for solo developer"`
**Follow-up search:** `"[your chosen framework] + getting started tutorial"`

### The Frontend Decision Guide

| Your Frontend Need | Research Path | Start With |
|---|---|---|
| Static page, no interactivity beyond links | Plain HTML and CSS | Nothing more needed |
| Simple interactivity, no routing | Plain HTML + a small JS library (Alpine.js, HTMX) | HTMX or Alpine.js |
| Interactive web app with routing and state | A lightweight framework | Astro, Svelte, or Vue (in order of simplicity) |
| Complex dashboard with real-time updates | Full framework with ecosystem | React or Vue (more ecosystem support) |
| Only an API backend, no UI | Skip frontend entirely | Build API only |

### Integrating a Frontend Framework

1. Follow the framework's official getting-started guide — start there, not at a third-party tutorial
2. Build one component. Just one. See if it works.
3. Then research how to connect it to your backend (API integration patterns for your specific stack)
4. Add complexity incrementally

---

## 4.2 — Backend Development

### How to Research Your Backend Approach

Your backend needs depend on what your project does. Research in this order:

1. **How does my frontend talk to the backend?** — Research "REST API" or "API" for your language. This is the primary pattern.
2. **Does my project need a database?** If yes, you need a backend to handle database queries.
3. **Does my project need authentication, file uploads, background jobs, email?** — These are separate backend components.

### The Backend Decision Guide

| Your Backend Need | Research Path | Start With |
|---|---|---|
| Simple API with CRUD operations | Search for "[language] + framework + REST API tutorial" | Your language's most popular web framework |
| Serverless/function-based | Search for "[platform] + serverless functions" | Cloud provider's built-in serverless tools |
| Real-time features (chat, live updates) | Search for "websockets in [your stack]" | Framework-native websocket support or a simple library |
| Background task processing | Search for "task queue in [your language]" | A task queue library in your ecosystem |

### How to Build a Simple API

**Research sequence:**
1. `"[language] + how to create a REST API"` — find the standard approach for your language
2. Follow the official tutorial for the most popular framework in your language
3. Test the API with curl, Postman, or a browser
4. Connect it to your database (search for "[database] + [language] + tutorial")
5. Add one endpoint. Test it. Commit it. Add the next.

**The "build one endpoint at a time" principle:** Don't design a whole API spec upfront. Build one endpoint, test it, then build the next. Research what you need at each step.

---

## 4.3 — Database Selection and Setup

### How to Choose a Database

1. **What's your data shape?**
   - Structured data with relationships (users, orders, posts) → relational database
   - Flexible, nested, or document-like data → document database
   - Key-value pairs (caching, sessions) → key-value store

2. **How much data do you expect?**
   - Less than a few GB → SQLite or a simple managed database is fine
   - More than that → PostgreSQL, or a managed service

3. **Do you need concurrency?**
   - Multiple users writing at the same time → PostgreSQL or a managed database
   - Mostly reads or single-user → SQLite is fine

### Database Decision Flow

- **Are you building a simple app for personal use or a small team?** → SQLite or a small managed DB
- **Are you building a web app with multiple concurrent users?** → PostgreSQL
- **Are you doing analytics on large datasets?** → DuckDB or columnar database

Search for `"[type of app] + recommended database"` and review the top results for your stack. The common choice is almost always fine for solo projects.

### How to Research Database Integration

Search for `"[database engine] + [your language] + tutorial"` or `"[database engine] + [your framework] + setup"`. Use the official documentation of your framework's database integration.

---

## 4.4 — Authentication: Build It or Use a Service?

### The Default Answer

For most solo developers: **use an existing service.** Don't build auth systems from scratch — they're complex, security-sensitive, and you'll learn more about auth research than about your actual project by building one.

### How to Research Auth Services

Search for `"[your language/framework] + authentication library"` or `"[your stack] + auth guide"`.

Common patterns:
- **JavaScript/Node.js:** Auth.js, Passport.js
- **Python (Django):** Built-in Django authentication
- **Python (FastAPI):** fastapi-users, firebase auth
- **Go:** goth, go-auth
- **Platform services:** Clerk, Supabase Auth, Firebase Auth, Auth0

### Evaluation Questions for Auth Services

1. Does it work with my framework? (Check integration docs)
2. Does it handle the auth patterns I need? (OAuth, email/password, JWT, sessions)
3. What does it cost? (Free tier? Pay per user?)
4. How much setup does it require? (Run the quickstart)
5. Does it require me to build a separate server component? (Some services are frontend-only)

### When to Actually Build Auth Yourself

Only if:
- You're learning about auth as a learning exercise (not a real project)
- You have very specific requirements that no service supports
- You're building a CLI tool or internal script where auth isn't needed

---

## 4.5 — Third-Party Integrations (APIs, Webhooks, Email, Payments)

### How to Find and Evaluate a Service

When you need to add a feature that someone else has already built a service for:

1. **Identify the category:** Email → Mailgun/SendGrid/Stripe → Payments → Stripe/Paddle
2. **Search for "[category] + for [your use case]"** — find what other solo devs use
3. **Apply the 5-point library checklist** (from Phase 2)
4. **Look for official SDKs** — most major APIs have official client libraries for popular languages
5. **Run the quickstart example** before committing to integrate

### Common Integration Categories

| Need | Research Path | Common Tools |
|---|---|---|
| Send emails | "[language] + email sending tutorial" | Resend, SendGrid, Postmark, Mailgun |
| Receive payments | "[your stack] + payment integration tutorial" | Stripe, Paddle, LemonSqueezy |
| Add a chat widget | "chat widget for [framework]" | Crisp, Intercom, ChatGPT widget |
| Analytics | "[framework] + analytics setup" | Plausible, Google Analytics, PostHog |
| File storage | "[language] + file upload tutorial" | AWS S3, Cloudflare R2, Supabase Storage |
| Background jobs | "[language] + background tasks tutorial" | Celery (Python), Bull (Node.js), Sidekiq (Ruby) |

### Integration Pattern

1. Install the SDK/library
2. Run the official quickstart
3. Adapt it to your project's data flow
4. Move secrets to environment variables
5. Test with real data (sandboxes/test modes)
6. Commit the integration

---

## 4.6 — The "Glue" Pattern: Using Services Instead of Building

Most features a solo developer needs are already available as services. The research skill is learning to discover them.

### Before You Build Anything, Ask:

- "Is there a SaaS service or API for this?"
- "Is there an open-source tool I can self-host?"
- "Can I use a managed/paas platform that includes this feature?"

Search for it. The answer is often "yes."

### The Build vs Use Decision Flow

```
Do I need this feature?
  ├─ No → Don't build it
  └─ Yes:
       ├─ Is there a service that does this well? (search for it)
       │   ├─ Yes → Use the service (API, SDK, or embedded tool)
       │   └─ No:
       └─ Do I have a strong reason to build it myself?
           ├─ Yes → Build it
           └─ No → Use the service anyway (it might be good enough)
```

---

## 4.7 — Debugging When You're Stuck

You will get stuck. The research skill here is finding the solution efficiently.

### The Debugging Research Sequence

1. **Read the error message carefully.** What exactly went wrong? Copy the error text.
2. **Search the exact error message.** Copy-paste it into your search engine.
3. **Read the top 3 results.** Do they solve your exact problem?
4. **If not, add context.** Add "in [your language/framework]" to search and try again.
5. **If still stuck, search for the concept, not the error.** "How does [concept] work in [framework]?"
6. **Check the library's official docs.** Search for "[library name] [your feature] docs".
7. **Ask a human.** Relevant Discord server, forum, or Stack Overflow.

### How to Ask for Help Effect

When asking a community or forum:
1. Describe what you're trying to do
2. Show the exact error or unexpected behavior
3. Show the relevant code (minimal example that reproduces the issue)
4. State what you've already tried
5. State what you expected to happen and what actually happened

---

## Research Task for Phase 4

**For your project:**

1. Identify the first component you need to build (pick one — frontend, backend, database, auth, or integration)
2. Research the standard approach for that component in your stack — search `"[component] + [your stack] + tutorial"`
3. Follow the official or top-rated tutorial and get a minimal working example
4. Apply it to your project with your own data/configuration
5. Commit the working component with a meaningful message

One component at a time. Research, build, commit, repeat.