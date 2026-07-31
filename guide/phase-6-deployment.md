# Phase 6: Deployment

Deployment is the moment your project becomes accessible to users (or to yourself, from another machine). This chapter teaches you how to research and navigate deployment by understanding what YOUR project needs, not by following a one-size-fits-all template.

---

## 6.1 — Understanding Deployment

At its core, deployment means: "making my code run on a server that people can access."

Everything else is just variation on that theme. You don't need to understand all deployment methods upfront. You need to understand the one that fits YOUR project right now.

### How to Research Deployment for YOUR Project

**Exact search to use:**
1. `"[your language/framework] + how to deploy"` — this is the #1 search
2. `"[your framework] + deployment guide"` — framework-specific guides
3. `"[your framework] + cheapest deployment"` or `"[your framework] + free deployment"` — budget-aware

**Read the official documentation's deployment section.** Every major framework has one. Start there.

---

## 6.2 — Choosing Where to Deploy

### The Deployment Research Method

Instead of memorizing hosting options, use this research flow:

1. **Identify your project type:**
   - Static site (no backend server)
   - Full-stack web app (frontend + backend)
   - API only
   - Background worker/CLI app

2. **Search for `"[project type] + best hosting for solo developer 2025"`** or `"[project type] + cheapest deployment"`

3. **Review the top 5 results** — what services do they recommend?
4. **Apply the 5-point checklist** (from Phase 2) to the options you find
5. **Pick the simplest one that supports your stack** and move on

### Deployment Options Ranked by Complexity

| Hosting Type | Best For | Research Path |
|---|---|---|
| Static site hosting (Netlify, Vercel, Cloudflare Pages) | Frontend-only sites, static HTML/CSS/JS | Search "[framework] + deploy to Netlify/Vercel/Cloudflare" |
| Full-stack PaaS (Render, Railway, Fly.io) | Web apps with backend + database | Search "[language/framework] + deploy to Railway/Render" |
| VPS (DigitalOcean Droplet, AWS Lightsail) | Apps needing full control | Search "[OS] + how to deploy [framework]" |
| Platform-specific (Vercel for Next.js, Heroku for Ruby) | Framework-specific optimized hosting | Check if your framework has its own hosting product |
| Serverless (AWS Lambda, Cloudflare Workers) | Event-driven or API-only apps | Search "[need] + serverless deployment tutorial" |

### The Solo Developer's Deployment Rule

**Start with the simplest option.** You can always migrate to a more complex option later. The research goal is to deploy your project, not to find the perfect hosting infrastructure.

---

## 6.3 — Domains and DNS

### How to Research This

1. **Get a domain:** Search `"cheap domain registrar 2025"` — Namecheap, Cloudflare Registrar, Porkbun are common options
2. **Point it to your hosting:** Search `"how to connect domain to [your hosting provider]"` — your hosting provider has specific instructions
3. **Research DNS:** DNS (Domain Name System) simply maps your domain name (e.g., yourapp.com) to the server where your app lives. Your registrar and hosting provider both have guides for this.

**The 5-minute research method for DNS:**
1. Search `"DNS tutorial for beginners"` — read one getting-started article
2. Search `"[registrar name] + how to set DNS"` — follow their specific instructions
3. Search `"how to verify DNS is working"` — confirm it works

### SSL/TLS (HTTPS)

**Do not set this up manually.** Most modern hosting providers give you SSL for free and set it up automatically.

- Search `"[your hosting] + how to enable SSL"` — if your provider doesn't do it automatically, that's a red flag about the provider
- Cloudflare offers free SSL as a CDN layer between your users and your server
- After deployment, check that your site loads at `https://` (with the padlock icon)

---

## 6.4 — Environment Variables and Secrets

Your project needs configuration that should never be public: API keys, database passwords, secret tokens.

### How to Research This for YOUR Framework

Search: `"[your framework] + environment variables"` and `"[your framework] + secrets management"`.

### The Basic Pattern (Works Everywhere)

1. Create a `.env` file in your project root with your local secrets
2. Add `.env` to your `.gitignore` file (so secrets don't get committed)
3. In your code, read environment variables using the language's built-in method
4. In production, set environment variables through your hosting provider's dashboard or CLI

**Search pattern for production env vars:** `"[your hosting provider] + set environment variables"` — the hosting provider has a specific way to do this.

---

## 6.5 — Database Deployment

### How to Research Your Database Deployment

If you're using a database in production, you need it hosted somewhere. Search for:

- `"[database engine] + free hosting"` (if budget is a concern)
- `"[database engine] + managed service for [your stack]"`
- `"[your hosting provider] + database setup"`

### The Solo Developer's Default

1. Use a **managed database service** from your hosting provider or a dedicated database host (Supabase, Neon, PlanetScale, etc.)
2. Do NOT self-host databases on a VPS unless you're comfortable with database administration
3. Set up backups — search `"[database engine] + automated backup"` and follow the docs

---

## 6.6 — CI/CD: What It Is and When You Need It

### What CI/CD Does (Simple Version)

CI/CD automatically runs tests and deploys your code whenever you push changes to your repository. This is useful when:
- You want to deploy without manually SSHing into a server
- You want to run tests before every deployment
- Multiple deployments make manual deployment error-prone

### When Solo Devs Need CI/CD

- **You do** if you deploy frequently and want it to be automated
- **You don't** if you deploy manually and are comfortable with it
- **Start without it** if you haven't deployed yet — manual deployment teaches you more about the process

### How to Research CI/CD for Your Project

Search: `"[your framework] + GitHub Actions deployment"` (assuming you use GitHub for code hosting).

Most frameworks have an official guide for deploying via GitHub Actions. Follow that guide step-by-step.

### Minimal CI/CD for Solo Devs

The simplest useful CI/CD pipeline:
1. Push code to GitHub
2. GitHub Actions runs your tests
3. If tests pass, GitHub Actions deploys to your hosting provider
4. You review the deployment in your hosting dashboard

**Research method:** Find your framework's official deployment guide for CI/CD. They almost always have one.

---

## 6.7 — Rollback Strategy: What If It Breaks?

Before your first deployment, have an answer to this question: "How do I revert if something goes wrong?"

### How to Research Rollback for YOUR Setup

Search: `"[your hosting] + how to rollback deployment"` or `"[your framework] + deployment rollback"`.

### The Simple Rollback Approach

1. **Git history IS your rollback tool.** If something breaks, revert to a previous Git commit and redeploy
2. **Tag your releases:** Use Git tags (`git tag v1.0.0`, `git tag v1.1.0`) so you can reference specific versions
3. **Keep the previous deployment artifacts** — most PaaS providers keep the last version and let you switch back

**Before your first deployment, verify:** Can you revert to the previous version within 5 minutes? If not, research how to make it possible.

---

## Research Task for Phase 6

1. Search `"[your framework] + deployment guide"` — find your framework's official deployment docs
2. Pick the simplest deployment option mentioned in those docs
3. Follow the deployment guide step-by-step to get your project live
4. Set up environment variables for your project (both local and remote)
5. Verify that your deployed app works and loads over HTTPS
6. Test rollback: make a trivial change, deploy it, then revert and redeploy the previous version

Your project should be live at this point. That's the goal. The rest of the deployment process can be learned and improved later.