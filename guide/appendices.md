# Appendices

---

## Appendix A: When to Build vs. When to Use

Use this reasoning method, not a fixed table, every time you face this question.

### The Decision Flow

1. **Is there a service or library that does this exact thing?** — Search for it. If yes, the default is to USE it.
2. **Does that service/library meet my needs?** — Try its quickstart. If yes → USE it.
3. **Does it not quite fit?** → Can I configure it to fit? If yes → USE it and configure it. If no → BUILD it.
4. **Am I building this to learn?** — If so, building is valid regardless of options. But be honest about your motivation.
5. **Is the cost of building greater than the cost of using a service?** — Consider time, maintenance, bugs, and opportunity cost. The cost of building a custom solution is ongoing forever.

### Solo Dev Heuristics

- If someone already solved the problem → use their solution, don't reinvent it
- If the build time is more than 20% of the total project estimate → use a service
- If maintaining a custom solution would require ongoing expertise you don't have → use a service
- If you're genuinely learning → building is justified, but scope it small

---

## Appendix B: Glossary of Common Terms

This is a working glossary. When you encounter an unfamiliar term, look it up here and research it further.

| Term | Plain Language Meaning |
|---|---|
| API | A way for two programs to talk to each other using defined rules |
| CI/CD | Automated process that runs tests and deploys your code when you push changes |
| CLI | Command Line Interface — typing commands instead of clicking buttons |
| DNS | A system that translates human-readable domain names (like yourapp.com) into server addresses |
| Deployment | Making your code available on a server so others can access it |
| Dependency | A library your project uses (someone else's code you installed) |
| Docker | A tool that packages your app and its environment together so it runs the same everywhere |
| Environment Variable | A configuration value set outside your code (API keys, passwords, URLs) |
| Framework | A pre-built structure that provides tools and patterns for building applications |
| Git | A version control system that tracks changes to your code |
| HTTPS | Secure version of HTTP — encrypted communication between your users and your server |
| Library | Reusable code someone else wrote that solves a specific problem |
| Linux | A free, open-source operating system commonly used for servers |
| Load Balancing | Distributing incoming traffic across multiple servers |
| Logging | Recording events and messages from your application for debugging and monitoring |
| Monitoring | Watching your application to know when it's down, slow, or broken |
| Package Manager | A tool that installs and manages libraries your project depends on |
| Pipeline | An automated sequence of steps that processes your code (build → test → deploy) |
| Production | The live version of your app that real users access |
| Rendering | Generating content (HTML pages, images) from your code |
| Repository (Repo) | Where your code is stored and version-controlled (GitHub, GitLab) |
| Rollback | Reverting your deployed code to a previous version |
| Routing | How your application handles different URLs and requests |
| SSL/TLS | Technology that encrypts communication between your server and users (HTTPS) |
| Stack | The collection of technologies used to build a project |
| Staging | A test environment that closely mirrors production |
| Static Site | A website where pages are pre-built, not generated on each request |
| Testing | Running your code through checks to verify it works correctly |
| Terminal/CLI | The text-based interface you use to interact with your computer |
| Virtual Environment | An isolated space where your project's dependencies live, separate from your system |
| VPS | Virtual Private Server — a virtual machine you control remotely |
| Webhook | Automated HTTP calls your app makes when specific events occur |
| YAML/JSON | Data formats used for configuration files |

### How to Use This Glossary

When you encounter a term you don't understand in this guide or in documentation:
1. Look it up here first
2. If it's still unclear, search `"what is [term]" + for [your language]` — you'll find beginner-friendly explanations

---

## Appendix C: Cost Comparison of Popular Hosting Services

### Static Site Hosting

| Provider | Free Tier | Paid From | Best For |
|---|---|---|---|
| Cloudflare Pages | Yes, unlimited bandwidth | Free | Static sites, JAMstack |
| Netlify | Yes, 100GB bandwidth/month | $19/month | Static sites, SPAs |
| Vercel | Yes, limited bandwidth | $20/month | Next.js, frontend-focused apps |
| GitHub Pages | Yes | Free (public repos) | Personal sites, docs |

### Full-Stack Hosting (PaaS)

| Provider | Free Tier | Paid From | Best For |
|---|---|---|---|
| Railway | $5 credit, no permanent free tier | $5/month | Full-stack apps, databases |
| Render | Yes, limited instances | Free → $7/month | Web apps, APIs |
| Fly.io | 3 shared-CPU VMs free | Usage-based | Containerized apps |
| Render | Free for static sites | $7/month for services | Full-stack apps |

### VPS Hosting

| Provider | Cheapest Plan | Notes |
|---|---|---|
| DigitalOcean | $4/month | 1GB RAM, most popular for solo devs |
| Hetzner | $3.29/month | European datacenter, very affordable |
| AWS Lightsail | $3.50/month | AWS ecosystem access |
| Oracle Cloud | Free tier (always free) | 2 VMs with 1GB RAM each |

### Database

| Service | Free Tier | Notes |
|---|---|---|
| Supabase | 500MB PostgreSQL, free tier | Best for app databases |
| Neon | 0.5GB storage, free tier | Serverless PostgreSQL |
| MongoDB Atlas | 512MB free tier | NoSQL option |
| PlanetScale | Free tier (shared) | MySQL-compatible |
| Turso | 1GB free | libSQL, SQLite-compatible |

**Research method for costs:** Hosting prices change. Search `"[service] + pricing 2025"` before deciding. The table above is a starting point, not guaranteed current rates.

---

## Appendix D: Curated Starting Points per Phase

When you're stuck and don't know where to start your research, use these pointers:

### Phase: Idea Validation
- Search: `"how to validate startup idea without building"`
- Search: `"MVP examples solo developer"`

### Phase: Tech Stack
- Search: `"2025 stack for [project type] solo developer"`
- Search: `"roadmap.sh"` for visual learning paths (various domains)

### Phase: Domain Navigation (Unfamiliar Terrain)
- Search: `"[domain] roadmap 2025"` (e.g., "machine learning roadmap 2025")
- Search: `"awesome [domain]" on GitHub` — curated, maintained lists
- Search: `"[domain] for beginners practical tutorial"`

### Phase: Build
- Search: `"[framework] official docs getting started guide"` — ALWAYS start with official docs first
- Search: `"[problem] solved with [your stack]"` — find real code examples

### Phase: Deployment
- Search: `"[your framework] + deployment tutorial 2025"`
- Search: `"[your hosting provider] + step by step guide"`

### Phase: Monitoring
- Search: `"[your framework] + logging and monitoring guide"`
- Search: `"free monitoring for solo projects 2025"`

---

## Appendix E: Pre-Launch Checklist (Research-Based)

This checklist isn't a fixed answer — it's a set of research questions to investigate before deploying publicly.

Before you press "deploy":

- [ ] **Can users access the app?** — Deploy and verify from a device you don't use for development
- [ ] **Does it load over HTTPS?** — Test in incognito/private mode
- [ ] **Are errors reported?** — Connect an error tracker (Section 7.3)
- [ ] **Are secrets not exposed?** — Check that `.env` files are in `.gitignore` and secrets aren't hardcoded
- [ ] **Is the database backed up?** — Verify automated backups exist
- [ ] **Is there a way to revert?** — Test the rollback process (Section 6.7)
- [ ] **Is the README complete enough for someone else to understand and contribute?** — Write a short README if you don't have one
- [ ] **Does the app work at production URL?** — Test every user-facing feature manually
- [ ] **Are environment variables configured correctly for production?** — Check all secrets and URLs are set in production environment
- [ ] **Uptime monitoring is active?** — Set up at least one monitoring check (Section 7.4)

### After Launch

- [ ] **Set a reminder** to check logs weekly (Section 7.2)
- [ ] **Set a reminder** to run dependency audits monthly (Section 7.6)
- [ ] **Set a reminder** to review uptime and error tracking weekly

---

## Appendix F: Common Mistakes Solo Developers Make

### Mistake 1: Following Tutorials Without Building Your Own Project
**Symptom:** You can follow a tutorial step-by-step but can't build something original.
**Fix:** After every tutorial, build something tiny that applies what you learned but in a new context.

### Mistake 2: Spending Weeks Choosing a Stack and Never Starting
**Symptom:** You've researched 10 frameworks and still haven't written any code.
**Fix:** Set a 2-hour research limit per decision, pick the first viable option, and start building.

### Mistake 3: Building Features Nobody Asked For
**Symptom:** You're adding features based on your vision, not user feedback.
**Fix:** Before adding any feature, ask "who will use this?" and "what evidence do I have that they need it?"

### Mistake 4: Ignoring Deployment Until the End
**Symptom:** The project is "almost done" but never reaches deployment because deployment turns out to be harder than expected.
**Fix:** Deploy the simplest version of your project as early as possible. Even to localhost, even to a free tier. Get it online. Iterate from there.

### Mistake 5: Learning Theory Before Practice
**Symptom:** You spend weeks watching ML or system design courses but never apply them.
**Fix:** Use the "Project-First" method (see Phase 2b). Build a minimal project first. Learn theory only when you hit a wall.

### Mistake 6: Not Tracking Dependencies
**Symptom:** Months later, a dependency has a security vulnerability or breaks compatibility and you didn't notice.
**Fix:** Run dependency audits monthly. Enable Dependabot/Renovate on your repository.

### Mistake 7: Comparing Your Project to Others' Finished Products
**Symptom:** You feel discouraged because other projects look polished while yours is rough.
**Fix:** Remember that other projects have been worked on for months or years. Compare your project to your last version, not to someone else's final version.

### Mistake 8: Not Shipping Because It's Not Perfect
**Symptom:** You keep refactoring, redesigning, or optimizing features that users haven't even seen yet.
**Fix:** Apply the MVP rule. The smallest thing that delivers value is "good enough." Ship it, get feedback, then improve.

### Mistake 9: Working in Isolation on Research
**Symptom:** You spend hours alone trying to solve a research problem that someone else solved already.
**Fix:** Ask the community early. Discord servers, forums, Stack Overflow, and even Twitter/X are useful for quick research questions.

### Mistake 10: Skipping Documentation Because "It's Just a Personal Project"
**Symptom:** You can't remember what your code does 3 months later, or you can't redeploy it easily.
**Fix:** Write a README with setup instructions at the start of every project. Update it as you go. It protects future-you.