# Phase 7: Monitoring and Maintenance

Deployment isn't the end — it's the beginning of keeping your project alive. Monitoring means knowing when things go wrong and understanding how your project is being used. Maintenance means keeping it working as your code, dependencies, and needs evolve.

This chapter teaches you how to research the monitoring and maintenance tools that make sense for YOUR project.

---

## 7.1 — "Do I Need Monitoring?"

If your project is a personal script or a weekend project, monitoring might be overkill. But once your project has users or real consequences, it matters.

### Research the Answer for YOUR Project

Ask yourself:
1. **What would happen if my app went down for an hour?**
   - Nothing matters → minimal or no monitoring
   - Users can't access something important → uptime monitoring
   - Financial loss or data loss → proactive error tracking and logging

2. **Do I have users/customers who depend on this?**
   - Yes → monitoring is part of the cost of having users
   - No (personal project) → monitoring is optional

3. **Am I actively debugging issues from users?**
   - Yes → error tracking will save you time
   - No → focus on uptime monitoring first

---

## 7.2 — Logging: Recording What Happens

### What Is Logging?

Logging is writing events (requests, errors, state changes) to a file or service so you can look back when something goes wrong.

### How to Research Logging for YOUR Stack

Search: `"[your language/framework] + logging best practices"` or `"[your framework] + where to log"`.

### The Basic Setup

Most languages have a built-in logging system. Research these two things:

1. **How to use the built-in logger** — search `"[language] + standard logging tutorial"`
2. **Where logs should go** — stdout (terminal) or a file, plus a log management service if needed

### The Solo Developer's Logging Approach

1. Start simple: use your language's built-in logging
2. Log at appropriate levels:
   - **INFO** — normal operation (requests received, jobs completed)
   - **WARNING** — something unexpected but not broken
   - **ERROR** — something failed that needs attention
3. Log useful context: what happened, when, and any relevant data (without logging secrets)
4. Review logs regularly — set a reminder to check them weekly

### When to Scale Logging Up

If you find yourself searching through terminal output or files to find a problem, that's when you research log management services.

Search for `"[your stack] + log management"` — you'll find options like BetterStack, Datadog, or Grafana Loki. Most have free tiers adequate for solo projects.

---

## 7.3 — Error Tracking

Error tracking automatically catches and reports errors so you know when something breaks in production.

### How to Research Error Tracking for YOUR Stack

Search for: `"[your stack] + error tracking tutorial"` or `"[your language] + crash reporting"`.

### Popular Options with Free Tiers (Solo Dev Friendly)

| Service | Best For | Free Tier |
|---|---|---|
| Sentry | Most frameworks, most languages | 5,000 events/month |
| BetterStack | Uptime + logging + errors | Free for small projects |
| LogRocket | Frontend error tracking | Limited free tier |
| Axiom | Log management + errors | Generous free tier |

### The Setup Process

1. Sign up for your error tracking service
2. Follow their getting-started guide for YOUR stack
3. Install their SDK/library in your project
4. Trigger an error to verify it's working
5. Commit the configuration

**Research method:** Each service has a "quickstart" — start there. Don't read the full manual. Get it working with one error type first.

---

## 7.4 — Uptime Monitoring

Uptime monitoring checks if your app is accessible and alerts you when it goes down.

### How to Research Uptime Monitoring

Search: `"free uptime monitoring for solo developer"` or `"[your domain/subdomain] + uptime monitoring"`.

### Popular Options

- **UptimeRobot** — free for up to 50 monitors, checks every 5 minutes
- **BetterStack** — free tier with additional features
- **Healthchecks.io** — open-source, cron job monitoring
- **Pingdom** — more enterprise, paid for small projects

### Setup

1. Create an account on your chosen monitoring service
2. Add a monitor for your deployment URL
3. Set the check interval (5 minutes is standard)
4. Configure alerting — most services can send an email when your site is down
5. Test by stopping your app temporarily and confirming you get an alert

---

## 7.5 — Analytics: Understanding Usage

### Do You Need Analytics?

Analytics tell you how users interact with your project. For solo devs:
- **If your project is public** — analytics help you understand what's useful
- **If you have a small user base** — analytics help you prioritize what to build next
- **If your project is private** — skip analytics

### How to Research Analytics for YOUR Project

Search: `"[your stack] + add analytics"` or `"lightweight analytics for solo projects"`.

### Privacy-Friendly Analytics Services

- **Plausible** — simple, no cookies, privacy-focused
- **Umami** — open-source, self-hosted, free
- **Google Analytics** — standard but heavy (cookie consent required in many regions)
- **PostHog** — open-source, product analytics, generous free tier

### The "Do I Need Analytics?" Decision

1. **Does my app have a public URL?**
   - No → skip analytics
   - Yes → research adding analytics
2. **What questions do I want answered?**
   - "How many people visit?" → simple page views
   - "What features do they use?" → event tracking
   - "Are they finding bugs?" → error tracking (Section 7.3)

Start with the simplest option. Add features only when you have a specific question you need to answer.

---

## 7.6 — Security Maintenance

### Dependency Vulnerability Scanning

Old or unmaintained dependencies can have security vulnerabilities.

**Research method:**
- Search `"[your package manager] + security audit"` — most package managers have built-in audit commands
  - npm → `npm audit`
  - pip → `pip audit` (or `safety check`)
  - cargo → `cargo audit`
  - Go → `govulncheck`
- **Dependabot** (GitHub) and **Renovate** — automated dependency update tools that also flag vulnerabilities

Setup research: `"[your code host] + automated dependency updates"` — follows the documentation of your code hosting platform (GitHub, GitLab, etc.).

### Basic Security Practices

1. **HTTPS everywhere** — your deployment provider should handle this (see Phase 6)
2. **No secrets in code** — everything sensitive in environment variables
3. **Rate limiting** — if your API is public, research how to rate limit in your framework
4. **Input validation** — never trust user input; validate it on the server side
5. **Regular dependency updates** — run `npm audit` / equivalent monthly (or automate with Dependabot)

---

## 7.7 — Backup Strategy

### How to Research Backups for YOUR Setup

Search: `"[your database] + automated backup guide"` and `"[your hosting] + backups"`.

### The Solo Developer's Backup Approach

1. **Does my hosting provider offer automated backups?** — Search for it. Most managed services do.
2. **If hosting my own database:** set up automated daily/weekly backups
3. **Test the backup** — research `"[database] + restore from backup"` and verify it works
4. **Store backups in a separate location** — doesn't need to be fancy, just not on the same server as your app

**Research checklist for backups:**
- [ ] Is there an automated backup from my hosting provider? (search for it)
- [ ] Do I have a manual backup strategy if there isn't? (set one up)
- [ ] Have I tested restoring from a backup? (search the restore process and try it once)

---

## 7.8 — Ongoing Maintenance as Research

Maintenance for a solo developer isn't a fixed task list — it's continuous research into whether your project still works as your tools and dependencies evolve.

### Monthly Maintenance Routine

Research and perform these checks monthly:

1. **Dependency updates** — run your package manager's audit command
2. **Log review** — check your logs for warnings or recurring errors
3. **Security check** — are any of your dependencies flagged as vulnerable?
4. **Error tracking review** — are new errors appearing?
5. **Uptime review** — has your app been down? When? How long?

### When to Research Upgrades

You need to upgrade when:
- A dependency has a security vulnerability you can't patch without upgrading
- Your hosting provider is deprecating features you use
- A dependency you rely on is no longer maintained
- New versions of your framework or language drop breaking changes

**Research method for upgrades:** `"how to upgrade [dependency/framework] from X to Y"` for each component. Check the upgrade/migration guide that most projects provide.

---

## Research Task for Phase 7

1. Search `"[your hosting provider] + monitoring and logging"` — find what's available through your provider
2. Sign up for Sentry (or equivalent error tracker) and connect it to your project
3. Set up UptimeRobot (or equivalent) to monitor your deployed URL
4. Run a dependency audit (`npm audit`, `pip audit`, etc.) and fix any critical issues
5. Set up Dependabot or Renovate for automatic dependency updates

These five tasks establish a basic monitoring and maintenance baseline for your project.