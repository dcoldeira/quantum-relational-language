# Bell Product Roadmap

*Last updated: March 1, 2026*
*Named after John Stewart Bell (1928–1990), physicist who proved quantum non-locality.*

> Bell is a quantum AI cloud platform built on QRL.
> Natural language → QRL → QPU → plain English answer.
> Target: scientific R&D companies (pharma, materials, quantum networks).

---

## Honest Current State

The platform works. It is not yet a product.

**What it does today:**
- Accepts natural language questions about quantum systems
- Generates correct QRL code via Claude (claude-haiku-4-5)
- Executes the code in a sandboxed QRL environment
- Returns a plain English explanation of the result
- Runs in Docker (`docker-compose up --build`)
- Handles ~15 question types across 3 domains

**What it cannot do today:**
- Answer questions outside its ~15 trained types reliably
- Handle follow-up questions (no conversation history)
- Authenticate users or bill for usage
- Run QPU jobs reliably end-to-end without manual setup
- Scale beyond a single process (in-memory job store)
- Be shown to a customer without explanation

**Honest verdict:** Impressive proof of concept. Not yet a product a company
would pay for. The gap between "works in Docker" and "company pays monthly" is
significant — but it is a gap that can be closed systematically.

---

## Is It a Real Product?

**Not yet, but the foundation is right.**

The core loop (question → QRL → result → explanation) is sound and working.
What is missing is everything around it: the breadth of questions it can answer,
the user experience, the authentication, the billing, and critically — the
domain coverage that makes it useful for a specific company's actual problems.

The platform is currently a **general quantum AI assistant** that can handle a
narrow set of question types. For it to become a product, it needs to be a
**specialist** in at least one domain that companies have real budgets for.

The most credible near-term domain: **quantum network design and analysis**
(already partially implemented in `src/qrl/domains/networks.py`).

---

## Who Would Buy This and Why

**Target: R&D teams at companies working on quantum-adjacent problems.**

These are not quantum computing companies — they are companies where quantum
effects matter: drug interactions, materials properties, network security,
financial risk correlation.

| Buyer | Their problem | What Bell gives them |
|-------|--------------|---------------------------|
| Pharma R&D | Molecular correlation structure for drug candidates | Causal analysis of quantum correlations without hiring a quantum physicist |
| Materials science | Entanglement properties of new materials | Fast screening of quantum properties |
| Quantum network operators | Network fidelity, bottleneck analysis, security | Instant answers to "where is my network weakest?" |
| Quantum computing startups | Benchmarking, circuit analysis | QRL as a second opinion on Qiskit results |

**The key insight:** these buyers do not want to learn QRL. They want answers.
The platform is the product, QRL is the engine. This is why the open source
language / proprietary platform split makes sense.

---

## The IDE Question

Three models for how companies interact with the platform:

**Model A: Hosted API**
Company sends `POST /ask` with a question, gets an answer back.
Integrates into their existing tools (Jupyter, Slack bot, internal dashboard).
- Lowest friction to adopt
- They bring their own UI
- Recurring revenue from API calls

**Model B: Hosted Web Platform**
Company logs in to `bell.entangledcode.dev`, asks questions in the browser.
More like a tool than an API.
- Higher friction to adopt but more self-contained
- Easier to demo and sell
- Works for non-technical buyers (managers, not just developers)

**Model C: IDE Plugin**
VS Code / Jupyter extension that adds QRL/Bell capabilities inline.
Developer types a comment like `# what is the entanglement fidelity here?`
and gets an answer inline.
- Highest adoption friction (requires install)
- Deepest integration into workflow
- Most compelling demo

**Recommendation: start with Model B (web platform), add Model A (API) as the
second tier for technical buyers.** Model C is a v2 feature — powerful but
premature. The web platform is already 80% built (`qai/static/index.html`).

---

## What "Usable for a Company" Actually Means

A company can use this when:

1. **It answers their specific questions reliably** — not generic demo questions
   but real questions about their actual system. This requires domain modules
   for their field (chemistry, materials, networks).

2. **It has authentication** — companies cannot share a single unauthenticated
   endpoint. Each team/user needs credentials, and usage needs to be auditable.

3. **It has an SLA** — companies need uptime guarantees. Single-process Docker
   on a laptop is not enough.

4. **It explains its reasoning** — "the answer is 0.81" is not enough.
   The explanation step (already implemented) needs to be good enough that
   a non-quantum expert can act on it. This is already better than expected
   with Claude — needs systematic testing.

5. **It handles follow-ups** — "what if we increase the noise to 20%?" should
   work without repeating context. Currently each question is stateless.

Items 1 and 5 are the hardest. Items 2, 3, 4 are engineering work.

---

## Technical Roadmap

### Phase 3 — Make It Deployable (next)
*Goal: something that could be shown to a pilot customer*

- [ ] **API key auth** — `X-API-Key` header check, keys in `.env`. Protects
      Claude credits. One afternoon of work.
- [ ] **Conversation history** — pass last N exchanges to Claude so follow-up
      questions work ("what if the noise is 20%?")
- [ ] **Job persistence** — SQLite store for jobs so they survive restarts
- [ ] **Hosted deployment** — deploy to a cloud VM (DigitalOcean, Hetzner EU)
      behind a domain. `bell.entangledcode.dev`
- [ ] **Better error UX** — "I don't know how to answer that" instead of
      raw Python tracebacks when questions fall outside coverage

### Phase 4 — Domain Depth
*Goal: answer real questions for a specific industry*

Pick one domain, go deep. Recommendation: **quantum networks** (already started,
clear commercial buyers, QRL coverage is strongest here).

- [ ] Expand `networks.py` to cover: multi-hop routing, decoherence budgets,
      key rate estimation, hardware-specific noise models
- [ ] Add 20+ few-shot examples for network questions
- [ ] Pilot with one quantum network operator or research lab

### Phase 5 — Platform Features
*Goal: something a company can actually pay for*

- [ ] Multi-tenancy (multiple organisations, isolated)
- [ ] Usage tracking and billing integration (Stripe)
- [ ] Audit logs (enterprise requirement)
- [ ] SLA and uptime monitoring
- [ ] API tier (programmatic access for technical buyers)

### Phase 6 — IDE Integration
*Goal: deepest workflow integration*

- [ ] VS Code extension with inline question/answer
- [ ] Jupyter kernel with `%%qrl` magic
- [ ] GitHub Copilot-style completions for QRL code

---

## Revenue Model

**SaaS subscription (primary):**
- Per-organisation monthly fee based on usage tier
- Tier 1 (startup/research): €200-500/month — limited queries, shared infra
- Tier 2 (SME): €1,000-3,000/month — higher limits, priority queue
- Tier 3 (enterprise): €10,000+/month — dedicated infra, SLA, audit logs

**Self-hosted licence (secondary):**
- Customer runs on their own GPU infra (pharmaceutical companies often require
  this for data sovereignty)
- Annual licence fee + support contract
- The Ollama option in `docker-compose.yml` already supports this model

**LLM cost pass-through:**
- Claude Haiku at ~$0.005-0.01/query is negligible at Tier 1-2 pricing
- At Tier 3 (enterprise), switch to dedicated Claude or self-hosted model
  to reduce per-query cost at scale

---

## Go-to-Market

**Before April 17 (QPL notification):**
- Nothing external-facing. Build Phase 3. Decide platform name.

**If QPL accepted:**
- Open source QRL at conference announcement (August 2026)
- Launch Bell as the commercial platform built on the open source language
- Use conference exposure to find first pilot customers from academic/industry attendees

**If QPL rejected:**
- Update Zenodo v2, publish on arXiv
- Quieter launch — reach out directly to quantum network operators and pharma
  quantum teams
- Same platform, less fanfare

**First customer strategy:**
- Do not sell to a Fortune 500 first. Find one quantum-adjacent research lab
  or startup willing to be a design partner.
- Give them free access in exchange for feedback on real questions.
- Use their questions to expand the few-shot examples and domain coverage.
- One real customer's questions are worth more than 100 demo scenarios.

---

## The Open Source / Platform Split

QRL (language) → open source, powers the platform
Bell → proprietary, built on QRL

Model: open core language, proprietary platform. Like Linux/Red Hat.

Nobody can replicate the platform just by forking QRL because the value is:
- The AI loop (Claude prompt engineering, few-shot examples)
- The domain modules and their coverage
- The hosted infrastructure and SLA
- The integrations (Quandela, PennyLane, future QPUs)

**Repo split (when open source decision is made):**
- Public: `src/qrl/`, tests, papers, docs
- Private: `qai/`, Dockerfile, docker-compose.yml, docs/api-reference.md
- Platform imports QRL as: `pip install quantum-relational-language`

**Open question:** does `src/qrl/domains/networks.py` go public or stay private?
Lean towards public — attracts quantum networking researchers, not a competitive risk.

---

## Realistic Assessment: Where Are We?

On a scale from idea to product:

```
Idea → Prototype → Proof of Concept → MVP → Beta → Product → Scale

                                    ↑
                              We are here
```

The proof of concept works end-to-end on real hardware. Getting to MVP requires
Phase 3 (auth, conversation history, hosted deployment) and meaningful domain
depth in at least one vertical. That is 2–3 months of focused work.

The QPL result in April shapes the launch narrative and the open source timing.
It does not change what needs to be built.
