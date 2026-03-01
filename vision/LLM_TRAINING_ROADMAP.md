# Bell — LLM Training Roadmap

*Last updated: March 1, 2026*

> **The most important thing in this entire project.**
>
> Bell is only as good as its code generator. If the LLM does not generate
> correct QRL code reliably, everything else — the UI, the projects, the file
> upload, the Docker deployment — is worthless scaffolding around a broken engine.
>
> The goal: a fine-tuned model that generates correct QRL for any quantum
> question in Bell's supported domains, reliably, without needing Claude API credits.

---

## Why This Matters More Than Anything Else

Current state: Bell uses Claude Haiku (Anthropic API) with a system prompt containing
~20 few-shot examples. This works, but:

- It costs money per query (not viable at scale or for self-hosted customers)
- It depends on Anthropic's infrastructure (not self-contained)
- The few-shot examples are narrow — outside those ~20 types, the LLM guesses
- We have not systematically verified accuracy across all supported question types
- A single bad answer to a real user's question destroys trust

The only path to a reliable, self-contained, cost-effective Bell is a fine-tuned
model trained on high-quality QRL question→code pairs.

---

## The Training Data Problem

Fine-tuning requires data. Specifically: pairs of the form:

```json
{
  "question": "What is the entanglement fidelity of a 50km fiber link?",
  "code": "net = QuantumNetwork(\"test\")\nnet.add_node(\"A\").add_node(\"B\")\nnet.add_link(\"A\", \"B\", fiber_channel(50))\nresult = net.entanglement_fidelity(\"A\", \"B\")",
  "result": 0.431,
  "domain": "networks"
}
```

We need **500–1,000 verified pairs minimum** for a useful fine-tune. Right now we have ~20.

**The plan: collect them passively while dogfooding.**

Every time Bell generates correct QRL code and executes successfully, that pair is
a candidate training example. We log it automatically. A human reviews and approves.
Over weeks of dogfooding, the dataset builds itself.

---

## Phase 1 — Collect (Now → ~500 pairs)

**What we're doing:**
- Dogfood Bell seriously across all supported domains
- Every successful `question → code → result` triple is logged to `training/pairs.jsonl`
- Mark each pair as `approved` or `rejected` based on whether the answer was correct
- Expand the question set deliberately: rephrase the same question 5 different ways,
  try edge cases, try follow-ups

**Domains to cover systematically:**

| Domain | Question types | Target pairs |
|--------|---------------|--------------|
| Quantum networks | Fidelity, bottleneck, security, multi-hop | 150 |
| Bell/CHSH | Inequality test, S value, violation | 80 |
| Noise channels | Depolarizing, dephasing, amplitude damping | 80 |
| Causal inference | DAG, interventions, d-separation, do-calculus | 100 |
| Quantum Markov | Chain test, QCMI, Petz recovery | 80 |
| QuantumSwitch | Causal order, process matrix | 50 |
| Hardware Bell | qpu:belenos, sim:belenos | 30 |
| **Total** | | **~570** |

**What makes a good training pair:**
- Question is natural language a real user would ask
- Code executes without error (ok=True)
- Result is physically correct (verified by us, not just "ran without error")
- Code uses QRL idioms, not numpy/Python boilerplate

**What to reject:**
- Code that works but uses wrong approach
- Results that are numerically correct but physically meaningless
- Questions that are too similar to existing pairs (no value in duplicates)

---

## Phase 2 — Verify (~500 pairs collected)

Before fine-tuning, verify the dataset:

- Run all approved pairs through the current Bell executor
- Check results are consistent (same question → same result on re-run)
- Remove near-duplicates (cosine similarity > 0.9 between questions)
- Ensure domain balance (not 400 network questions and 10 causal ones)
- Target: **500 clean, verified, balanced pairs**

---

## Phase 3 — Fine-tune (~500 pairs verified)

**Model choice:** Llama 3.1 8B or Qwen 2.5 Coder 7B
- Small enough to run on a single GPU (8GB VRAM)
- Large enough to follow complex system prompts
- Both have strong code generation capability

**Method:** LoRA (Low-Rank Adaptation)
- Fine-tune only a small adapter layer, not the full model
- Training cost: ~$10–30 on Together.ai or Replicate (no local GPU needed)
- Result: a LoRA adapter that, combined with the base model, generates QRL

**Training format:** Instruction fine-tuning
```
System: [QRL system prompt — same as current Claude system prompt]
User: [question]
Assistant: [correct QRL code]
```

**Infrastructure:**
- Together.ai Fine-tuning API (most practical — no GPU setup needed)
- Alternatively: RunPod A100 instance (~$1.50/hour, 5-10 hours = ~$10)

---

## Phase 4 — Evaluate

Fine-tuning is only useful if the result is better than the baseline.

**Evaluation set:** 50 held-out pairs not used in training (set aside during Phase 1)

**Metrics:**
- **Execution rate:** % of generated code that runs without error (target: >90%)
- **Correctness rate:** % of results that are physically correct (target: >85%)
- **Comparison:** fine-tuned model vs Claude Haiku on same questions

**Acceptance criteria:** Fine-tuned model must match or beat Claude Haiku on
execution rate. If it doesn't, collect more data and retrain.

---

## Phase 5 — Deploy

Once the fine-tuned model passes evaluation:

1. **Export to GGUF format** — compatible with Ollama
2. **Add to Ollama** — `ollama create bell-qrl -f Modelfile`
3. **Update Bell config** — `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=bell-qrl`
4. **Test end-to-end** — full Bell stack with local model, no API calls
5. **Update Docker** — add Ollama service to docker-compose.yml (already stubbed)

Result: Bell runs entirely locally. No API costs. Ships in a container.
Self-hosted enterprise customers get the full model in their Docker stack.

---

## The Training Data File

All pairs are logged to:
```
quantum-relational-language/training/pairs.jsonl
```

Format (one JSON object per line):
```json
{
  "id": "uuid",
  "question": "...",
  "code": "...",
  "result": ...,
  "domain": "networks|bell|noise|causal|markov|switch|hardware",
  "approved": true,
  "notes": "optional reviewer note",
  "created_at": "2026-03-01T..."
}
```

The file is gitignored during collection (contains unpublished question→code pairs).
When we have enough data and decide to open source QRL, the training set can be
released alongside the model weights as a community contribution.

---

## Realistic Timeline

| Milestone | When | Condition |
|-----------|------|-----------|
| 100 pairs collected | ~2 weeks of dogfooding | Active testing |
| 500 pairs collected | ~6-8 weeks | Systematic coverage |
| Fine-tune run | After 500 pairs | ~$10-30, one weekend |
| Evaluation complete | +1 week | |
| Local model in Docker | +1 week | If evaluation passes |
| **Bell runs without Claude API** | **~2 months** | |

---

## What This Unlocks

Once Bell has its own fine-tuned model:

- **Self-hosted tier** — enterprise customers run Bell on their own infrastructure,
  no data leaves their building, no API costs
- **Offline mode** — Bell works without internet (except for QPU jobs)
- **Cost structure changes** — variable API cost becomes fixed infrastructure cost
- **Independence** — not dependent on Anthropic pricing or availability
- **Open source story** — release model weights alongside QRL language

This is the difference between a demo and a product.
