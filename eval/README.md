# Eval Harness

Three-stage evaluation for the Meridian Assistant. All stages are deterministic
shell scripts that write JSON results to `results/` and a human-readable
`summary.md` at the end.

## Layout

```
eval/
├── test_set.json          # 30 cases (20 from 13_customer_messages.pdf + 10 edge)
├── case_models.py         # Pydantic schemas
├── run_retrieval.py       # Stage 1: Hit@3, MRR, recall@k  (no LLM)
├── run_action.py          # Stage 2: full agent vs expected_action  (LLM)
├── run_judge.py           # Stage 3: LLM-as-judge for answer quality  (LLM)
├── run_eval.py            # Orchestrator
└── results/               # JSON + summary.md output
```

## Run

```bash
# All three stages (~5-10 min on OpenRouter free tier)
python -m eval.run_eval

# Faster: skip the judge
python -m eval.run_eval --skip-judge

# Just one stage
python -m eval.run_eval --only retrieval
python -m eval.run_eval --only action
python -m eval.run_eval --only judge
```

The action stage needs the live LLM reachable (set `OPENROUTER_API_KEY` in `.env`).

## Metrics

| Metric | Stage | Target | Why |
|---|---|---|---|
| `hit_at_3` | retrieval | ≥ 0.85 | Top-3 should contain the right source |
| `hit_at_5` | retrieval | ≥ 0.95 | |
| `mrr` | retrieval | ≥ 0.75 | First relevant doc should rank high |
| `action_accuracy` | action | ≥ 0.85 | Agent picks the right action class |
| `handoff_precision` | action | ≥ 0.90 | Don't over-handoff (UX cost) |
| `handoff_recall` | action | ≥ 0.90 | Don't miss emergencies |
| `handoff_f1` | action | ≥ 0.90 | Safety-critical balance |
| `grounded` | judge | ≥ 0.80 | No hallucination |
| `correct` | judge | ≥ 0.80 | Answers the question |
| `cited` | judge | ≥ 0.80 | Cites a source |

## Test set

`test_set.json` has 30 cases:

- **Cases 1–20** are transcribed verbatim from `resources/13_customer_messages.pdf`
  (the 20 representative customer messages Meridian supplied). Each has the
  intent label, expected resolution path, and handoff decision noted in the PDF.
- **Cases 21–30** are the 10 edge cases from spec §7.1.2 (out-of-area ZIP,
  Sunday booking, warranty claim, missing ZIP, commercial net-30, etc.)

Each case has: `user_message`, `expected_action`, `expected_sources`,
`expected_keywords`, `expected_handoff` (and reason).

## Action classes

- `answer` — agent answered directly, no handoff.
- `answer_or_confirm` — answer is fine, OR agent is in the confirmation
  pre-state (acceptable for booking cases).
- `answer_or_handoff` — answer is fine, OR a clean handoff is fine
  (e.g. status checks where the agent may not have backend access).
- `handoff` — agent MUST call `handoff_to_human`.
- `create_booking` — booking must appear in `data/bookings.db`.

## Methodology / guardrails

- **Retrieval is deterministic** (Chroma + sentence-transformers). No LLM
  in the loop. Safe to re-run.
- **Action eval calls the live agent.** Results depend on model and
  prompt state. Pin the model and freeze the prompt before trusting numbers.
- **Judge uses the same model as the agent** (no separate account). Known
  bias: the model grades itself leniently. A separate, larger judge model
  would help. Documented as a limitation in `summary.md`.
- **Test set is hand-authored.** No LLM-generated adversarial cases yet.
  Spec §7.1.2 mentions this as an extension.
- **Eval is offline.** No telemetry, no human feedback, no online metrics.
  Production would add thumps-up/down logging, handoff rate per branch,
  retrieval latency p50/p95.

## Re-running safely

The agent's action eval mutates `data/bookings.db` (any case classified as
`create_booking` will insert a row). If you need a clean baseline, snapshot
the DB first:

```bash
cp data/bookings.db data/bookings.db.bak
python -m eval.run_eval --only action
mv data/bookings.db.bak data/bookings.db
```
