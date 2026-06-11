# Path to Production — Meridian Assistant

## Scope and assumptions
- Target: 11 branches, ~8,500 interactions/month, 3 regions.
- Inherited from prototype: LangGraph agent, Chroma + HF embeddings, OpenRouter, SQLite mock.
- Channels today: voice (6,000 calls) + email (2,500). Adding chat/DM is the trigger for this memo.
- Steady state ~12 RPS peak (8,500 × ~3 min each / 21 business days / 8 hr) — bursty around morning hours and post-storm spikes; LLM latency (3–8s) dominates, not CPU.

## Hardening

### Retrieval — RAG + Knowledge Graph
- Two `action_accuracy=0` rows in `eval/summary.md` (`edge_out_of_area_zip`, `edge_umd_campus`) are policy-graph failures; `hit@3` is 0.8462 (target 0.85). Vector retrieval alone won't close the gap.
- **RAG + Graph**: BM25 + vector + `NetworkX`/`Neo4j` keyed on `(zip → branch → trade → policy)`. Graph = source of truth for branch policy ("ZIP 20742 = UMD campus, electrical needs facilities-office co-ordination"). Vector store = free-text snippets (hours, pricing, FAQ wording).
- **Bigger embedding**: `bge-large-en-v1.5` or `text-embedding-3-large` via OpenRouter. `all-MiniLM-L6-v2` is fine on 13 docs, too small for branch-specific lookups.
- **Per-branch namespace** in Chroma + a `branch` filter as a hard constraint, so a PG County policy can't bleed into a Loudoun answer.

### Data
- **DB → Turso/LibSQL**. Hosted, branchable, embedded-replica reads, concurrent writes (Support is coming for concurrent writes). Also, since it uses persitant Chroma and Sqlite - live in memory (volume mounts) - so shutdown also erases app memory. Later, we can move to PostGREs - which is hosted.  
- **Storage → S3/GCS bucket**. Currently stored to in memory volume on Digital Ocean. 
- **Cache → Upstash Redis**, two purposes:
  1. Response cache keyed on `(session_id, last_user_msg_hash)` for FAQ hits (hours, pricing, service-area yes/no) — TTL 24h, invalidate on doc update.
  2. Rate-limit + idempotency-token store.
- **Queue → Redis Stream** for async channels. Worker picks up, returns `202` with job id, streams results over **SSE**.

### Scale
- **Digital Ocean App**, 1 vCPU / 1 GB shared, `min-instances=1` to avoid cold starts. 
- **Triggers** for moving off shared: p95 LLM-tail > 4s, error-rate drift, sustained concurrency > 10/instance. 
- **Multi-instance** with Turso (read replicas) + single primary (writes) — reads scale horizontally, writes funnel to primary. Current SQLite + Chroma in-process writes will fail after >1 instance.

### API hygiene
- **Auth**: bearer-token (`API_KEY` per branch) for booking API; per-session signed cookies for the chat widget. Branch key doubles as the retrieval namespace.
- **Rate limit**: `slowapi` backed by Redis. 60 req/min/session (chat), 5 req/min/IP (booking), 10 req/min/key (branch ingest). 429 with `Retry-After`.
- **PII removal**: Can consider the new openai model - https://openai.com/index/introducing-openai-privacy-filter/
- **Guardrails**
  - Input: profanity / injection classifier before retrieval; reject + ask to rephrase.
  - Output: JSON-schema-validate the LLM response; schema fail → `handoff_to_human`.
  - Tool: typed Pydantic schema + hard allow-list per tool. No free-form shell/SQL.
  - Safety models can be considered like - https://guardrails.openai.com/ (under preview)
 
## Observability
- **Logs**: structured JSON via existing `app/logger.py` with `session_id`, `branch`, `llm_calls`, `tool_calls`, `handoff_reason`. 
- **Traces**: **OpenTelemetry** (`opentelemetry-instrumentation-fastapi` + `langchain` instrumentation) → OTLP →  Signoz. 

- **Metrics** (Prometheus on `/metrics`):
  - Per-request: latency p50/p95/p99, LLM tokens in/out, tool-call count, sampled retrieval hit@3.
  - Per-channel: volume, handoff rate, action success rate, error rate.
  - Per-branch: traffic share, escalations, complaint keywords.
  - Per-model: cost (USD), error rate, p95 latency.
- **Alerts**: error-rate > 2% (5m), p95 latency > 8s, handoff-rate drift > 20% from baseline, prod retrieval hit@3 < 0.7, queue depth > 1k.

## Evaluation
- Move to **DeepEval**:
  - Faithfulness / groundedness regression on 200+ golden set, nightly.
  - LLM-as-judge with a stronger judge than agent model (current pattern — keep).
  - Per-branch suites: Loudoun plumbing = sub-contracted → handoff; PG County electrical = not licensed → handoff to EcoPower. 
  - Action harness regression as a first-class CI gate (`eval/run_eval.py` already exits 1 on threshold breach — wire it into deploy).
- **Online evals**: sample 1% of prod traffic, run groundedness/correctness/cited, write to same dashboard. Drift = open ticket.

## Testing
- Add pytest coverage:
  - `db.py` — waiver math, fee tiers, date math
  - `tools.py` — mock the LLM.
- Integration test: full graph with a **stubbed LLM** (deterministic fixture responses) for graph regressions without burning OpenRouter credits.

## Branch-specific policy
- ZIP→trade map is hard-coded in `db.py:23`. Move to a `branch_policies` table (or YAML in repo) keyed on `(branch_id, trade, rule_type)` — `["Loudoun", "plumbing", "handoff"]`, `["Alexandria", "electrical", "handoff_until_2026_q2"]`.
- Retrieval query rewritten to include the customer's branch when known. ZIP missing → "ask for ZIP" before answering policy questions.
- **Safety handoff rules** (extend `_HANDOFF_KEYWORDS` in `graph.py:57`):
  - Refund / waiver / fee dispute → always handoff, never the booking tool.
  - Commercial accounts, "speak to a manager", complaint, frustrated → always handoff.
  - Confidence floor: retrieval `hit@3 < 0.5` for the current context → handoff. Don't guess pricing or policy.
  - **Fuzzy match for emergency + safety keywords**: today's pre-filter is a case-insensitive substring (`graph.py:74-81`) — breaks on typos ("smellin gas"), paraphrases ("there's a gas smell in the basement"), and partial words. Move to a fuzzy finder (e.g. `rapidfuzz` / `thefuzz` with a threshold, or an embedding similarity against a curated emergency-intent set) so a missed spelling doesn't bypass the safety net. Keep the high-precision substring list as a fast first pass; fuzzy as the second pass before declaring "no handoff."

## Deployment & CI
- Migrations (SQLite → Turso) follow the `migrations/` pattern `aimi-ai` already uses.
- CI setup with DO integration. 
- **Rollback**: DO App Platform keeps prior versions; one-click revert. Keep app stateless, no breaking DB schema on deploy.
- **Kill switch**: env var for the active embedding model name + branch-policies source. Embedding swap or KB-rebuild going bad → flip the var, next request uses the prior config. Pairs with the `hit@3 < 0.7` alert.

## Debugging methodology
- **LLM-only issue**: replay the exact prompt with `temperature=0` for determinism.
- **Eval regression**: `python -m eval.run_eval --no-gate` to collect numbers, then `run_retrieval` / `run_action` / `run_judge` in isolation to find the broken stage.
- **Handoff loop / stuck conversation**: `chat_history.db` has every `messages` row for that session.
- **"Model changed its mind"**: check `retrieved_context` between good and bad traces first. Almost always a retrieval diff, not a model diff.


# Voice layer: 
- currenlty 6k+ requests are voice calls
- this will require expansion wrt TTS models + SST. 
- we can consider deploying on top of providers like OpenAI, Eleven Labs, Smallest.AI etc.
- For now, application only has text based environment
