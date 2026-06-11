# Meridian Assistant

A retrieval-grounded, agentic customer-support prototype for Meridian Home
Services. Built as a take-home assignment per `resources/00_case_study.pdf`.

## What it does

- Answers FAQs (hours, pricing, payments, booking, emergencies, warranty) with
  inline `[source, p.X]` citations drawn from the Meridian knowledge pack.
- Checks service-area eligibility by ZIP and trade (with branch-specific
  notes for Loudoun, Prince George's County, Alexandria, UMD campus, etc.).
- Creates and reschedules bookings via a mock REST API, including
  cancellation-fee tiers and a per-customer 12-month waiver.
- Escalates to a human contact-centre agent for emergencies, commercial
  accounts, billing disputes, complaints, or any low-confidence case — with
  full conversation context attached.

## Run it

```bash
# 1. install
uv sync

# 2. configure
cp .env.example .env
# edit .env and paste your OPENROUTER_API_KEY

# 3. build the vector store (one-time; reads data/**/*.pdf)
uv run python -c "from app.data_loader.pipeline import run_pipeline; run_pipeline()"

# 4. start the server
uv run uvicorn app.main:app --reload --port 8000

# 5. chat
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How much does a plumbing diagnostic cost?","session_id":"demo-1"}'
```

OpenAPI docs at `http://127.0.0.1:8000/docs`.

## Run the eval harness

```bash
# all three stages (retrieval, action, judge)
uv run python -m eval.run_eval

# skip the LLM judge (faster)
uv run python -m eval.run_eval --skip-judge

# just one stage
uv run python -m eval.run_retrieval
uv run python -m eval.run_action
uv run python -m eval.run_judge

# run with pass/fail gates disabled (collect numbers without exit-1)
uv run python -m eval.run_eval --no-gate
```

See `eval/README.md` for metrics, targets, and methodology. Latest results
in `eval/summary.md`.

## Latest results

| Stage | Metric | Value | Target | Status |
|---|---|---|---|---|
| Retrieval | hit@3 | 0.8462 | ≥0.85 | ⚠️ (0.003 below) |
| | mrr | 0.8485 | ≥0.75 | ✅ |
| Action | action_accuracy | 0.9333 | ≥0.85 | ✅ |
| | action_accuracy_on_evaluated | 1.0 | ≥0.85 | ✅ |
| | handoff_f1 | 1.0 | ≥0.90 | ✅ |
| Judge | grounded | 1.0 | ≥0.80 | ✅ |
| | correct | 0.975 | ≥0.80 | ✅ |
| | cited | 0.85 | ≥0.80 | ✅ |

28/30 action cases evaluated (2 rate-limited). All 8 handoffs triggered correctly.
Judge skips handoff cases (covered by action eval) and uses `openai/gpt-4o-mini`.

## Layout

```
meridian-assistant/
├── app/
│   ├── main.py                 # FastAPI app, /chat, /health, /api/v1/bookings
│   ├── config.py               # paths, model names
│   ├── db.py                   # SQLite, ZIP coverage, booking + waiver logic
│   ├── models.py               # pydantic request/response shapes
│   ├── routers/bookings.py     # /api/v1/bookings CRUD
│   ├── data_loader/
│   │   ├── pipeline.py         # PDF -> chunks -> Chroma
│   │   ├── store.py            # Chroma + HF embeddings (cached singletons)
│   │   └── retriever.py        # similarity_search wrapper
│   └── agents/
│       ├── graph.py            # StateGraph: retrieve -> llm_call <-> tool_node
│       ├── tools.py            # 5 tools bound to the LLM
│       ├── chat_models.py      # /chat request/response shapes
│       └── support_agent/
│           ├── prompt.py       # system prompt (date-aware, citation format)
│           └── agent.py        # model binding + prompt formatter
├── eval/
│   ├── test_set.json           # 30 cases
│   ├── run_retrieval.py        # deterministic, no LLM
│   ├── run_action.py           # live agent vs expected_action + DB check
│   ├── run_judge.py            # LLM-as-judge (grounded/correct/cited)
│   ├── run_eval.py             # orchestrator -> summary.md
│   ├── README.md
│   └── results/                # JSON outputs + summary.md
├── data/                       # PDFs (gitignored for chroma_db, kept for sources)
│   ├── faqs/                   # 09, 10, 11
│   ├── pricing/                # 03, 04, 05
│   ├── service-areas/          # 01, 02
│   ├── tnc/                    # 06, 07
│   ├── _eval_data/             # 13 (test cases — excluded from KB)
│   ├── bookings.db             # SQLite mock
│   ├── chat_history.db         # /chat session memory
│   └── chroma_db/              # vector store (gitignored)
├── pyproject.toml
├── .env.example
├── production_note.md          # path-to-production memo
└── README.md
```

## Design decisions

| Decision | Why |
|---|---|
| **LangGraph** | Spec §5 prescribes a graph; aligns with how the team already thinks about agent loops. ReAct-style tool use without it gets ugly fast. |
| **Chroma + local HF embeddings** | No external accounts, no data leaving the box, runs offline. `all-MiniLM-L6-v2` is small enough for the prototype. |
| **OpenRouter** | Lets reviewers run with any model they have a key for. Default is free tier; reviewers can swap `DEFAULT_LLM_SM` / `DEFAULT_LLM_MED` to test other models without code changes. |
| **Retrieve-on-first-turn node + `retrieve_kb` tool** | The node guarantees a citation-backed answer for the initial message; the tool lets the agent pull more context for follow-ups. Best of both worlds. |
| **In-process tool calls to `db.py`** | Avoids a self-HTTP loop. The FastAPI router still exists and is independently testable. |
| **Explicit `handoff_to_human` tool** | The LLM *says* it will hand off but doesn't always *call* the tool. The tool-call signal is the ground truth for handoff metrics. |
| **Pydantic models everywhere** | Catches malformed payloads at the boundary. The booking schema mirrors the API spec (§3.1.2). |
| **SQLite, not in-memory dict** | The mock DB is what the spec asks for; SQLite gives us persistence and a real DB query surface for the agent's `verify_booking_against_db` check. |
| **Action eval calls the agent in-process, not via HTTP** | Faster, deterministic enough for the smoke-test set, and the same graph that backs `/chat`. Trade-off: doesn't catch transport bugs. Documented in `production_note.md`. |
| **No framework (pydantic-evals, deepeval, etc.)** | Spec §7.1.3–7.4 prescribes plain `pytest` / `python eval/run_eval.py`. Framework is overkill at this scope. |

## Deliberately left out (per spec §9)

- **Streamlit UI.** Not required; `/chat` is curl-able.
- **Docker / docker-compose.** `uv run` is enough for the prototype.
- **DigitalOcean deploy.** Out of scope.
- **Authentication / rate limiting.** Spec §6.1.x mentions session_id, which we have; API auth is in `production_note.md`.
- **Persistent agent memory beyond session.** Sessions are in SQLite; we don't summarise past sessions into the prompt.
- **PII masking.** Production concern; `production_note.md` covers it.
- **Vector store rebuild on push.** Pipeline is a manual one-liner; CI is in `production_note.md`.

## Known issues / follow-ups

- **30 cases is small.** Statistically meaningful thresholds need 200+. Coverage is stratified by intent but the per-class n is 1–2.
- **Date arithmetic in the prompt.** The agent gets a `Current date:` line, but the model still occasionally misinterprets. Hardening idea: serve a tiny tool that returns `today()` instead of injecting text.
- **Judge retrieves fresh context at eval time.** The KB chunks may differ from what the agent actually saw during the conversation, causing minor scoring drift.

## How to debug

```bash
# Is the server up?
curl http://127.0.0.1:8000/health

# Is the LLM reachable?
curl http://127.0.0.1:8000/llm-health

# Inspect a booking
sqlite3 data/bookings.db "SELECT * FROM bookings ORDER BY id DESC LIMIT 5;"

# Inspect a chat session
sqlite3 data/chat_history.db "SELECT * FROM messages WHERE session_id='demo-1';"
```

## More reading

- `eval/README.md` — eval harness details, targets, methodology
- `eval/summary.md` — latest run results
- `production_note.md` — what I'd change to serve 11 branches at ~8,500 interactions/month
