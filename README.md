# DealerDesk — AI/ML track submission

Backend core + **Dealer Assistant** (tool-calling LLM) + **demand forecast and reorder suggestions**,
for the VIKMO take-home. Track: **AI / ML**.

**LLM provider:** Groq, model `llama-3.3-70b-versatile` (`GROQ_API_KEY`). The code also supports Gemini,
OpenAI and Anthropic via their standard env vars; any provider with a key is used as an automatic
fallback if the primary errors or is rate-limited. See `assistant/llm.py`.

**Video walkthrough:** _add your unlisted link here_

## Run it (the grader's exact path)

```bash
pip install -r requirements.txt
export DATABASE_URL=postgres://user:pass@localhost:5432/dealerdesk   # empty PostgreSQL database
export GROQ_API_KEY=...                                              # or another provider's key
python manage.py migrate --noinput
python manage.py seed_data        # idempotent; run it twice if you like
python manage.py runserver 0.0.0.0:8000 --noreload
```

Checks: `python smoke_test.py --base-url http://localhost:8000 --track aiml` ·
`python manage.py test` (42 tests, needs a PostgreSQL user that can create databases) ·
`python scripts/assistant_live_check.py` (real LLM, see below).

## What is where

| Area | Files |
|---|---|
| Products, inventory, dealers (A1) | `catalog/` |
| Draft orders: pricing, GST, numbering (A1) | `orders/` |
| Dealer Assistant (C) | `assistant/` — `agent.py` (loop), `tools.py`, `retrieval.py`, `grounding.py`, `safety.py`, `llm.py`, `prompts.py` |
| Forecast + reorder (D) | `ml/` (training, committed `forecast.csv`), `forecasting/` (API) |
| Error envelope, pagination, strict validation | `core/` |
| Design decisions | [`DECISIONS.md`](DECISIONS.md) |

## Dealer Assistant in one paragraph

`POST /api/assistant/chat/` runs an LLM ↔ tool loop (`assistant/agent.py`). The model can call
`search_parts`, `check_stock` and `create_draft_order`. Prices and stock come only from the database at
call time. Before a reply is returned, every SKU, order number, price and stock figure in it is checked
against what this turn's tools returned (`assistant/grounding.py`); a failing reply is sent back to the
model to fix, and after two failures replaced by a reply built directly from tool data. Product
descriptions are never shown to the model. Drafts only: the assistant cannot confirm, cancel or discount.

## Forecast in one paragraph

A Poisson GLM with a level per SKU, a global trend, a pooled lift per **week-relative-to-Diwali**, and a
promo multiplier. Backtested on 24 rolling folds: WAPE **0.208** vs 0.329 for the best baseline
(4-week average). Details, validation and weak spots in DECISIONS.md (answers 6–7).

## Honest status

- **Tested here:** everything that doesn't need a live LLM — 42 automated tests (money rules, a 12-thread
  order-number race, the reorder formula checked against an independent calculation for all 40 SKUs,
  retrieval, tools, grounding and safety checks, and the agent loop driven by a scripted LLM), plus the
  provided smoke test (10/10) on a fresh PostgreSQL database.
- **Tested against a real LLM:** `scripts/assistant_live_check.py` was run against Groq (`openai/gpt-oss-120b`),
  3 runs × 15 checks, all passing: grounded price/stock, live stock after an inventory change, clarifying
  questions, draft-only orders, refusal to oversell, injection resistance, off-topic refusal.
  It was run on a development database, once, on one provider. Other providers are supported but untested.
  Groq retired `llama-3.3-70b-versatile` on 2026-08-16, so the app detects an unavailable model and
  switches to one the key can use (`assistant/llm.py`); model names are env-overridable.
- **Not built (optional parts):** order confirm/cancel/dispatch and idempotency keys (A2), supplier sync
  (B), frontend (E). `Idempotency-Key` is ignored; order creation is not duplicate-safe.
