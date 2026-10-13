# DECISIONS — AI / ML track

Line numbers refer to this repository at submission.

## 1. Schema, relationships and `on_delete`

| Model | Key fields | Foreign keys → `on_delete` and why |
|---|---|---|
| `Product` (`catalog/models.py:4`) | `sku` unique, `price` Decimal(12,2) with `CHECK price > 0` | none |
| `Inventory` (`catalog/models.py:30`) | `quantity` with `CHECK quantity >= 0` | `product` OneToOne → **CASCADE**: a stock row means nothing without its product. Separate table so stock writes lock one narrow row and never contend with catalogue edits. |
| `InventoryAdjustment` (`:42`) | old/new quantity, reason | `product` → **CASCADE**: audit rows are meaningless without the product. |
| `Dealer` (`:52`) | `dealer_code` unique, `credit_limit` | none |
| `Order` (`orders/models.py:14`) | `order_number` unique, `status`, stored totals | `dealer` → **PROTECT** (`:17`): a dealer with order history must not be deletable. |
| `OrderItem` (`orders/models.py:30`) | `quantity >= 1` check, `unique(order, product)` | `order` → **CASCADE** (`:32`): lines have no meaning without the order. `product` → **PROTECT** (`:34`): a product on an order can't vanish underneath it. |
| `DailyOrderCounter` (`:49`) | `date` PK, `last_seq` | none; the locked row that makes order numbers gapless. |
| `ChatSession` (`assistant/models.py:6`) | `unique(session_id, dealer)` | `dealer` → **CASCADE**: conversations belong to the dealer. |
| `ChatMessage` | `role`, `content` | `session` → **CASCADE**: messages die with their conversation. |

`ChatSession` is unique on `(session_id, dealer)` (`assistant/models.py:15`), so the same `session_id` used by
two dealers is two conversations: memory cannot leak between dealers. Money is `DecimalField`, never float.

## 2. How retrieval works, and where it fails

Dealers ask "*part* for *vehicle*". The vehicle is a **hard constraint**, so it is resolved against the
real fitment values (`retrieval.py:153` `resolve_vehicle`) and applied as a filter, not left to a similarity
score. Resolution is token-based (`NS200` → `ns 200`), typo-tolerant (`pulser` → `pulsar`), and returns one of
`resolved / ambiguous / unknown`. "Duke" is ambiguous (200 vs 390); "Honda Activa" is *unknown* (we stock Honda parts but
not that model), and the tool says so rather than guessing (`retrieval.py:263` `search` returns
`ambiguous_vehicle` / `unknown_vehicle` / `needs_vehicle` / `needs_part_type`).

The part type is ranked with two signals fused (`retrieval.py:211` `scores`): **BM25F** over name (×3), category,
brand, fitment and a sanitised description, and **cosine over character 3-gram TF-IDF vectors** (typos, "tire/tyre").
A term-coverage factor (`:303`) stops a *Tank Pad* ranking for "brake pads" just because it shares "pad".
Exact-fit parts get a bonus; Universal parts fill in behind them.

*Why this and not neural embeddings:* the catalogue is 600 short, templated names ("Brake Pad Set - Bajaj Pulsar 150").
Exact words and the vehicle filter carry nearly all the signal; the n-gram vectors add typo tolerance. Everything is
in-process, deterministic and needs no embeddings API at grading time. I did **not** use a dense neural model; that is
a deliberate trade, not an oversight. Prices/stock are never indexed: results are hydrated from the DB at query time
(`tools.py:98`), and the index rebuilds when the product table changes (`retrieval.py:244`).

*Where it fails:* (a) paraphrase with no shared words ("stopping power upgrade") finds nothing, which a neural embedding would
catch; (b) the synonym table is hand-written and small; (c) `MIN_SCORE = 0.30` (`:49`) was tuned by eye on ~20 queries, not
on a labelled set; (d) "oil" is ambiguous between engine oil and oil filters and asks for a vehicle it does not need;
(e) a vehicle not in the fitment list cannot be resolved, so a brand-new model added by supplier sync is invisible until the index rebuilds.

## 3. The tool-calling loop

`assistant/agent.py:39` `run_turn`: build `[system prompt, last 10 messages, user message]` (`:41`, history from
`load_history` `:28`), then loop up to 7 LLM calls (`:45`):
1. `router.chat(...)` (`:47`) tries providers in order with retry/fallback (`llm.py:172`).
2. If the model returned tool calls, each is executed by `run_tool` (`:57`, `tools.py:158`) and its JSON result is appended
   as a `tool` message; loop again.
3. If it returned text, `verify_reply` (`:59`, `grounding.py:31`) checks it. Clean → return it.
4. Not clean → the problems are fed back to the model and it retries (max 2, `:64`). If the problem is "figures without a
   tool call", the retry sets `tool_choice="required"` (`:70`), so the model *must* look the data up.
5. Still not clean, or every provider failed → `fallback_reply` (`:73`, `grounding.py:73`) builds the reply straight from tool
   output; if nothing was fetched and the LLM is down, a plain "try again" message. The API never returns a 5xx for an LLM outage.

`cited_skus` is derived from the *final reply text* intersected with real SKUs (`agent.py:33`), not from what the model says it cited.
The dealer id is injected from the request, never taken from model arguments (`tools.py`: `create_draft_order` ignores any `dealer_id`).

## 4. How prices and stock stay true

* Tools read the database on every call: `search_parts` (`tools.py:98`) and `check_stock` (`:108`) query `Product`/`Inventory` live.
  Nothing about price or stock is cached or stored in chat history (only reply *text* is, `agent.py:76`).
* `verify_reply` (`grounding.py:31`) rejects any ₹ amount or stock count not present in **this turn's** tool results
  (`:58`), any SKU that doesn't exist, and any real SKU that no tool returned this turn (`:39`), which blocks answering from
  a stale earlier turn. A figure with no tool call at all is rejected and forces `tool_choice="required"`.
* Allowed derived numbers: quantities the dealer typed, and price × such a quantity.
* Order numbers must come from `create_draft_order`; claims of "confirmed", "reserved" or a discount are rejected.
* Test: `assistant/tests.py` `test_stale_numbers_from_history_force_a_fresh_tool_call` changes stock between turns and shows the
  answer follows the DB.

*Limit:* the checker finds figures near ₹/Rs/"units/in stock/available". A number written some other way ("a hundred and twenty") or a
wrong *non-numeric* claim ("fits all bikes") is not caught.

## 5. Untrusted supplier descriptions, and where the defence stops

Layered, cheapest first:
1. **Descriptions never reach the model.** Tools return name, brand, category, fitment, price, stock only (`tools.py:87` `_item`);
   `test_descriptions_never_reach_the_llm` checks all 600 products. The usual carrier of hidden instructions is simply absent.
2. **Fields that do reach it are sanitised**: NFKC-normalised, control/zero-width/bidi characters stripped, and replaced with
   `[removed]` if they read like an instruction to an AI (`safety.py:46`, patterns `:14`). The retrieval index drops offending
   sentences from descriptions (`safety.py:54`), so a poisoned description cannot make a product match unrelated queries.
3. **The system prompt** says catalogue text is data (`prompts.py`, rule 5).
4. **Output checks don't trust the model's intent.** Even if an instruction got through, the reply is rejected if it contains a discount or
   promo code, a "confirmed / reserved" claim, an unknown SKU, or an order number we didn't create (`grounding.py:21`, `:31`).
5. **Capability limits.** The model can only create DRAFTs, for the request's dealer, and never above available stock (`tools.py:128`).
   Even a fully hijacked model cannot confirm an order, cancel one, or order for another dealer.

*Where it stops working:* the pattern list is regex, so a novel phrasing or another language slips past layer 2; layer 1 only protects
description text, so a malicious **name** or **brand** with a new phrasing would reach the model; layer 4 catches *harmful outputs we thought
of* (discounts, confirmations), not arbitrary misbehaviour such as leaking the system prompt. The hard guarantee comes from layer 5, not from detection.

## 6. Forecasting approach, validation, and the leakage guard

**Model:** a Poisson GLM (`ml/models.py:52`): a level per SKU, one global trend, one pooled multiplier per *week relative to Diwali*
(−4…+3, `ml/common.py:39`), a pooled promo multiplier plus a heavily shrunk per-SKU promo deviation. Poisson because the target is a
non-negative count whose noise grows with its mean. I checked the structure in the data first: the festive lift is ~1.35×, 1.5×, 1.6×, 1.7×,
2.0× for weeks −4…0 then ~0.9× after; and promo × festive is multiplicative (observed 3.36× vs 3.37× predicted from the two main effects).

**Validation:** rolling-origin backtest, 24 folds (origin every 2 weeks, 2025-09-29 → 2026-08-17), 4-week horizon, vs the three baselines
(`ml/train_forecast.py`, results in `ml/backtest_report.json`):

| | MAE | RMSE | WAPE |
|---|---|---|---|
| last value | 14.21 | 25.36 | 0.397 |
| 4-week average (best baseline) | 11.76 | 20.29 | 0.329 |
| same week last year | 12.77 | 20.96 | 0.357 |
| gradient boosting (challenger) | 9.14 | 15.53 | 0.255 |
| **shipped GLM** | **7.43** | **11.22** | **0.208** |

A separate fold mirrors the real task (origin 2025-09-15: forecast weeks 5,4,3,2 before Diwali, one festive season of history): GLM 0.187 vs
best baseline 0.293. Error is flat across horizons (0.205–0.211). Hyper-parameters (regularisation, promo shrinkage, Fourier terms,
category×festive terms, recency weights) were compared in the same backtest; all variants were within ~3% of each other except removing the trend,
which was clearly worse, so the choice is not fragile.

**Leakage guard:** `ml/backtest.py:20` — `train = sales[sales.week_start < origin]` — each fold sees nothing at or after its origin. The production fit
has the same guard at `ml/train_forecast.py:30` with an assertion at `:31`. Festival dates and promo flags are used only for the *target* week, because
both are known in advance. The weeks-to-Diwali feature uses the calendar only, never sales.

## 7. Worst SKUs, promotions, Diwali, rare and new SKUs

*Worst (established SKUs, backtest WAPE):* `BRK-1060` 0.38, `FIL-1031` 0.30, `ELE-1019` 0.30, `BRK-1021` 0.27, `OIL-1004` 0.27. These are low-volume
(≈11/week) or noisy; there is no signal left to find. The model still beats the 4-week average on **37 of 38** established SKUs.
*Promotions:* WAPE 0.192 in promo weeks vs 0.478 for the baseline; promo flags are known inputs. A SKU's own promo lift is estimated from few
promo weeks, hence the heavy shrinkage toward the pooled lift.
*Diwali:* modelled as weeks-to-Diwali, not calendar month, because Diwali moved 11 days between 2024 and 2025. In festive weeks WAPE is 0.205 vs 0.412.
The 2026 forecast weeks are 5, 4, 3, 2 weeks before Diwali (2026-11-08), so week 1 gets no lift and weeks 2–4 do. A same-week-last-year baseline gets this wrong:
last year's matching weeks were 3, 2, 1, 0 weeks before Diwali, so it overshoots.
*Rarely sold* (`BRK-1088`, `ELE-1006`, `GRP-1043`, ~0.6/week): WAPE 0.9–1.25, essentially tied with baselines; counts of 0/1/2 are noise. I forecast the **mean**, not the
median: the mean is what reorder maths needs (lead-time demand sums), though a median of 0 would give slightly lower MAE on these.
*New SKUs* (`BRK-1105`, `CHN-1006`, 6 weeks): the regularised GLM over-shrinks a SKU with a few weeks of data, so for SKUs under 26 weeks of history the forecast is rescaled by
actual/fitted over the last 3 weeks (`ml/models.py:145`, `COLD_START_WEEKS` `:141`). **Weak evidence:** the big backtest cannot test this (these SKUs don't exist in
any earlier training window); I checked it on only ~10 points (MAE 4.1 vs 7.6 without the correction). Treat the new-SKU forecasts as the least trustworthy numbers.

**Reorder** (`forecasting/service.py:72`): `lead_time_demand` sums the first `L` forecast weeks with `Decimal`; `safety_stock = ceil(1.65·σ·√L)` (`:78`) with σ the sample std (n−1) of the
last 12 weeks; stock is read live from `Inventory`. `forecasting/tests.py` recomputes all 40 SKUs independently and compares.

## 8. Weakest part, and two more days

1. **Live-LLM evidence is thin.** I ran `scripts/assistant_live_check.py` against Groq `openai/gpt-oss-120b`
   (3 runs × 15 checks, all passed), but only on one provider, one model and one dev database, and some
   checks are loose string tests (e.g. the oversell check looks for the available quantity in the reply).
   The grader's hidden cases will phrase things differently. If a reply keeps failing the grounding
   checker, users get the plain templated fallback: correct but stiff. Also, Groq retired
   `llama-3.3-70b-versatile` mid-project; the model-unavailable handling in `assistant/llm.py` exists because of it.
2. **Groq free-tier rate limits** are the likely real-world failure; each turn costs 2–4 model calls. The router retries and falls back to other keyed providers, but a
   single-provider setup under a burst will degrade to fallback replies.
3. Retrieval has no labelled evaluation set (answer 2c). Two days: build ~100 labelled queries, tune `MIN_SCORE` and the weights, and try a dense embedding as a third signal.
4. Forecast: no prediction intervals, so the safety stock uses historical σ rather than forecast error; new-SKU handling is validated on ~10 points.
5. Not built: order confirm/cancel and idempotency keys, supplier sync, frontend. `POST /api/orders/` is not duplicate-safe.
