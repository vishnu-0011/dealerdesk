# DealerDesk — AI/ML Track Submission

Backend core + **Dealer Assistant** (tool-calling LLM with grounding & safety guardrails) + **Demand Forecasting & Reorder Suggestions** for the VIKMO take-home evaluation.

- **Track:** AI / ML
- **Python Version:** 3.11+
- **Framework:** Django 5.1 + Django REST Framework
- **Primary LLM Provider:** Groq, model `openai/gpt-oss-120b` (with auto-fallback to `qwen/qwen3.6-27b`, Gemini, OpenAI, Anthropic)
- **Video Walkthrough:** _add your unlisted link here_
- **Design & Architecture Decisions:** Detailed rationale and answers in [`DECISIONS.md`](DECISIONS.md)

---

## 1. Quick Start (The Grader's Exact Path)

The application runs directly against a fresh PostgreSQL database.

### Linux / macOS (Bash)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
export DATABASE_URL=postgres://user:pass@localhost:5432/dealerdesk   # empty PostgreSQL database
export GROQ_API_KEY=your_groq_api_key                               # or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY

# 3. Migrate and seed data (seed_data is idempotent)
python manage.py migrate --noinput
python manage.py seed_data
python manage.py seed_data

# 4. Start the server
python manage.py runserver 0.0.0.0:8000 --noreload
```

### Windows (PowerShell)
```powershell
pip install -r requirements.txt
$env:DATABASE_URL = "postgres://user:pass@localhost:5432/dealerdesk"
$env:GROQ_API_KEY = "your_groq_api_key"
python manage.py migrate --noinput
python manage.py seed_data
python manage.py runserver 0.0.0.0:8000 --noreload
```

> **Note on `.env` file:** A `.env.example` file is included in the repository. You can copy it to `.env` for local development.

---

## 2. Verification & Testing Suite

| Check | Command | Description |
|---|---|---|
| **Contract Smoke Test** | `python smoke_test.py --base-url http://localhost:8000 --track aiml` | Automated validation of all required AI/ML contract endpoints (10/10 checks). |
| **Test Suite** | `python manage.py test` | 42 unit/integration tests covering financial arithmetic, concurrency, grounding, tools, retrieval, and safety. |
| **LLM Connectivity Check** | `python manage.py check_llm` | Diagnostic command testing plain chat and tool-calling execution across configured providers. |
| **Live Assistant Evaluation** | `python scripts/assistant_live_check.py --base-url http://localhost:8000 --runs 3` | 15 real-world scenarios repeated 3 times testing live stock, clarifying questions, injection resistance, and draft creation. |
| **Forecast Model Retraining** | `python -m ml.train_forecast` | Retrains Poisson GLM, generates `ml/forecast.csv`, and runs the 24-fold rolling backtest. |

---

## 3. Architecture & Core Components

```
c:\Project\dealerdesk\
├── assistant/          # Dealer Assistant: agent loop, tools, retrieval, grounding, safety, LLM router
├── catalog/            # Catalogue management: Products, Inventory, Dealers
├── core/               # Shared errors, custom exception handler, pagination, seed command
├── data/               # Seed datasets (catalogue, dealers, festivals, lead times, promotions, sales)
├── forecasting/        # Forecast & Reorder REST API endpoints and services
├── ml/                 # Demand forecast training, Poisson GLM, rolling backtest, forecast.csv
├── orders/             # Order engine: Draft orders, pricing, GST calculation, gapless numbering
├── scripts/            # Live LLM evaluation harness (assistant_live_check.py)
├── DECISIONS.md        # Deep-dive design decisions, backtest metrics, and trade-off analysis
└── smoke_test.py       # Grader contract compliance test script
```

### 3.1 Dealer Assistant (`assistant/`)
An intelligent, tool-calling conversational agent designed for automotive spare parts ordering:
- **Stateful Sessions:** `POST /api/assistant/chat/` maintains conversation history partitioned by `(session_id, dealer_id)` so context never leaks across dealers.
- **Tools Available to Model:**
  - `search_parts(query, vehicle)`: Hybrid BM25F + 3-gram character TF-IDF search with deterministic vehicle fitment resolution.
  - `check_stock(sku)`: Queries live inventory directly from PostgreSQL at turn execution time.
  - `create_draft_order(items)`: Creates DRAFT orders bound strictly to the calling dealer up to available stock.
- **Real-Time Grounding Verification (`grounding.py`):**
  - Every SKU, price, stock quantity, and order number in the model's reply is cross-checked against the database results returned by tools **in that specific turn**.
  - Prevents hallucinations and references to stale data from past turns.
  - Failing replies are returned to the model with corrective feedback (up to 2 retries). If still non-compliant, a deterministic fallback reply built directly from tool data is returned.
- **5-Layer Security & Prompt Injection Defense (`safety.py`, `prompts.py`):**
  1. *Zero description leakage:* Product descriptions are completely excluded from tool payloads sent to the LLM.
  2. *Text sanitisation:* Product names and metadata are NFKC-normalised; prompt injection patterns are stripped.
  3. *Prompt boundaries:* System prompt enforces strict role isolation and instructs that catalogue data is untrusted.
  4. *Output policy checks:* Replies promising unauthorized discounts, cancellations, or confirmations are rejected.
  5. *Capability limits:* The LLM can only create `DRAFT` orders and cannot confirm, cancel, or modify prices.

### 3.2 Demand Forecasting & Reorder Engine (`ml/`, `forecasting/`)
- **Forecasting Model:** A Poisson Generalized Linear Model (GLM) tailored for non-negative count demand with variance scaling with the mean. Features include:
  - Base demand level per SKU
  - Global market trend
  - Pooled multiplier per **week-relative-to-Diwali** ($\pm 4$ weeks) capturing festive seasonality
  - Pooled promotional lift with regularised SKU-level deviations
  - Cold-start correction for new SKUs with limited history (< 26 weeks)
- **Validation:** 24 rolling-origin backtest folds (every 2 weeks across historical data) evaluated over a 4-week forecast horizon:
  - **GLM WAPE: 0.208** (beats 4-week moving average baseline at 0.329, same-week-last-year at 0.357, and GBDT challenger at 0.255).
  - Outperforms the best baseline on **37 out of 38** established SKUs.
- **Leakage Prevention:** Hard temporal cutoffs ensure training data strictly precedes the forecast origin (`sales.week_start < origin`).
- **Reorder Suggestions:**
  - Lead-time demand: $\text{LTD} = \sum_{w=1}^{L} \text{forecast}_w$
  - Safety stock: $\text{SS} = \lceil 1.65 \cdot \sigma_{\text{12wk}} \cdot \sqrt{L} \rceil$
  - Reorder point: $\text{ROP} = \text{LTD} + \text{SS}$
  - Suggested quantity: $\max(0, \lceil \text{ROP} - \text{current\_stock} \rceil)$ using live inventory from `Inventory`.

### 3.3 Core Order Engine (`orders/`, `catalog/`)
- **Strict Monetary Arithmetic:** Uses Python `Decimal` with `ROUND_HALF_UP` to the nearest paisa.
- **Line-by-Line GST (18%):** Tax computed per line item and aggregated, strictly matching commercial invoicing rules.
- **Gapless Order Numbering:** Sequential format `ORD-YYYYMMDD-XXXX` guaranteed by row-level locking (`SELECT ... FOR UPDATE`) on `DailyOrderCounter`.

---

## 4. API Endpoints Reference

All API responses follow the standard envelope format. Error responses adhere to `{"error": {"code": "...", "message": "...", "details": [...]}}`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health/` | Health check (`{"status": "ok"}`). |
| `GET` | `/api/products/` | Paginated product list. Filters: `search`, `category`, `vehicle`. |
| `POST` | `/api/products/` | Create a product (`201` Created, initial stock 0). |
| `GET` | `/api/products/{sku}/` | Retrieve product details including current stock. |
| `PATCH` | `/api/products/{sku}/` | Update product attributes (e.g., price). |
| `PUT` | `/api/inventory/{sku}/` | Set stock quantity (`quantity` $\ge 0$, `reason`). |
| `GET` | `/api/dealers/` | Paginated list of registered dealers. |
| `POST` | `/api/dealers/` | Register a new dealer (`dealer_code`, `credit_limit`). |
| `GET` | `/api/dealers/{id}/` | Retrieve dealer details. |
| `GET` | `/api/orders/` | List orders. Filters: `dealer_id`, `status`. |
| `POST` | `/api/orders/` | Create a new DRAFT order with items and live pricing. |
| `GET` | `/api/orders/{order_number}/` | Retrieve order breakdown (subtotal, tax, grand total). |
| `POST` | `/api/assistant/chat/` | Chat with Dealer Assistant (`session_id`, `dealer_id`, `message`). |
| `GET` | `/api/forecast/{sku}/` | 4-week demand forecast for a SKU. |
| `GET` | `/api/reorder-suggestions/` | Reorder suggestions across all SKUs using live stock & lead times. |

---

## 5. Environment Variables & LLM Configuration

The application uses standard `requests` for LLM calls without proprietary vendor SDK lock-in.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | _None (Required)_ | PostgreSQL connection string (`postgres://user:pass@host:5432/dbname`). |
| `GROQ_API_KEY` | _Empty_ | API key for Groq. |
| `GEMINI_API_KEY` | _Empty_ | API key for Google Gemini. |
| `OPENAI_API_KEY` | _Empty_ | API key for OpenAI. |
| `ANTHROPIC_API_KEY` | _Empty_ | API key for Anthropic Claude. |
| `LLM_PROVIDER` | `groq` | Preferred primary provider (`groq`, `gemini`, `openai`, `anthropic`). |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model identifier for Groq. |
| `GROQ_FALLBACK_MODEL` | `qwen/qwen3.6-27b` | Secondary Groq model if primary is rate-limited or unavailable. |
| `ASSISTANT_REQUEST_BUDGET_SECONDS` | `48` | Time budget per chat turn before graceful fallback. |

---

## 6. Project Status & Test Coverage Summary

- **Automated Unit & Integration Tests:** 42 passing tests (`manage.py test`):
  - Decimal precision & Round-Half-Up taxation.
  - Multi-threaded order number generation under concurrency.
  - Reorder calculation independently verified against all 40 catalogue SKUs.
  - Agent retrieval, tool calling, grounding verification, and prompt injection defense.
- **Grader Smoke Test:** 10/10 checks passing (`smoke_test.py --track aiml`).
- **Live LLM Evaluation:** 15 scenarios tested with 3 repetitions (`scripts/assistant_live_check.py`):
  - Verified grounded price & stock citations from live DB.
  - Clarifying dialogues when vehicle or part type is ambiguous.
  - Rejection of out-of-scope requests and prompt manipulation attempts.
  - Proper draft order creation and refusal to exceed available stock.
- **Out of Scope (Optional Track Features Not Built):**
  - Supplier catalog synchronization (§8).
  - Order status state machine transitions beyond draft (§7.2–§7.4: confirm, cancel, dispatch).
  - Dedicated web frontend (§11).
