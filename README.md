# Razorpay CloseLoop — Financial Operations

AI-assisted financial reconciliation and exception-resolution system that detects reconciliation issues, explains them using financial evidence, applies safety and verification controls, and keeps humans in the loop for resolution and learning.

## Problem

Financial reconciliation across payments, settlements, refunds, fees, taxes, and adjustments produces exceptions that require investigation. Manual review is slow, inconsistent, and misses patterns. CloseLoop automates detection, evidence gathering, risk assessment, and resolution — while keeping guardrails and human judgment in the loop.

## Key Capabilities

- **Deterministic Reconciliation** — Multi-way matching across financial records
- **Evidence Collection** — Structured financial evidence with coverage analysis
- **ML Classification** — Exception type prediction with similarity to historical cases
- **Resolution Candidates** — Ranked resolution options with confidence scores
- **Financial Guardrails** — Safety checks that block unsafe automated resolutions
- **Verification** — Post-resolution verification before execution
- **Human Review** — Controlled routing to human reviewers when guardrails block automation
- **LLM Explanation** — Natural language explanations grounded in financial evidence
- **Feedback Learning** — Reviewer decisions feed back into system improvement

## Architecture

```
Financial Records → Reconciliation → Exceptions → Evidence → Classification
                                                                ↓
Guardrails ← Candidates ← Similarity ← ML Features ←───────────┘
    ↓
Decision (Auto / Human Review / Blocked)
    ↓
Verification → Execution → Feedback → Learning
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, Pydantic |
| Database | PostgreSQL 16, pgvector |
| ML | scikit-learn, XGBoost, sentence-transformers |
| Agent | LangGraph |
| LLM | OpenAI / Anthropic (optional) |
| MCP | Model Context Protocol tools |
| Frontend | Next.js, TypeScript, Tailwind CSS, Recharts |
| Orchestration | Docker Compose |

## Project Structure

```
razorpay-closeloop/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes and services
│   │   ├── core/          # Structured logging, config
│   │   ├── agent/         # LangGraph workflow
│   │   ├── guardrails/    # Financial safety checks
│   │   ├── llm/           # LLM explanation layer
│   │   ├── ml/            # ML classification, similarity
│   │   └── reconciliation/
│   ├── mcp/               # MCP tools and adapters
│   ├── data/              # Synthetic financial data
│   ├── scripts/           # Demo seeding scripts
│   ├── tests/             # Backend tests (5000+)
│   └── Dockerfile
├── frontend/
│   ├── app/               # Next.js pages
│   ├── components/        # UI components
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Seed demo data
python scripts/seed_demo.py

# Start server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- -p 3000
```

Open:
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (in-memory) | PostgreSQL connection string |
| `LLM_ENABLED` | `false` | Enable LLM explanations |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai`, `anthropic`) |
| `LLM_OPENAI_API_KEY` | — | OpenAI API key |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model to use |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL |

## LLM Configuration

CloseLoop works **without an LLM** using deterministic template-based explanations. When an LLM API key is configured:

```bash
# In backend/.env or environment:
LLM_ENABLED=true
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-your-key-here
```

The LLM is used for natural language explanation only. It never makes financial decisions, sets amounts, or bypasses guardrails.

## Running with Docker

```bash
docker compose up --build
```

Services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

Demo data is seeded automatically on backend startup.

## API Documentation

Full interactive docs at http://localhost:8000/docs

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/metrics` | GET | System-wide reconciliation metrics |
| `/metrics/safety` | GET | Guardrail and verification metrics |
| `/metrics/throughput` | GET | Processing throughput |
| `/exceptions` | GET | List all exceptions (filterable) |
| `/exceptions/{id}` | GET | Exception detail |
| `/exceptions/{id}/evidence` | GET | Financial evidence for exception |
| `/exceptions/{id}/explain` | GET | LLM/deterministic explanation |
| `/exceptions/{id}/analyze` | GET | Full AI-assisted analysis |
| `/exceptions/{id}/similar` | GET | Similar historical cases |
| `/exceptions/{id}/approve` | POST | Approve exception |
| `/exceptions/{id}/reject` | POST | Reject exception |
| `/exceptions/{id}/escalate` | POST | Escalate to human review |
| `/batches` | GET/POST | List or create batches |
| `/batches/{id}/run` | POST | Run batch reconciliation |
| `/learning/metrics` | GET | Learning and feedback metrics |
| `/models` | GET | Model registry |
| `/feedback` | GET/POST | Feedback records |

## Demo Workflow

1. **Open Control Center** — See reconciliation overview, risk distribution, exception types
2. **Open Exceptions** — Browse 30 curated financial reconciliation exceptions
3. **Select a case** — View financial difference, risk level, status
4. **View Evidence** — See payment, settlement, refund, fee records
5. **Run Analysis** — ML classification and resolution recommendation
6. **Read Explanation** — AI/deterministic explanation of the exception
7. **Take Action** — Approve, Reject, or Escalate
8. **View Learning** — See how reviewer decisions become learning data

### Demo Dataset

- 30 curated exceptions across 9 types
- Risk distribution: LOW (13), MEDIUM (9), HIGH (8)
- Status distribution: APPROVED (7), PENDING (15), ESCALATED (4), REJECTED (4)
- Types: Exact Match, Partial Settlement, Timing Difference, Tax Adjustment, Fee Difference, Duplicate, Missing Record, Complex Multi-Adjustment, Unknown

## Testing

```bash
cd backend
pytest                    # Run all tests (5000+)
pytest tests/test_safety  # Safety-specific tests
pytest tests/test_guardrails  # Guardrail tests
```

## Known Limitations

- **LLM explanations** use deterministic templates when no API key is configured
- **Safety metrics** (auto_decisions, guardrail_blocks) return zeros unless the full LangGraph workflow has processed batches through the guardrail engine
- **Batch registry** is in-memory — batch data resets on server restart (exception data persists on disk)
- **pgvector similarity** uses in-memory fallback when PostgreSQL is not configured
- **No CI/CD pipeline** included
- **No authentication** — intended for internal deployment behind existing auth infrastructure

## License

Internal — Razorpay.
