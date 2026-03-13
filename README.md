# Ciridae Takehome

AI-powered reconciliation tool that compares a contractor's JDR estimate against an insurance estimate, matches line items across both documents using a multi-step LLM pipeline, and produces an annotated PDF report with green/orange/blue classification per item.

---

## Prerequisites

- **Python 3.13+** with [uv](https://docs.astral.sh/uv/) installed
- **Node.js 18+** with npm installed
- A gateway API key (see Environment Setup below)

---

## Environment Setup

Copy the example env file and fill in your credentials:

```bash
cp apps/api/.env.example apps/api/.env
```

Required variables in `apps/api/.env`:

```env
DATABASE_URL=sqlite:///./data/local-dev.db
GATEWAY_API_KEY=your_gateway_api_key_here
LLM_GATEWAY_BASE_URL=https://llm-gateway-5q22j.ondigitalocean.app
```

The database is created automatically on first run — no migrations needed for local dev.

---

## Running the App

Open two terminals:

**Terminal 1 — API:**
```bash
cd apps/api && uv run uvicorn src.main:app --reload
```

**Terminal 2 — Web:**
```bash
cd apps/web && npm install && npm run dev
```

Then open `http://localhost:5173` in your browser. API docs are at `http://localhost:8000/docs`.

---

## Pipeline

Each uploaded PDF pair runs through a 5-step pipeline:

| Step | Endpoint | Description |
|------|----------|-------------|
| 1 | `POST /uploads` | Upload JDR (A) + Insurance (B) PDFs → returns `run_id` |
| 2 | `POST /runs/{id}/extract` | LLM extracts line items from both PDFs (vision + text fallback) |
| 3 | `POST /runs/{id}/map-rooms` | LLM aligns room names across both documents (handles renames, splits, merges) |
| 4 | `POST /runs/{id}/match` | 2-pass LLM matching per room with confidence scoring |
| 5 | `POST /runs/{id}/render` | Generates annotated PDF with highlights and summary page |

All steps can be run at once with real-time SSE streaming via `GET /runs/{id}/pipeline/stream`.

---

## Item Classification

| Label | Meaning |
|-------|---------|
| 🟢 Green | Same scope — qty, unit price, and total all within ±2% |
| 🟠 Orange | Same scope — but ≥1 metadata field differs beyond ±2% |
| 🔵 Blue | JDR-only — no matching item found in the insurance estimate |
| Nugget | Insurance-only — present in B but not in A (surfaced in report summary) |

---

## Project Structure

```
apps/
├── api/                          # FastAPI backend (Python 3.13, uv)
│   ├── src/
│   │   ├── main.py               # App entrypoint, router registration
│   │   ├── settings.py           # All config (env vars + defaults)
│   │   ├── extract_pdf_llm.py    # PDF extraction: vision primary, text fallback
│   │   ├── room_mapping.py       # Hybrid deterministic + LLM room alignment
│   │   ├── matching_llm.py       # 2-pass LLM matching with reviewer fallback
│   │   ├── routes_match.py       # Classification logic: green/orange/blue/nugget
│   │   ├── render_report.py      # PDF annotation + summary appendix
│   │   └── routes_pipeline.py    # SSE streaming pipeline orchestration
│   └── data/
│       ├── local-dev.db          # SQLite dev database (auto-created)
│       └── uploads/              # Uploaded PDFs per run
└── web/                          # React + Vite frontend (TypeScript, Tailwind)
    └── src/
        └── App.tsx               # SSE-based UI with real-time progress and results
```

---

## Optional Configuration

All settings can be overridden in `.env`. Key options:

```env
# Model selection
EXTRACT_VISION_MODEL=gemini/gemini-2.5-pro
MATCHING_FIRST_PASS_MODEL=openai/gpt-4.1-mini
MATCHING_SECOND_PASS_MODELS=anthropic/claude-3-5-sonnet-latest
ROOMMAP_MODEL=openai/gpt-4.1

# Classification thresholds
MATCHING_GREEN_AMOUNT_TOLERANCE_PCT=0.02   # ±2% for green classification
MATCHING_SECOND_PASS_TRIGGER_CONFIDENCE=0.90
```
