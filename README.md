# Ciridae Takehome

AI-powered tool that compares a contractor's JDR estimate (Doc A) against an insurance estimate (Doc B), matches line items across both documents, and produces an annotated PDF report with green/orange/blue classification per item.

---

## Prerequisites

- **Python 3.13+** with [uv](https://docs.astral.sh/uv/) installed
- **Node.js 18+** with npm installed
- API access via a gateway key (see Environment below)

---

## Environment Setup

Create `apps/api/.env` from the example:

```bash
cp apps/api/.env.example apps/api/.env
```

Required variables in `.env`:

```env
DATABASE_URL=sqlite:///./data/local-dev.db
GATEWAY_API_KEY=your_gateway_api_key_here
LLM_GATEWAY_BASE_URL=https://llm-gateway-5q22j.ondigitalocean.app
```

The database file is created automatically on first run. No migrations needed for local dev.

---

## Running the API

```bash
cd apps/api
uv run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## Running the Web App

```bash
cd apps/web
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## Running Both Together

Open two terminals:

**Terminal 1 — API:**
```bash
cd apps/api && uv run uvicorn src.main:app --reload
```

**Terminal 2 — Web:**
```bash
cd apps/web && npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## Pipeline

The app runs a 4-step pipeline for each pair of uploaded PDFs:

| Step | Endpoint | Description |
|------|----------|-------------|
| 1 | `POST /uploads` | Upload JDR (A) + Insurance (B) PDFs → returns `run_id` |
| 2 | `POST /runs/{id}/extract` | LLM extracts line items from both PDFs |
| 3 | `POST /runs/{id}/map-rooms` | LLM maps room names across both documents |
| 4 | `POST /runs/{id}/match` | 2-pass LLM matching per room |
| 5 | `POST /runs/{id}/render` | Generates annotated PDF report |

Or run all steps at once with SSE streaming:

```
GET /runs/{id}/pipeline/stream
```

---

## Item Classification

| Color | Meaning |
|-------|---------|
| 🟢 Green | Same scope + all metadata (qty, unit_price, total) within ±2% |
| 🟠 Orange | Same scope but ≥1 metadata field differs beyond ±2% |
| 🔵 Blue | JDR-only — no matching item in the insurance estimate |
| Nugget | Insurance-only — present in B but not in A (shown in report summary) |

---

## Optional Configuration

All settings can be overridden in `.env`. Key options:

```env
# Models
EXTRACT_VISION_MODEL=gemini/gemini-2.5-pro
MATCHING_FIRST_PASS_MODEL=openai/gpt-4.1-mini
MATCHING_SECOND_PASS_MODELS=anthropic/claude-3-5-sonnet-latest
ROOMMAP_MODEL=openai/gpt-4.1

# Thresholds
MATCHING_GREEN_AMOUNT_TOLERANCE_PCT=0.02   # ±2% for green classification
RESCUE_GREEN_PRICE_TOL=0.05                # ±5% for price-proximity rescue → green
RESCUE_ORANGE_PRICE_TOL=0.15               # ±15% for price-proximity rescue → orange
```

---

## Project Structure

```
apps/
├── api/                  # FastAPI backend (Python 3.13, uv)
│   ├── src/
│   │   ├── main.py           # App entrypoint, router registration
│   │   ├── settings.py       # All config (env vars + defaults)
│   │   ├── extract_pdf_llm.py    # PDF extraction (vision + text)
│   │   ├── room_mapping.py       # LLM room name alignment
│   │   ├── matching_llm.py       # LLM matching calls (2-pass)
│   │   ├── routes_match.py       # Match logic + green/orange/blue classification
│   │   ├── render_report.py      # PDF annotation + summary report
│   │   └── routes_pipeline.py    # SSE streaming pipeline
│   └── data/
│       ├── local-dev.db      # SQLite dev database (auto-created)
│       └── uploads/          # Uploaded PDFs
└── web/                  # React + Vite frontend (TypeScript, Tailwind)
    └── src/
        └── App.tsx           # Main UI with SSE-based real-time progress
```
