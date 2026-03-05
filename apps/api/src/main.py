from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .routes_upload import router as upload_router
from .routes_extract import router as runs_router
from .routes_roommap import router as roommap_router
from .routes_match import router as match_router
from .routes_report import router as report_router
from .routes_pipeline import router as pipeline_router
from .db import init_db

app = FastAPI(title="Ciridae Takehome API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload_router)
app.include_router(runs_router)
app.include_router(roommap_router)
app.include_router(match_router)
app.include_router(report_router)
app.include_router(pipeline_router)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "llm_base_url": settings.llm_gateway_base_url}
