from fastapi import FastAPI
from .settings import settings
from .routes_upload import router as upload_router
from .routes_extract import router as runs_router
from .routes_roommap import router as roommap_router
from .db import init_db

app = FastAPI(title="Ciridae Takehome API")
app.include_router(upload_router)
app.include_router(runs_router)
app.include_router(roommap_router)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "llm_base_url": settings.llm_gateway_base_url}
