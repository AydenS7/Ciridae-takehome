from fastapi import FastAPI
from .settings import settings
from .routes_upload import router as upload_router

app = FastAPI(title="Ciridae Takehome API")
app.include_router(upload_router)

@app.get("/health")
def health():
    return {"ok": True, "llm_base_url": settings.llm_gateway_base_url}
