from fastapi import FastAPI
from .settings import settings

app = FastAPI(title="Ciridae Takehome API")

@app.get("/health")
def health():
    return {
        "ok": True,
        "db": settings.database_url.split("@")[-1],  # avoids printing password
        "llm_base_url": settings.llm_gateway_base_url,
    }
