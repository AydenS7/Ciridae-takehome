import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from .db import SessionLocal
from .models import Run

router = APIRouter(prefix="/uploads", tags=["uploads"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("")
async def upload(proposal_a: UploadFile = File(...), proposal_b: UploadFile = File(...)):
    run_id = str(uuid.uuid4())
    out_dir = UPLOAD_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_one(f: UploadFile) -> str:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail=f"{f.filename}: must be a PDF")
        out_path = out_dir / f.filename
        with out_path.open("wb") as w:
            w.write(f.file.read())
        return str(out_path)

    a_path = save_one(proposal_a)
    b_path = save_one(proposal_b)

    with SessionLocal() as db:
        run = Run(proposal_a_path=a_path, proposal_b_path=b_path)
        db.add(run)
        db.commit()
        db.refresh(run)

    return {"run_id": run.id, "proposal_a_path": a_path, "proposal_b_path": b_path}
