from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from datetime import datetime
import aiofiles
import os

# ──────────────────────────────
# 🔧 Supabase setup
# ──────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ──────────────────────────────
# 🚀 FastAPI App
# ──────────────────────────────
app = FastAPI(title="HangLogic Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────
# 📁 Upload endpoint
# ──────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not (file.filename.endswith(".step") or file.filename.endswith(".dxf")):
            raise HTTPException(status_code=400, detail="Only STEP and DXF files are supported.")

        # Save locally before upload
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Upload to Supabase Storage (bucket: 'uploads')
        with open(temp_path, "rb") as f:
            storage_path = f"uploads/{file.filename}"
            supabase.storage.from_("uploads").upload(storage_path, f)

        # Insert metadata into table
        data = {
            "file_name": file.filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "uploaded"
        }
        supabase.table("analyzed_parts").insert(data).execute()

        return {"message": "✅ File uploaded successfully!", "file_name": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────
# 🧠 Health check
# ──────────────────────────────
@app.get("/")
def root():
    return {"message": "HangLogic Analyzer API running!"}
