from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import tempfile
import os
import numpy as np

# ✅ Supabase configuratie
SUPABASE_URL = "https://sywnjytfygvotskufvzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d25qeXRmeWd2b3Rza3VmdnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1ODAzMTAsImV4cCI6MjA3ODE1NjMxMH0.rwdyRnjOAG5pUrPufoZL13_O0HAQhuP8E2O_Al2kqMY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ FastAPI-app
app = FastAPI(title="HangLogic Analyzer", description="STEP/DXF analyser gekoppeld aan Supabase", version="1.0")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "HangLogic analyzer draait!"}

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """
    Upload een .STEP-bestand, analyseer geometrie (mock) en sla resultaat op in Supabase.
    """
    try:
        # ⏳ Bestand tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ⚙️ (Mock analyse — hier kan later pythonOCC of een CAD-analyzer komen)
        # Voor nu doen we alsof we de afmetingen en gaten analyseren
        dimensions = {
            "length": round(np.random.uniform(50, 500), 2),
            "width": round(np.random.uniform(20, 300), 2),
            "height": round(np.random.uniform(5, 200), 2),
        }
        holes = np.random.randint(1, 6)

        # 🧾 Resultaat opslaan in Supabase
        data = {
            "filename": file.filename,
            "dimensions": dimensions,
            "holes_detected": int(holes)
        }
        supabase.table("analyzed_parts").insert(data).execute()

        # 🧹 Opruimen
        os.remove(tmp_path)

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "dimensions": dimensions,
            "holes_detected": holes
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

