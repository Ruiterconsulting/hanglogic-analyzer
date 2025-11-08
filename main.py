import os, tempfile, uuid
from fastapi import FastAPI, UploadFile, File
from supabase import create_client, Client
import cadquery as cq

app = FastAPI(title="HangLogic Analyzer API")

SUPABASE_URL = "https://sywnjytfygvotskufvzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d25qeXRmeWd2b3Rza3VmdnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1ODAzMTAsImV4cCI6MjA3ODE1NjMxMH0.rwdyRnjOAG5pUrPufoZL13_O0HAQhuP8E2O_Al2kqMY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
def ping():
    return {"message": "STEP analyzer live ✅ (CadQuery mode)"}


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    try:
        # tijdelijk bestand schrijven
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # STEP inladen via cadquery
        shape = cq.importers.importStep(tmp_path)
        bb = shape.val().BoundingBox()

        result = {
            "id": str(uuid.uuid4()),
            "file_name": file.filename,
            "bounding_box_mm": {
                "x": round(bb.xlen, 2),
                "y": round(bb.ylen, 2),
                "z": round(bb.zlen, 2),
            },
        }

        supabase.table("analyzed_parts").insert(result).execute()
        os.remove(tmp_path)
        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "details": str(e)}
