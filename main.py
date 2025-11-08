from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import cadquery as cq
import tempfile, os, uuid

SUPABASE_URL = "https://sywnjytfygvotskufvzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d25qeXRmeWd2b3Rza3VmdnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1ODAzMTAsImV4cCI6MjA3ODE1NjMxMH0.rwdyRnjOAG5pUrPufoZL13_O0HAQhuP8E2O_Al2kqMY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="HangLogic Analyzer API (STEP Parser)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "STEP Analyzer API active ✅"}

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """Lees STEP-file, bereken bounding box, volume, holes"""
    try:
        # Tijdelijk bestand
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Analyse via CadQuery
        model = cq.importers.importStep(tmp_path)
        bounding_box = model.val().BoundingBox()
        holes = [f for f in model.faces() if "Circle" in str(f.geomType())]
        volume = model.val().Volume()

        dims = {
            "x_mm": round(bounding_box.xlen, 2),
            "y_mm": round(bounding_box.ylen, 2),
            "z_mm": round(bounding_box.zlen, 2),
        }

        result = {
            "id": str(uuid.uuid4()),
            "file_name": file.filename,
            "dimensions_mm": dims,
            "holes_detected": len(holes),
            "volume_mm3": round(volume, 2),
        }

        # Opslaan in Supabase
        supabase.table("analyzed_parts").insert(result).execute()

        # Verwijder tijdelijk bestand
        os.remove(tmp_path)

        return {"status": "success", "analysis": result}

    except Exception as e:
        return {"status": "error", "details": str(e)}
