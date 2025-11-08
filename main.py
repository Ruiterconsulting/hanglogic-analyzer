from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
import tempfile, os, uuid

# --- Supabase setup ---
SUPABASE_URL = "https://sywnjytfygvotskufvzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d25qeXRmeWd2b3Rza3VmdnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1ODAzMTAsImV4cCI6MjA3ODE1NjMxMH0.rwdyRnjOAG5pUrPufoZL13_O0HAQhuP8E2O_Al2kqMY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI app ---
app = FastAPI(title="HangLogic STEP Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "STEP analyzer is live ✅"}


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """Upload .STEP file → analyze geometry → store result in Supabase"""
    try:
        # Tijdelijk bestand aanmaken
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # STEP-bestand inlezen
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(tmp_path)
        if status != 1:
            return {"status": "error", "details": "Kon STEP-bestand niet lezen"}

        step_reader.TransferRoots()
        shape = step_reader.OneShape()

        # Bounding box berekenen
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        result = {
            "id": str(uuid.uuid4()),
            "file_name": file.filename,
            "bounding_box_mm": {
                "x": round(xmax - xmin, 2),
                "y": round(ymax - ymin, 2),
                "z": round(zmax - zmin, 2),
            }
        }

        # Wegschrijven naar Supabase
        supabase.table("analyzed_parts").insert(result).execute()

        # Opschonen
        os.remove(tmp_path)
        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "details": str(e)}
