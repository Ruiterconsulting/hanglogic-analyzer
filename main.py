# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import cadquery as cq
import tempfile
import os
import traceback

# === Initialize FastAPI ===
app = FastAPI(title="HangLogic Analyzer API", version="1.0.0")

# === Allow CORS (for Lovable frontend) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # eventueel beperken tot jouw domeinen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Supabase setup ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Routes ===
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery mode)"}

@app.get("/health")
def health():
    return {"status": "ok"}


def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP"
        )

# === Main Analyze Route ===
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """
    Ontvangt een .STEP/.STP bestand, laadt het met CadQuery,
    berekent bounding box en volume, en slaat alles op in Supabase.
    """
    try:
        # 1️⃣ Validatie bestandstype
        _ensure_step_extension(file.filename)

        # 2️⃣ Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3️⃣ STEP importeren
        model = cq.importers.importStep(tmp_path)
        shape = model.val()  # shape extraheren

        # 4️⃣ Bounding box berekenen (in mm)
        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3),
        }

        # 5️⃣ Volume bepalen
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 6️⃣ Verwijder tijdelijk bestand
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # 7️⃣ Opslaan in Supabase
        try:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {
                    "x": dims["X"],
                    "y": dims["Y"],
                    "z": dims["Z"]
                },
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "volume_mm3": round(volume_mm3, 3) if volume_mm3 else None,
                "units": "mm",
                "holes_detected": 0
            }).execute()
        except Exception as e:
            print("⚠️ Error saving to Supabase:", e)

        # 8️⃣ Response naar frontend
        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
                "filename": file.filename,
            }
        )

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}", "trace": tb},
        )
