# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cadquery as cq
import tempfile
import os
import traceback
from supabase import create_client, Client

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # eventueel beperken tot jouw Lovable domeinen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🔑 Supabase setup
# -------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client | None = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase.")
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery mode)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧰 Hulpfunctie
# -------------------------------
def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP"
        )

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """
    Ontvangt een .STEP/.STP bestand, laadt het met CadQuery,
    bepaalt bounding box (mm) en volume (mm³),
    slaat resultaat op in Supabase.
    """
    try:
        # 1️⃣ Validatie
        _ensure_step_extension(file.filename)

        # 2️⃣ Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3️⃣ Importeren in CadQuery
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # 4️⃣ Bounding box bepalen
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(float(bbox.xlen), 3),
            "y": round(float(bbox.ylen), 3),
            "z": round(float(bbox.zlen), 3),
        }

        # Sorteer op grootte: X=langste, Y=middelste, Z=kleinste
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {
            "X": sorted_dims[0],
            "Y": sorted_dims[1],
            "Z": sorted_dims[2],
        }

        # 5️⃣ Volume berekenen (kan mislukken bij open shapes)
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 6️⃣ Opslaan in Supabase
        if supabase:
            try:
                supabase.table("analyzed_parts").insert({
                    "filename": file.filename,
                    "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                    "holes_detected": 0,
                    "created_at": "now()",
                    "units": "mm",
                    "bounding_box_x": dims["X"],
                    "bounding_box_y": dims["Y"],
                    "bounding_box_z": dims["Z"],
                }).execute()
            except Exception as e:
                print("⚠️ Error saving to Supabase:", e)

        # 7️⃣ Terugsturen naar frontend
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
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
