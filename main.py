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
app = FastAPI(title="HangLogic Analyzer API", version="1.2.0")

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
# 📦 Upload helper
# -------------------------------
print("Buckets:", supabase.storage.list_buckets())

def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """Upload een bestand naar Supabase Storage en geef publieke URL terug."""
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"

    with open(local_path, "rb") as f:
        # upload zonder bools
        supabase.storage.from_(bucket).upload(remote_path, f)

    # publieke URL ophalen
    public_url = supabase.storage.from_(bucket).get_public_url(remote_path)
    return public_url

# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery + STL export)"}

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
    bepaalt bounding box, volume, exporteert STL en slaat op in Supabase.
    """
    tmp_path = None
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
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 5️⃣ Volume berekenen
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # -------------------------------
        # 🧱 STL-export
        # -------------------------------
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # Upload naar Supabase
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # -------------------------------
        # 💾 Opslaan in Supabase database
        # -------------------------------
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "holes_detected": 0,
                "created_at": "now()",
                "units": "mm",
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "model_url": stl_public_url,
            }).execute()

        # -------------------------------
        # ✅ Terug naar frontend
        # -------------------------------
        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
                "filename": file.filename,
                "modelURL": stl_public_url,
            }
        )

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}", "trace": tb})
    finally:
        # Alle tijdelijke bestanden opruimen
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
