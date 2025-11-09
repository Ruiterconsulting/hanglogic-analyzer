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
app = FastAPI(title="HangLogic Analyzer API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later kun je dit beperken tot jouw Lovable domeinen
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
        # Log beschikbare buckets
        try:
            buckets = supabase.storage.list_buckets()
            print("📦 Buckets found:", [b.name for b in buckets])
        except Exception as e:
            print("⚠️ Could not list buckets:", e)
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")


# -------------------------------
# 📦 Upload helper (definitieve versie)
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """
    Upload bestand naar Supabase Storage.
    Bestaat het al → dan verwijderen we het eerst.
    Retourneert de publieke URL.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)

    try:
        # 🔹 Bestaat het bestand al? -> verwijder het
        try:
            files = storage.list(path="analyzed")
            for f in files:
                if f["name"] == remote_name:
                    print(f"⚠️ Bestand {remote_name} bestaat al — verwijderen...")
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden of kon niet lezen:", e)

        # 🔹 Upload nieuw bestand
        with open(local_path, "rb") as f:
            res = storage.upload(remote_path, f)
        print("✅ Upload gelukt:", res)

        # 🔹 Publieke URL ophalen
        public_url = storage.get_public_url(remote_path)
        print("🌍 Public URL:", public_url)
        return public_url

    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")


# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery + STL export + Supabase upload)"}

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
    Ontvangt een .STEP/.STP bestand, berekent bounding box en volume,
    exporteert STL, uploadt naar Supabase, en slaat metadata op.
    """
    tmp_path = None
    try:
        # 1️⃣ Validatie
        _ensure_step_extension(file.filename)

        # 2️⃣ Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3️⃣ Inladen in CadQuery
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # 4️⃣ Bounding box
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(float(bbox.xlen), 3),
            "y": round(float(bbox.ylen), 3),
            "z": round(float(bbox.zlen), 3),
        }
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 5️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 6️⃣ STL-export
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 7️⃣ Upload naar Supabase
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # 8️⃣ Opslaan in database
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

        # 9️⃣ Antwoord naar frontend
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
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}", "trace": tb},
        )

    finally:
        # Verwijder tijdelijke bestanden
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
