from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import traceback
import httpx
from supabase import create_client, Client

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# 🕳️ Gatdetectie helper (CadQuery)
# -------------------------------
def detect_cylindrical_holes(shape):
    """
    Detecteert cilinders in het model (gesloten of doorlopende gaten).
    Combineert boven- en onderzijden tot één gat.
    Retourneert lijst met {x, y, z, diameter}.
    """
    holes = []
    try:
        cyl_faces = []
        for face in shape.Faces():
            try:
                if face.geomType() == "CYLINDER":
                    surf = face.Surface()
                    radius = float(surf.Radius())
                    loc = surf.Location().toTuple()[0][:3]
                    cyl_faces.append({
                        "x": round(loc[0], 3),
                        "y": round(loc[1], 3),
                        "z": round(loc[2], 3),
                        "radius": round(radius, 3)
                    })
            except Exception:
                continue

        # Combineer cilinders met dezelfde XY (boven/onder)
        used = set()
        for i, f1 in enumerate(cyl_faces):
            if i in used:
                continue
            for j, f2 in enumerate(cyl_faces):
                if i >= j or j in used:
                    continue
                dx = abs(f1["x"] - f2["x"])
                dy = abs(f1["y"] - f2["y"])
                dr = abs(f1["radius"] - f2["radius"])
                if dx < 0.5 and dy < 0.5 and dr < 0.2:
                    z_mid = (f1["z"] + f2["z"]) / 2
                    holes.append({
                        "x": f1["x"],
                        "y": f1["y"],
                        "z": round(z_mid, 3),
                        "diameter": round(f1["radius"] * 2, 3)
                    })
                    used.add(i)
                    used.add(j)
                    break

        # Voeg losse cilinders toe (bijv. blinde gaten)
        for i, f in enumerate(cyl_faces):
            if i not in used:
                holes.append({
                    "x": f["x"],
                    "y": f["y"],
                    "z": f["z"],
                    "diameter": round(f["radius"] * 2, 3)
                })

        print(f"🕳️ Detected {len(holes)} cylindrical holes")
    except Exception as e:
        print("⚠️ Hole detection failed:", e)
    return holes


# -------------------------------
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")
    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)
    try:
        # Verwijder oud bestand
        try:
            files = storage.list(path="analyzed")
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    print(f"⚠️ Bestand {remote_name} bestaat al — verwijderen...")
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden:", e)

        with open(local_path, "rb") as f:
            res = storage.upload(remote_path, f)
        print("✅ Upload gelukt:", res)

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
    return {"message": "STEP analyzer live ✅ (CadQuery + Cylindrical holes + Supabase)"}

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
    tmp_path = None
    try:
        _ensure_step_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        print("🧠 Analyzing file:", file.filename)

        # Bounding box
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(float(bbox.xlen), 3),
            "y": round(float(bbox.ylen), 3),
            "z": round(float(bbox.zlen), 3),
        }
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # Export STL
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # Hole detection (via CadQuery)
        holes = detect_cylindrical_holes(shape)
        holes_detected = len(holes)

        # Upload STL
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # Opslaan in Supabase
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "holes_detected": holes_detected,
                "holes_data": holes,
                "units": "mm",
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "model_url": stl_public_url,
                "created_at": "now()",
            }).execute()

        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
                "filename": file.filename,
                "holesDetected": holes_detected,
                "holes": holes,
                "modelURL": stl_public_url,
            }
        )

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}", "trace": tb})
    finally:
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


# -------------------------------
# 🌐 Proxy endpoint (CORS-fix)
# -------------------------------
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    """
    Haalt STL-bestand op van Supabase Storage en voegt juiste CORS headers toe.
    Hierdoor kan Lovable het model rechtstreeks renderen.
    """
    url = f"https://sywnjytfygvotskufvzs.supabase.co/storage/v1/object/public/{path}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": r.headers.get("content-type", "application/octet-stream"),
        }
        return StreamingResponse(iter([r.content]), headers=headers)
    except Exception as e:
        print("⚠️ Proxy fetch failed:", e)
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {str(e)}"})
