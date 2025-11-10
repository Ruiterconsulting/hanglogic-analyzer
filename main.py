from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import traceback
import trimesh
import httpx
from supabase import create_client, Client

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.7.0")

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
# 🕳️ Gatdetectie helper (Trimesh)
# -------------------------------
def detect_mesh_holes(stl_path):
    """
    Detect holes from a triangulated STL mesh.
    Finds circular boundary loops (through-holes).
    """
    holes = []
    try:
        mesh = trimesh.load_mesh(stl_path)
        boundaries = mesh.boundary_edges  # gebruik boundary_edges, niet boundary_vertices
        if boundaries is None or len(boundaries) == 0:
            print("⚠️ No open boundaries found.")
            return holes

        for loop in boundaries:
            pts = mesh.vertices[list(loop)]
            if len(pts) < 6:
                continue
            center = pts.mean(axis=0)
            dists = ((pts - center) ** 2).sum(axis=1) ** 0.5
            diameter = 2 * dists.mean()
            holes.append({
                "x": round(float(center[0]), 3),
                "y": round(float(center[1]), 3),
                "z": round(float(center[2]), 3),
                "diameter": round(float(diameter), 3)
            })
        print(f"🕳️ Detected {len(holes)} mesh holes")
    except Exception as e:
        print("⚠️ Mesh hole detection failed:", e)
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
        # Verwijder oud bestand (indien aanwezig)
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
    return {"message": "STEP analyzer live ✅ (CadQuery + STL + Trimesh holes + Supabase)"}

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

        # Hole detection (mesh)
        holes = detect_mesh_holes(stl_path)
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
# 🌐 Proxy endpoint (CORS-fix voor STL viewer)
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
