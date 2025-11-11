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
app = FastAPI(title="HangLogic Analyzer API", version="2.0.0")

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
# 🕳️ Detectie van binnencontouren
# -------------------------------
def detect_inner_contours(shape):
    """
    Detecteert alle binnencontouren (inner wires) op vlakken van het model.
    Retourneert lijst met punten + center.
    """
    contours = []
    try:
        face_index = 0
        for face in shape.Faces():
            wires = face.Wires()
            if len(wires) <= 1:
                continue  # geen binnencontouren
            outer_wire = wires[0]
            for wire in wires[1:]:
                pts = []
                for v in wire.Vertices():
                    p = v.toTuple()
                    pts.append([round(p[0], 3), round(p[1], 3), round(p[2], 3)])
                if not pts:
                    continue
                # bereken center
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                cz = sum(p[2] for p in pts) / len(pts)
                contours.append({
                    "id": len(contours) + 1,
                    "face_id": face_index,
                    "points": pts,
                    "center": [round(cx, 3), round(cy, 3), round(cz, 3)]
                })
            face_index += 1
        print(f"🟦 Found {len(contours)} inner contours.")
    except Exception as e:
        print("⚠️ Inner contour detection failed:", e)
    return contours

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
        # verwijder oude versie
        try:
            files = storage.list(path="analyzed")
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception:
            pass
        with open(local_path, "rb") as f:
            storage.upload(remote_path, f)
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
    return {"message": "STEP analyzer v2 ✅ (inner contour detection + STL + Supabase)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Validatie
        ext = (os.path.splitext(file.filename)[1] or "").lower()
        if ext not in [".step", ".stp"]:
            raise HTTPException(status_code=415, detail="Only .STEP/.STP files allowed")

        # Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Laden in CadQuery
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # Bounding box
        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3),
        }

        # Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # Binnencontouren detecteren
        inner_contours = detect_inner_contours(shape)

        # STL exporteren
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # Uploaden
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # Opslaan in database
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "inner_contours": inner_contours,
                "created_at": "now()",
                "units": "mm",
                "model_url": stl_public_url
            }).execute()

        # Terug naar frontend
        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
                "filename": file.filename,
                "innerContours": inner_contours,
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
# 🌐 Proxy endpoint
# -------------------------------
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
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
