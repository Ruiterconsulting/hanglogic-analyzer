from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import math
import traceback
from supabase import create_client, Client
import httpx

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.8.0 - innerContours")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # evt. beperken tot jouw Lovable domeinen
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
# 🧠 Binnencontouren detectie
# -------------------------------
def extract_inner_contours(shape):
    contours = []
    for face_id, face in enumerate(shape.Faces()):
        try:
            inner_wires = face.Wires()
            for w in inner_wires:
                edges = w.Edges()
                if not edges:
                    continue
                pts = []
                for e in edges:
                    for v in e.Vertices():
                        p = v.toTuple()
                        pts.append([round(p[0], 3), round(p[1], 3), round(p[2], 3)])
                # Controleer of dit een echte binnencontour is (geen buitenrand)
                if len(pts) >= 3:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    cz = sum(p[2] for p in pts) / len(pts)
                    contours.append({
                        "id": len(contours) + 1,
                        "face_id": face_id,
                        "points": pts,
                        "center": [round(cx, 3), round(cy, 3), round(cz, 3)]
                    })
        except Exception:
            continue
    print(f"✅ Found {len(contours)} inner contours")
    return contours

# -------------------------------
# 🕳️ Strikte ronde gaten (optioneel)
# -------------------------------
def detect_analytic_holes_strict(shape, d_min=1.0, d_max=1000.0):
    circles = []
    for face in shape.Faces():
        for edge in face.Edges():
            if edge.geomType() != "CIRCLE":
                continue
            try:
                circ = edge._geomAdaptor().Circle()
                loc = circ.Location()
                r = float(circ.Radius())
                if r <= 0:
                    continue
                cx, cy, cz = float(loc.X()), float(loc.Y()), float(loc.Z())
                d = 2.0 * r
                if not (d_min <= d <= d_max):
                    continue
                circles.append({
                    "x": round(cx, 3),
                    "y": round(cy, 3),
                    "z": round(cz, 3),
                    "diameter": round(d, 3)
                })
            except Exception:
                continue
    print(f"🕳️ Detected {len(circles)} round holes (analytic).")
    return circles

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
    return {"message": "STEP analyzer live ✅ (v1.8.0 - innerContours + STL export)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧰 Bestandstype check
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

        # 1️⃣ STEP importeren
        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # 2️⃣ Bounding box
        bbox = shape.BoundingBox()
        raw_dims = {"x": round(bbox.xlen, 3), "y": round(bbox.ylen, 3), "z": round(bbox.zlen, 3)}
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 3️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 4️⃣ Detecties
        inner_contours = extract_inner_contours(shape)
        holes = detect_analytic_holes_strict(shape)
        holes_detected = len(holes)

        # 5️⃣ STL export
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 6️⃣ Upload
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # 7️⃣ Opslaan in Supabase database
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "holes_detected": holes_detected,
                "holes_data": holes,
                "inner_contours": inner_contours,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "volume_mm3": volume_mm3,
                "model_url": stl_public_url,
            }).execute()

        # 8️⃣ Resultaat terugsturen
        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "filename": file.filename,
            "holesDetected": holes_detected,
            "holes": holes,
            "innerContours": inner_contours,
            "modelURL": stl_public_url,
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

# -------------------------------
# 🌐 Proxy voor CORS
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
