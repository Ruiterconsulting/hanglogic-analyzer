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
# 🕳️ Hole detection via CadQuery faces
# -------------------------------
def detect_analytic_holes(shape):
    """
    Detect holes by analyzing circular faces and aligned axes.
    Works for both through and blind holes.
    """
    holes = []
    try:
        circle_faces = []
        for face in shape.Faces():
            geom_type = face.geomType()
            if geom_type in ["CYLINDER", "PLANE"]:
                for edge in face.Edges():
                    if edge.geomType() == "CIRCLE":
                        circ = edge._geomAdaptor().Circle()
                        loc = circ.Location()
                        center = (loc.X(), loc.Y(), loc.Z())
                        radius = circ.Radius()
                        normal = face.normalAt(0.5, 0.5).toTuple()
                        circle_faces.append({
                            "center": center,
                            "radius": radius,
                            "normal": normal
                        })

        used = set()
        for i, f1 in enumerate(circle_faces):
            if i in used:
                continue
            for j, f2 in enumerate(circle_faces):
                if i >= j or j in used:
                    continue
                dx = abs(f1["center"][0] - f2["center"][0])
                dy = abs(f1["center"][1] - f2["center"][1])
                dr = abs(f1["radius"] - f2["radius"])
                if dx < 0.5 and dy < 0.5 and dr < 0.2:
                    z_mid = (f1["center"][2] + f2["center"][2]) / 2
                    holes.append({
                        "x": round(f1["center"][0], 3),
                        "y": round(f1["center"][1], 3),
                        "z": round(z_mid, 3),
                        "diameter": round(f1["radius"] * 2, 3)
                    })
                    used.add(i)
                    used.add(j)
                    break

        # Als een cirkel geen tegenhanger heeft → blind hole
        for i, f in enumerate(circle_faces):
            if i not in used:
                holes.append({
                    "x": round(f["center"][0], 3),
                    "y": round(f["center"][1], 3),
                    "z": round(f["center"][2], 3),
                    "diameter": round(f["radius"] * 2, 3)
                })

        print(f"🕳️ Detected {len(holes)} holes (analytic method)")
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
        # Verwijder eventueel bestaand bestand
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

        return storage.get_public_url(remote_path)
    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        if not file.filename.lower().endswith((".step", ".stp")):
            raise HTTPException(status_code=415, detail="Only .STEP/.STP allowed")

        # Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Inladen en analyseren
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3)
        }

        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # Hole detection
        holes = detect_analytic_holes(shape)
        holes_detected = len(holes)

        # STL export
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # Supabase DB insert
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
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
                "volumeMM3": volume_mm3,
                "filename": file.filename,
                "holesDetected": holes_detected,
                "holes": holes,
                "modelURL": stl_public_url,
            }
        )

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
# 🌐 Proxy endpoint (CORS)
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
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {e}"})
