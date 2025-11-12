from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
from cadquery import exporters
import tempfile
import os
import math
import traceback
from supabase import create_client, Client
import httpx
import trimesh
import uuid
from datetime import datetime

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="2.2.0")

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
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# -------------------------------
# 🕳️ Detectie ronde gaten (info)
# -------------------------------
def detect_analytic_holes_strict(
    shape,
    d_min=2.0, d_max=30.0,
    xy_tol=0.2, d_tol=0.2,
    z_pair_min=1.0, z_pair_max=40.0,
    full_circle_tol_ratio=0.02
):
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
                L = float(edge.Length())
                full_L = 2.0 * math.pi * r
                if abs(L - full_L) > full_circle_tol_ratio * full_L:
                    continue
                cx, cy, cz = float(loc.X()), float(loc.Y()), float(loc.Z())
                d = 2.0 * r
                if not (d_min <= d <= d_max):
                    continue
                circles.append((cx, cy, cz, d))
            except Exception:
                continue

    if not circles:
        return []

    groups = []
    for cx, cy, cz, d in circles:
        placed = False
        for g in groups:
            if (
                abs(g["x"] - cx) <= xy_tol and
                abs(g["y"] - cy) <= xy_tol and
                abs(g["d"] - d) <= d_tol
            ):
                g["zs"].append(cz)
                placed = True
                break
        if not placed:
            groups.append({"x": cx, "y": cy, "d": d, "zs": [cz]})

    holes = []
    for g in groups:
        zs = sorted(g["zs"])
        if len(zs) < 2:
            continue
        for i in range(len(zs) - 1):
            dz = abs(zs[i + 1] - zs[i])
            if z_pair_min <= dz <= z_pair_max:
                z_avg = (zs[i] + zs[i + 1]) / 2.0
                holes.append({
                    "x": round(g["x"], 3),
                    "y": round(g["y"], 3),
                    "z": round(z_avg, 3),
                    "diameter": round(g["d"], 3),
                    "dz": round(dz, 3)
                })
                break
    print(f"🕳️ Detected {len(holes)} clean through-holes (strict).")
    return holes

# -------------------------------
# 📦 Upload helper (unieke namen)
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(remote_name)
    unique_name = f"{name}_{timestamp}_{unique_id}{ext}"
    remote_path = f"analyzed/{unique_name}"
    storage = supabase.storage.from_(bucket)

    try:
        with open(local_path, "rb") as f:
            storage.upload(remote_path, f)
        public_url = storage.get_public_url(remote_path)
        print(f"✅ Uploaded new file version: {unique_name}")
        return public_url
    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")

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
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer ✅ (v2.2.0) — GLB + full-thickness contour fill"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    stl_path = None
    glb_path = None
    try:
        _ensure_step_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # Bounding box & volume
        bbox = shape.BoundingBox()
        dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3)
        }
        volume = float(shape.Volume()) if shape.Volume() else None

        # Detectie ronde gaten (voor rapportage)
        strict_holes = detect_analytic_holes_strict(shape)

        # STL export
        base_name, _ = os.path.splitext(file.filename)
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")
        stl_url = upload_to_supabase(stl_path, f"{base_name}.stl")

        # 🌈 GLB aanmaken
        green = [0, 255, 0, 255]
        light_blue = [150, 200, 255, 255]

        mesh = trimesh.load_mesh(stl_path)
        mesh.visual.vertex_colors = [light_blue] * len(mesh.vertices)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="body")

        # Voeg exacte contourvullingen toe — over volledige plaatdikte
        for f_idx, face in enumerate(shape.Faces()):
            try:
                outer = face.outerWire()
            except Exception:
                outer = None

            for w_idx, wire in enumerate(face.Wires()):
                if outer and wire.isSame(outer):
                    continue
                try:
                    normal = face.normalAt(0.5, 0.5)
                    # Vul volledige plaatdikte (99% van totale Z-lengte)
                    extrusion_depth = bbox.zlen * 0.99 * (1 if normal.z >= 0 else -1)
                    solid = cq.Workplane().add(wire).toPending().extrude(extrusion_depth)
                    tmp_fill = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                    exporters.export(solid, tmp_fill.name, "STL")
                    filled = trimesh.load_mesh(tmp_fill.name)
                    filled.visual.vertex_colors = [green] * len(filled.vertices)
                    scene.add_geometry(filled, node_name=f"fill_{f_idx}_{w_idx}")
                    tmp_fill.close()
                    os.remove(tmp_fill.name)
                except Exception as e:
                    print(f"⚠️ Failed to extrude contour on face {f_idx}: {e}")

        # GLB exporteren
        glb_bytes = scene.export(file_type="glb")
        glb_path = tmp_path.replace(".step", ".glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        glb_url = upload_to_supabase(glb_path, f"{base_name}.glb")

        # Opslaan in DB
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "holes_detected": len(strict_holes),
                "holes_data": strict_holes,
                "units": "mm",
                "model_url": stl_url,
                "model_url_glb": glb_url
            }).execute()

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume, 3) if volume else None,
            "holesDetected": len(strict_holes),
            "modelURL": stl_url,
            "modelURLGLB": glb_url
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        for path in [tmp_path, stl_path, glb_path]:
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
    url = f"{SUPABASE_URL}/storage/v1/object/public/{path}"
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
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {str(e)}"})
