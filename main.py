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
import trimesh
import uuid
from datetime import datetime

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
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# -------------------------------
# 🕳️ Oude strikte ronde gaten (ter info)
# -------------------------------
def detect_analytic_holes_strict(shape,
                                 d_min=2.0, d_max=30.0,
                                 xy_tol=0.2, d_tol=0.2,
                                 z_pair_min=1.0, z_pair_max=40.0,
                                 full_circle_tol_ratio=0.02):
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
                if L <= 0:
                    continue
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
            if (abs(g["x"] - cx) <= xy_tol and abs(g["y"] - cy) <= xy_tol and abs(g["d"] - d) <= d_tol):
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
# 🟢 Algemene binnencontour-detectie
# -------------------------------
def detect_all_inner_contours(shape):
    """
    Detecteert ALLE binnencontouren (rond, vierkant, sleuf, willekeurig)
    Retourneert lijst met {x,y,z,diameter,dz}.
    """
    holes = []
    for face in shape.Faces():
        try:
            outer = face.outerWire()
        except Exception:
            outer = None

        for wire in face.Wires():
            if outer and wire.isSame(outer):
                continue
            bb = wire.BoundingBox()
            cx = (bb.xmin + bb.xmax) / 2
            cy = (bb.ymin + bb.ymax) / 2
            cz = (bb.zmin + bb.zmax) / 2
            dz = bb.zlen if bb.zlen > 0 else 1.0
            diameter_eq = max(bb.xlen, bb.ylen)
            holes.append({
                "x": round(cx, 3),
                "y": round(cy, 3),
                "z": round(cz, 3),
                "diameter": round(diameter_eq, 3),
                "dz": round(dz, 3)
            })
    print(f"🟢 Detected {len(holes)} inner contours (any shape).")
    return holes

# -------------------------------
# 📦 Upload helper (no overwrite)
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """
    Uploadt bestand naar Supabase met unieke naam (geen overschrijving).
    """
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

        bbox = shape.BoundingBox()
        bbox_dims = {"x": round(bbox.xlen, 3), "y": round(bbox.ylen, 3), "z": round(bbox.zlen, 3)}
        volume_mm3 = float(shape.Volume()) if shape.Volume() else None

        strict_holes = detect_analytic_holes_strict(shape)
        all_inner = detect_all_inner_contours(shape)

        base_name, _ = os.path.splitext(file.filename)
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")
        stl_url = upload_to_supabase(stl_path, f"{base_name}.stl")

        green = [0, 255, 0, 255]
        light_blue = [150, 200, 255, 255]
        mesh = trimesh.load_mesh(stl_path)
        mesh.visual.vertex_colors = [light_blue] * len(mesh.vertices)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="body")

        for idx, hole in enumerate(all_inner):
    dx = float(hole["diameter"])  # breedte (X)
    dy = float(hole["diameter"])  # hoogte (Y)
    dz = float(hole["dz"]) or 1.0  # dikte
    # Maak een groen blok dat het gat volledig vult
    box = trimesh.creation.box(extents=[dx, dy, dz])
    box.visual.vertex_colors = [green] * len(box.vertices)
    box.apply_translation((hole["x"], hole["y"], hole["z"]))
    scene.add_geometry(box, node_name=f"hole_{idx}")

        glb_bytes = scene.export(file_type="glb")
        glb_path = tmp_path.replace(".step", ".glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        glb_url = upload_to_supabase(glb_path, f"{base_name}.glb")

        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": bbox_dims,
                "holes_detected": len(strict_holes),
                "holes_data": strict_holes,
                "inner_contours": all_inner,
                "units": "mm",
                "model_url": stl_url,
                "model_url_glb": glb_url
            }).execute()

        return JSONResponse(content={
            "status": "success",
            "boundingBoxMM": bbox_dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "holesDetected": len(strict_holes),
            "innerContoursDetected": len(all_inner),
            "filename": file.filename,
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
