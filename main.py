from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cadquery as cq
import tempfile
import os
import traceback
from supabase import create_client, Client
import trimesh
import numpy as np
from io import BytesIO

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="2.3.0")

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
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase.")
else:
    print("⚠️ Missing Supabase credentials.")

# -------------------------------
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)
    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)
    return storage.get_public_url(remote_path)

# -------------------------------
# 🕳️ Binnencontour detectie
# -------------------------------
def detect_inner_faces(shape):
    inner_faces = []
    for face in shape.Faces():
        try:
            normal = face.normalAt(0.5, 0.5)
            if normal.z < 0:  # heuristiek voor “binnen”
                inner_faces.append(face)
        except Exception:
            continue
    return inner_faces

# -------------------------------
# 🎨 Maak gekleurde Trimesh uit CQ-shape
# -------------------------------
def make_colored_mesh_from_shape(shape, color_hex="#0A0F4B"):
    """Exporteer CadQuery-shape naar STL, lees het met trimesh, en kleur."""
    stl_bytes = cq.exporters.toString(shape, "STL").encode("utf-8")
    mesh = trimesh.load(BytesIO(stl_bytes), file_type="stl")

    color_rgb = np.array([
        int(color_hex[1:3], 16),
        int(color_hex[3:5], 16),
        int(color_hex[5:7], 16),
        255
    ], dtype=np.uint8)

    mesh.visual.vertex_colors = np.tile(color_rgb, (len(mesh.vertices), 1))
    return mesh

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # 1️⃣ opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2️⃣ import
        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # 3️⃣ binnencontouren
        inner_faces = detect_inner_faces(shape)
        print(f"✅ Found {len(inner_faces)} inner faces")

        # 4️⃣ kleuren
        blue_mesh = make_colored_mesh_from_shape(shape, "#0A0F4B")
        green_meshes = []
        for f in inner_faces:
            try:
                m = make_colored_mesh_from_shape(f, "#09D34B")
                green_meshes.append(m)
            except Exception as e:
                print(f"⚠️ Failed face mesh: {e}")

        combined = trimesh.util.concatenate([blue_mesh] + green_meshes)

        # 5️⃣ export GLB
        glb_path = tmp_path.replace(".step", ".glb")
        combined.export(glb_path)
        glb_url = upload_to_supabase(glb_path, file.filename.replace(".step", ".glb"))
        print("🌍 GLB uploaded to:", glb_url)

        # 6️⃣ Supabase log
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "model_url_glb": glb_url
            }).execute()

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "model_url_glb": glb_url
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        print("❌ Error:", e)
        return JSONResponse(status_code=500, content={
            "error": f"Analysis failed: {e}",
            "trace": tb
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
