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
import datetime

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
    print("⚠️ Missing SUPABASE credentials.")

# -------------------------------
# 🕳️ Hole Detection
# -------------------------------
def detect_holes(shape, d_min=1.0, d_max=1000.0):
    holes = []
    for face in shape.Faces():
        for edge in face.Edges():
            if edge.geomType() != "CIRCLE":
                continue
            circ = edge._geomAdaptor().Circle()
            r = circ.Radius()
            if d_min <= r * 2 <= d_max:
                loc = circ.Location()
                holes.append((float(loc.X()), float(loc.Y()), float(loc.Z()), float(r * 2)))
    print(f"🕳️ Found {len(holes)} circular holes.")
    return holes

# -------------------------------
# 🟦🟩 Model coloring
# -------------------------------
def colorize_shape(shape: cq.Shape):
    """
    Maakt een samengestelde Assembly:
      - Basismodel = blauw (#0A0F4B)
      - Binnencontouren = groen (#09D34B)
    """
    asm = cq.Assembly(name="colored_model")
    asm.add(shape, color=cq.Color(0.039, 0.059, 0.294))  # blauw hoofddeel

    try:
        for face in shape.Faces():
            for wire in face.Wires():
                if len(wire.Edges()) >= 3 and not wire.isOuterWire():
                    # Binnencontour
                    hole_face = cq.Face.makeFromWires(wire)
                    asm.add(hole_face, color=cq.Color(0.035, 0.827, 0.294))  # groen vlak
    except Exception as e:
        print("⚠️ Colorize failed:", e)

    return asm

# -------------------------------
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase not initialized.")
    bucket = "cad-models"
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{timestamp}{ext}"
    remote_path = f"analyzed/{unique_name}"
    storage = supabase.storage.from_(bucket)
    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)
    public_url = storage.get_public_url(remote_path)
    print(f"🌍 Uploaded: {public_url}")
    return public_url

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        if not file.filename.lower().endswith((".step", ".stp")):
            raise HTTPException(status_code=415, detail="Unsupported file type")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"🧠 Analyzing: {file.filename}")
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        bbox = shape.BoundingBox()
        dims = {"X": round(bbox.xlen, 3), "Y": round(bbox.ylen, 3), "Z": round(bbox.zlen, 3)}
        volume = round(shape.Volume(), 3)
        holes = detect_holes(shape)

        # 🟦 maak kleurmodel
        colored = colorize_shape(shape)

        # 📤 export STL
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")
        stl_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # 📤 export GLB (met kleur)
        glb_path = tmp_path.replace(".step", ".glb")
        cq.exporters.export(colored, glb_path, "GLTF")
        glb_url = upload_to_supabase(glb_path, file.filename.replace(".step", ".glb"))

        # 🧾 save + response
        if supabase:
            try:
                supabase.table("analyzed_parts").insert({
                    "filename": file.filename,
                    "bounding_box_x": dims["X"],
                    "bounding_box_y": dims["Y"],
                    "bounding_box_z": dims["Z"],
                    "volume_mm3": volume,
                    "holes_detected": len(holes),
                    "holes_data": holes,
                    "model_url_stl": stl_url,
                    "model_url_glb": glb_url,
                    "created_at": "now()",
                }).execute()
            except Exception as e:
                print("⚠️ DB insert skipped:", e)

        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": volume,
            "holesDetected": len(holes),
            "holes": holes,
            "modelURL_stl": stl_url,
            "modelURL_glb": glb_url
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        print("❌ Error:", e)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        if tmp_path:
            for ext in [".step", ".stl", ".glb"]:
                p = tmp_path.replace(".step", ext)
                if os.path.exists(p):
                    os.remove(p)

# -------------------------------
# 🌐 Proxy for CORS
# -------------------------------
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    url = f"https://sywnjytfygvotskufvzs.supabase.co/storage/v1/object/public/{path}"
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
