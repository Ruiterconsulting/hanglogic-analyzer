from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import traceback
from supabase import create_client, Client
import httpx
import datetime

app = FastAPI(title="HangLogic Analyzer API", version="2.1.0")

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
        print("⚠️ Could not init Supabase:", e)
else:
    print("⚠️ Missing SUPABASE credentials.")

# -------------------------------
# 🟦🟩 Model coloring
# -------------------------------
def colorize_shape(shape: cq.Shape) -> cq.Assembly:
    """
    Returneert een Assembly:
      - blauw (#0A0F4B) voor het hoofddeel
      - groen (#09D34B) voor binnencontouren
    """
    asm = cq.Assembly(name="colored_model")
    # hoofddeel blauw
    asm.add(shape, color=cq.Color(0.039, 0.059, 0.294))  

    # binnencontouren groen
    try:
        for face in shape.Faces():
            for wire in face.Wires():
                if not wire.isOuterWire():
                    inner_face = cq.Face.makeFromWires(wire)
                    asm.add(inner_face, color=cq.Color(0.035, 0.827, 0.294))
    except Exception as e:
        print("⚠️ Colorization warning:", e)

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
    url = storage.get_public_url(remote_path)
    print(f"🌍 Uploaded: {url}")
    return url

# -------------------------------
# 🧮 Analyzer
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        if not file.filename.lower().endswith((".step", ".stp")):
            raise HTTPException(status_code=415, detail="Upload .STEP only")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        shape = cq.importers.importStep(tmp_path).val()
        print(f"🧠 Analyzing: {file.filename}")

        bbox = shape.BoundingBox()
        dims = {
            "X": round(bbox.xlen, 3),
            "Y": round(bbox.ylen, 3),
            "Z": round(bbox.zlen, 3)
        }
        volume = round(shape.Volume(), 3)

        # 🟦 maak gekleurde Assembly
        asm = colorize_shape(shape)
        assert isinstance(asm, cq.Assembly), "colorize_shape() must return Assembly"

        # Export STL (blauw/grijs model)
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, exportType="STL")
        stl_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # Export GLTF/GLB (met kleuren)
        glb_path = tmp_path.replace(".step", ".glb")
        cq.exporters.export(asm, glb_path, exportType="GLTF")
        glb_url = upload_to_supabase(glb_path, file.filename.replace(".step", ".glb"))

        # Opslaan
        if supabase:
            try:
                supabase.table("analyzed_parts").insert({
                    "filename": file.filename,
                    "bounding_box_x": dims["X"],
                    "bounding_box_y": dims["Y"],
                    "bounding_box_z": dims["Z"],
                    "volume_mm3": volume,
                    "model_url_stl": stl_url,
                    "model_url_glb": glb_url,
                    "created_at": "now()",
                }).execute()
            except Exception as e:
                print("⚠️ DB insert skipped:", e)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "boundingBoxMM": dims,
            "volumeMM3": volume,
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
# 🌐 Proxy
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
