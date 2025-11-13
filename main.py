from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import os
import traceback

from supabase import create_client, Client


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Supabase
# =====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    print("⚠️ Missing Supabase credentials")


def upload_new_file(local_path: str, remote_name: str) -> str:
    """Uploads file to Supabase with guaranteed unique filename."""
    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# =====================================================
# Utility: Always uppercase dimensions (Lovable safety)
# =====================================================

def force_dimensions_uppercase(dims):
    """Guarantees X/Y/Z always exist with uppercase keys."""
    return {
        "X": float(dims.get("X") or dims.get("x") or 0.0),
        "Y": float(dims.get("Y") or dims.get("y") or 0.0),
        "Z": float(dims.get("Z") or dims.get("z") or 0.0),
    }


# =====================================================
# Safe response wrappers (prevent Lovable crashes)
# =====================================================

def safe_response_success(filename, dims, stl_url, glb_url, filled):
    dims = force_dimensions_uppercase(dims)
    return {
        "status": "success",
        "filename": filename,
        "dimensions": dims,      # ALWAYS X/Y/Z
        "filledContours": filled,
        "modelURL": stl_url,     # STL (Lovable can ignore this)
        "modelURL_GLB": glb_url  # GLB (WITH COLORS)
    }


def safe_response_error(message, trace=""):
    """Lovable-compatible error. Never breaks UI."""
    return {
        "status": "error",
        "filename": None,
        "dimensions": {"X": 0, "Y": 0, "Z": 0},
        "filledContours": 0,
        "modelURL": None,
        "modelURL_GLB": None,
        "message": message,
        "trace": trace
    }


# =====================================================
# Root endpoint
# =====================================================

@app.get("/")
def root():
    return {"status": "HangLogic v3.0.0 running"}


# =====================================================
# Only allow STEP files
# =====================================================

def ensure_step(filename):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(415, "Upload .STEP or .STP only")


# =====================================================
# MAIN ANALYZER — stable version (no filling yet)
# =====================================================

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None

    try:
        ensure_step(file.filename)

        # Save input STEP file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # Load with CadQuery
        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        # Compute bounding box
        bbox = shape.BoundingBox()

        dims = force_dimensions_uppercase({
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3)
        })

        # =====================================================
        # Build GLB with color (STL has no color!)
        # =====================================================
        base_color = Color(0.6, 0.8, 1.0)  # light blue

        asm = cq.Assembly()
        asm.add(shape, name="base", color=base_color)

        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # Export STL (no color)
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Upload both
        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

        # Store in DB
        supabase.table("analyzed_parts").insert({
            "filename": file.filename,
            "dimensions": dims,
            "bounding_box_x": dims["X"],
            "bounding_box_y": dims["Y"],
            "bounding_box_z": dims["Z"],
            "holes_detected": 0,   # filling komt later weer terug
            "units": "mm",
            "model_url": stl_url,
            "model_url_glb": glb_url,
            "created_at": "now()"
        }).execute()

        # SUCCESS RESPONSE (Lovable safe)
        return safe_response_success(
            file.filename,
            dims,
            stl_url,
            glb_url,
            filled=0
        )

    except Exception as e:
        return safe_response_error(str(e), traceback.format_exc())

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
