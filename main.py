from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import os
import math
import traceback

from supabase import create_client, Client
from OCP.ShapeFix import ShapeFix_Wire


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="3.6.0")

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
    if not supabase:
        raise RuntimeError("Supabase not initialized")

    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# =====================================================
# JSON helpers — belangrijkste fix voor Lovable
# =====================================================

def safe_response_success(filename, dims, stl_url, glb_url, filled):
    """Altijd hoofdletters X/Y/Z om Lovable crashes te voorkomen."""
    return {
        "status": "success",
        "filename": filename,
        "dimensions": {
            "X": dims["X"],
            "Y": dims["Y"],
            "Z": dims["Z"],
        },
        "filledContours": filled,
        "modelURL": stl_url,
        "modelURL_GLB": glb_url,
    }


def safe_response_error(message, trace=""):
    """Ook bij errors nooit undefined dimensions → voorkomt frontend crash."""
    return {
        "status": "error",
        "filename": None,
        "dimensions": {"X": 0, "Y": 0, "Z": 0},
        "filledContours": 0,
        "modelURL": None,
        "modelURL_GLB": None,
        "message": message,
        "trace": trace,
    }


# =====================================================
# Wire Repair
# =====================================================

def repair_wire(wire):
    """Hard-fixes STEP wires."""
    try:
        fixer = ShapeFix_Wire(wire)
        fixer.ClosedWireMode()
        fixer.FixReorder()
        fixer.FixConnected()
        fixer.FixSelfIntersection()
        fixer.SetPrecision(1e-6)
        fixed = fixer.Wire()
        return fixed if not fixed.IsNull() else wire
    except:
        return wire


# =====================================================
# Detect **gesloten** binnencontouren (alleen vlakke faces)
# =====================================================

def detect_inner_wires(shape):
    """
    Detecteert A-type gaten:
    - alleen vlakke faces
    - grootste wire = outer
    - smallere wires = binnencontouren (gaten)
    """
    cleaned = []

    for f_idx, face in enumerate(shape.Faces()):
        try:
            if not face.isPlane():
                continue
        except:
            continue

        wires = list(face.Wires())
        if len(wires) <= 1:
            continue

        # Compute wire area to detect outer
        areas = []
        for w in wires:
            try:
                fw = cq.Face.makeFromWires(w)
                areas.append(abs(fw.Area()))
            except:
                areas.append(0)

        outer_index = max(range(len(areas)), key=lambda i: areas[i])

        inner = []
        for i, w in enumerate(wires):
            if i == outer_index:
                continue
            try:
                if w.isNull():
                    continue
                if hasattr(w, "isClosed") and not w.isClosed():
                    continue
                if len(w.Edges()) == 0:
                    continue
            except:
                continue
            inner.append(w)

        if inner:
            cleaned.append((f_idx, face, inner))

    print(f"🟢 Detected planar inner contours on {len(cleaned)} faces")
    return cleaned


# =====================================================
# Fill a single closed contour
# =====================================================

def build_solid_from_wire(face, wire, thickness, f_idx, w_idx):
    """Extrudes inner contour exactly in both directions."""
    wire = repair_wire(wire)

    try:
        if wire.isNull() or wire.Length() < 0.1:
            return None
    except:
        return None

    try:
        normal = face.normalAt()
    except:
        normal = cq.Vector(0, 0, 1)

    try:
        center = wire.Center()
    except:
        center = face.Center()

    plane = cq.Plane((center.x, center.y, center.z),
                     (normal.x, normal.y, normal.z))

    try:
        wp = cq.Workplane(plane).add(wire)

        pos = wp.toPending().extrude(thickness / 2)
        neg = wp.toPending().extrude(-thickness / 2)

        return pos.union(neg)

    except Exception as e:
        print(f"⚠️ Fill failed on face {f_idx} wire {w_idx}: {e}")
        return None


# =====================================================
# API Root
# =====================================================

@app.get("/")
def root():
    return {"status": "HangLogic v3.6.0 running"}


# =====================================================
# Main Analyzer
# =====================================================

def ensure_step(filename):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(415, "Upload .STEP or .STP only")


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None

    try:
        ensure_step(file.filename)

        # Save temp STEP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # Load STEP
        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        # Dimensions
        bbox = shape.BoundingBox()
        dims = {
            "X": round(bbox.xlen, 3),
            "Y": round(bbox.ylen, 3),
            "Z": round(bbox.zlen, 3),
        }

        thickness = min(dims.values())
        if thickness <= 0:
            thickness = 1.0

        # Detect + fill holes
        inner_wires = detect_inner_wires(shape)
        filled_solids = []

        for f_idx, face, wires in inner_wires:
            for w_idx, wire in enumerate(wires):
                solid = build_solid_from_wire(face, wire, thickness, f_idx, w_idx)
                if solid:
                    filled_solids.append(solid)

        # Build Assembly
        asm = cq.Assembly()
        asm.add(shape, name="base", color=Color(0.6, 0.8, 1.0))

        for i, solid in enumerate(filled_solids):
            asm.add(solid, name=f"fill_{i}", color=Color(0, 1, 0))

        # Export GLB
        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # Export STL
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Upload to Supabase
        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

        # Store DB record
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "holes_detected": len(filled_solids),
                "units": "mm",
                "model_url": stl_url,
                "model_url_glb": glb_url,
                "created_at": "now()"
            }).execute()

        return safe_response_success(file.filename, dims, stl_url, glb_url, len(filled_solids))

    except Exception as e:
        return safe_response_error(str(e), traceback.format_exc())

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
