from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import os
import traceback

from supabase import create_client, Client

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.ShapeFix import ShapeFix_Wire


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="2.3.1")

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
# Utility: Wire Repair
# =====================================================

def repair_wire(wire):
    """Forcefully repair and close problematic STEP wires."""
    try:
        fixer = ShapeFix_Wire(wire)
        fixer.ClosedWireMode()
        fixer.FixReorder()
        fixer.FixConnected()
        fixer.FixSelfIntersection()
        fixer.SetPrecision(1e-6)
        return fixer.Wire()
    except:
        return wire


# =====================================================
# Utility: Detect inner wires
# =====================================================

def detect_inner_wires(shape):
    data = []
    for f_idx, face in enumerate(shape.Faces()):
        wires = list(face.Wires())

        if len(wires) <= 1:
            continue

        inner = wires[1:]
        data.append((f_idx, face, inner))
    return data


# =====================================================
# Utility: Fill hole → perfect extrusion
# =====================================================

def build_solid_from_wire(face, wire, thickness, f_idx, w_idx):
    """Extrudes a repaired wire EXACT along the true face normal."""

    # 1️⃣ Repair the wire
    wire = repair_wire(wire)

    # Skip empty wires after fix
    if wire.isNull() or wire.Length() < 0.1:
        print(f"⚠️ Repaired wire on face {f_idx} still invalid")
        return None

    # 2️⃣ Real surface
    surf = BRepAdaptor_Surface(face.wrapped)

    # 3️⃣ Real (u,v) location
    center = wire.Center()
    try:
        u, v = face._geomAdaptor().ValueOfUV(center.x, center.y, center.z)
    except:
        try:
            # fallback: find closest UV
            u, v = surf.ValueOfUV(center.x, center.y, center.z)
        except:
            return None

    # 4️⃣ Real surface normal
    n = surf.Normal(u, v)
    normal = cq.Vector(n.X(), n.Y(), n.Z()).normalized()

    # 5️⃣ Construct precise plane in face orientation
    plane = cq.Plane((center.x, center.y, center.z), (normal.x, normal.y, normal.z))

    try:
        wp = cq.Workplane(plane).add(wire)

        pos = wp.toPending().extrude(thickness / 2)
        neg = wp.toPending().extrude(-thickness / 2)

        return pos.union(neg)

    except Exception as e:
        print(f"⚠️ fill failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# API Root
# =====================================================

@app.get("/")
def root():
    return {"status": "HangLogic v2.3.1 running"}


# =====================================================
# Main analyzer
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

        # Save STEP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        # Dimensions
        bbox = shape.BoundingBox()
        dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3),
        }
        thickness = dims["z"]

        # Detect holes/inner contours
        inner_data = detect_inner_wires(shape)

        filled_solids = []

        # Build solids for each inner contour
        for f_idx, face, wires in inner_data:
            for w_idx, wire in enumerate(wires):

                # Hard validation
                if wire.isNull() or wire.Length() < 0.1:
                    print(f"⚠️ Skipping null/tiny wire {w_idx}")
                    continue

                solid = build_solid_from_wire(face, wire, thickness, f_idx, w_idx)

                if solid:
                    filled_solids.append(solid)

        # =====================================================
        # Build GLB Assembly
        # =====================================================

        base_color = Color(0.6, 0.8, 1.0)
        fill_color = Color(0.0, 1.0, 0.0)

        asm = cq.Assembly()
        asm.add(shape, name="base", color=base_color)

        for idx, solid in enumerate(filled_solids):
            asm.add(solid, name=f"fill_{idx}", color=fill_color)

        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # STL
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Upload
        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

        # Save DB entry
        supabase.table("analyzed_parts").insert({
            "filename": file.filename,
            "dimensions": dims,
            "bounding_box_x": dims["x"],
            "bounding_box_y": dims["y"],
            "bounding_box_z": dims["z"],
            "holes_detected": len(inner_data),
            "units": "mm",
            "model_url": stl_url,
            "model_url_glb": glb_url,
            "created_at": "now()"
        }).execute()

        return {
            "status": "success",
            "filename": file.filename,
            "dimensions": dims,
            "filledContours": len(filled_solids),
            "modelURL": stl_url,
            "modelURL_GLB": glb_url,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
