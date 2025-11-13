from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import traceback
import os
import numpy as np

from supabase import create_client, Client

from OCP.ShapeFix import ShapeFix_Wire
from OCP.BRepAdaptor import BRepAdaptor_Surface

import trimesh
import json
from io import StringIO


# =====================================================
# FastAPI Setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Supabase Setup
# =====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    print("⚠️ Missing Supabase credentials")


def upload_unique(local_path: str, original_name: str) -> str:
    """Upload met unieke naam, nooit overschrijven."""
    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(original_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# =====================================================
# Reliable GLB Exporter (kleur + single-file)
# =====================================================

def assembly_to_glb(assembly: cq.Assembly, out_path: str):
    """
    - Converteert CadQuery Assembly (met kleuren) naar één .glb bestand.
    - Compatibel met alle CadQuery versies (geen Assembly.export nodig).
    - Gebruikt ThreeJS JSON + Trimesh → GLB converter.
    """

    meshes = []

    def walk(node: cq.Assembly, parent_transform=np.eye(4)):
        # Lokale matrix naar numpy
        local = np.array(node.loc.toMatrix())
        transform = parent_transform @ local

        # Als node een shape heeft → mesh maken
        if node.shape:
            threejs_str = cq.exporters.toString(node.shape, "TJS")

            # Trimesh lastig → direct JSON lezen
            data = json.load(StringIO(threejs_str))
            tm = trimesh.load(data, file_type='dict')
            tm.apply_transform(transform)

            # Kleur
            if node.color:
                r, g, b, a = node.color.wrapped.GetRGB()
                rgba = [int(r*255), int(g*255), int(b*255), int(a*255)]
                tm.visual.vertex_colors = np.tile(rgba, (len(tm.vertices), 1))

            meshes.append(tm)

        for child in node.children:
            walk(child, transform)

    walk(assembly)

    scene = trimesh.Scene(meshes)
    scene.export(out_path)  # ext bepaalt GLB export


# =====================================================
# Wire Fix Helper
# =====================================================

def repair_wire(wire):
    """Repareert slechte STEP wires (open, self-intersecting)."""
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
# Detect Inner Closed Contours
# =====================================================

def detect_inner_contours(shape):
    """
    Voor elke face:
    - Vind ALLE wires
    - Outer wire = grootste area
    - Inner wires = rest
    """
    results = []

    for f_idx, face in enumerate(shape.Faces()):
        wires = list(face.Wires())
        if len(wires) <= 1:
            continue

        # bepaal area per wire
        areas = []
        for w in wires:
            try:
                areas.append(abs(w.Area()))
            except:
                areas.append(0)

        # grootste = outer
        max_idx = areas.index(max(areas))
        inner = [wires[i] for i in range(len(wires)) if i != max_idx]

        if inner:
            results.append((f_idx, face, inner))

    return results


# =====================================================
# Build Fill Solid
# =====================================================

def build_fill(face, wire, thickness, f_idx, w_idx):
    """Extrude het wire precies langs de face normaal."""
    wire = repair_wire(wire)

    if wire.isNull() or wire.Length() < 0.1:
        print(f"⚠️ Wire {w_idx} on face {f_idx} is invalid.")
        return None

    # center
    center = wire.Center()

    # surface normal
    surf = BRepAdaptor_Surface(face.wrapped)
    try:
        u, v = surf.ValueOfUV(center.x, center.y, center.z)
    except:
        return None

    n = surf.Normal(u, v)
    normal = cq.Vector(n.X(), n.Y(), n.Z()).normalized()

    # plane
    plane = cq.Plane(
        (center.x, center.y, center.z),
        (normal.x, normal.y, normal.z)
    )

    try:
        wp = cq.Workplane(plane).add(wire)
        a = wp.toPending().extrude(thickness / 2)
        b = wp.toPending().extrude(-thickness / 2)
        return a.union(b)
    except Exception as e:
        print(f"⚠️ Fill failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# API Root
# =====================================================

@app.get("/")
def root():
    return {"status": "HangLogic Analyzer v4.0.0 running"}


# =====================================================
# Analyze Endpoint
# =====================================================

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None

    try:
        if not file.filename.lower().endswith((".step", ".stp")):
            raise HTTPException(415, "Upload alleen .STEP / .STP")

        # Save temp STEP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # Load
        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        # Bounding box
        bb = shape.BoundingBox()
        dims = {
            "x": round(bb.xlen, 3),
            "y": round(bb.ylen, 3),
            "z": round(bb.zlen, 3),
        }

        thickness = dims["z"]

        # Detect holes
        inner_data = detect_inner_contours(shape)
        fills = []

        for f_idx, face, wires in inner_data:
            for w_idx, wire in enumerate(wires):
                solid = build_fill(face, wire, thickness, f_idx, w_idx)
                if solid:
                    fills.append(solid)

        # Assembly met kleur
        asm = cq.Assembly()
        asm.add(shape, name="base", color=Color(0.6, 0.8, 1.0))

        for idx, solid in enumerate(fills):
            asm.add(solid, name=f"fill_{idx}", color=Color(0, 1, 0))

        # Export STL
        tmp_stl = tmp_step.replace(".step", "_base.stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Export GLB (perfect)
        tmp_glb = tmp_step.replace(".step", ".glb")
        assembly_to_glb(asm, tmp_glb)

        # Upload
        stl_url = upload_unique(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_unique(tmp_glb, file.filename.replace(".step", ".glb"))

        return {
            "status": "success",
            "units": "mm",
            "filename": file.filename,
            "boundingBoxMM": {"X": dims["x"], "Y": dims["y"], "Z": dims["z"]},
            "holesDetected": len(inner_data),
            "modelURL": stl_url,
            "modelURL_GLB": glb_url,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "trace": traceback.format_exc()}
        )

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
