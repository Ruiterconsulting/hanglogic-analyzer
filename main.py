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

app = FastAPI(title="HangLogic Analyzer API", version="3.9.0")

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


def upload_unique(local_path: str, original_name: str) -> str:
    """
    Upload file to Supabase with a guaranteed unique name.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized")

    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(original_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# =====================================================
# Helpers
# =====================================================

def ensure_step(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(status_code=415, detail="Upload .STEP or .STP only")


def make_bbox_dims(bbox) -> dict:
    """
    Zelfde stijl als je oude implementatie:
    sorteer de drie maten, grootste = X, middel = Y, kleinste = Z.
    """
    raw = [float(bbox.xlen), float(bbox.ylen), float(bbox.zlen)]
    sorted_dims = sorted(raw, reverse=True)
    return {
        "X": round(sorted_dims[0], 3),
        "Y": round(sorted_dims[1], 3),
        "Z": round(sorted_dims[2], 3),
    }


def safe_error_payload(filename: str | None, message: str, trace: str):
    """
    Zorgt dat Lovable nooit meer crasht op missing fields.
    """
    return {
        "status": "error",
        "units": "mm",
        "boundingBoxMM": {"X": 0.0, "Y": 0.0, "Z": 0.0},
        "volumeMM3": None,
        "filename": filename,
        "holesDetected": 0,
        "holes": [],
        "modelURL": None,
        "modelURL_GLB": None,
        "message": message,
        "trace": trace,
    }


# =====================================================
# Geometry: binnencontouren
# =====================================================

def repair_wire(wire: cq.Wire) -> cq.Wire:
    """
    Gebruik ShapeFix_Wire om rommelige wires te sluiten/repareren.
    """
    try:
        fixer = ShapeFix_Wire(wire.wrapped)
        fixer.ClosedWireMode()
        fixer.FixReorder()
        fixer.FixConnected()
        fixer.FixSelfIntersection()
        fixer.SetPrecision(1e-6)
        fixed_wrapped = fixer.Wire()
        fixed = cq.Wire(fixed_wrapped)
        if fixed and (not fixed.isNull()):
            return fixed
    except Exception as e:
        print("⚠️ ShapeFix_Wire failed:", e)

    return wire


def detect_inner_wires_planar(shape: cq.Shape):
    """
    Detect gesloten binnencontouren op vlakke faces:

    - Neem alle wires op een face
    - Grootste area = buitencontour
    - Alle kleinere gesloten wires = binnencontouren (gaten)

    Geeft lijst terug: (face_index, face, [inner_wires])
    """
    result = []

    for f_idx, face in enumerate(shape.Faces()):
        try:
            if not face.isPlane():
                continue
        except Exception:
            continue

        wires = list(face.Wires())
        if len(wires) <= 1:
            continue

        # area per wire
        areas = []
        for w in wires:
            try:
                fw = cq.Face.makeFromWires(w)
                areas.append(abs(float(fw.Area())))
            except Exception:
                areas.append(0.0)

        if not areas:
            continue

        outer_idx = max(range(len(areas)), key=lambda i: areas[i])

        inners: list[cq.Wire] = []
        for i, w in enumerate(wires):
            if i == outer_idx:
                continue
            try:
                if w.isNull():
                    continue
                if hasattr(w, "isClosed") and not w.isClosed():
                    continue
                if len(w.Edges()) == 0:
                    continue
            except Exception:
                continue
            inners.append(w)

        if inners:
            result.append((f_idx, face, inners))

    print(f"🟢 Planar faces with inner wires: {len(result)}")
    return result


def build_fill_from_wire(face: cq.Face,
                         wire: cq.Wire,
                         thickness: float,
                         f_idx: int,
                         w_idx: int):
    """
    Maakt een 'plug' door de plaat heen:

    - wire repareren
    - local plane op center + normal van de face
    - extrude ± thickness/2
    """
    wire = repair_wire(wire)

    try:
        if wire.isNull() or wire.Length() < 0.1:
            print(f"⚠️ Skipping invalid/tiny wire {w_idx} on face {f_idx}")
            return None
    except Exception:
        return None

    try:
        normal = face.normalAt()
    except Exception:
        normal = cq.Vector(0, 0, 1)

    try:
        center = wire.Center()
    except Exception:
        center = face.Center()

    plane = cq.Plane((center.x, center.y, center.z),
                     (normal.x, normal.y, normal.z))

    try:
        wp = cq.Workplane(plane).add(wire)
        depth = thickness if thickness > 0 else 1.0

        pos = wp.toPending().extrude(depth / 2.0)
        neg = wp.toPending().extrude(-depth / 2.0)
        solid = pos.union(neg)
        return solid
    except Exception as e:
        print(f"⚠️ Fill failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# Routes
# =====================================================

@app.get("/")
def root():
    return {"message": "HangLogic analyzer live ✅ (v3.9.0)"}


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None
    filename = file.filename

    try:
        ensure_step(filename)

        # 1️⃣ STEP tijdelijk wegschrijven
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # 2️⃣ STEP inladen
        model = cq.importers.importStep(tmp_step)
        shape = model.val()
        print(f"🧠 Analyzing: {filename}")

        # 3️⃣ Bounding box + dikte
        bbox = shape.BoundingBox()
        dims = make_bbox_dims(bbox)
        thickness = min(dims["X"], dims["Y"], dims["Z"])
        if thickness <= 0:
            thickness = 1.0

        # 4️⃣ Volume (optioneel)
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 5️⃣ Binnencontouren zoeken en vullen
        inner_data = detect_inner_wires_planar(shape)
        fills: list[cq.Solid] = []

        for f_idx, face, wires in inner_data:
            for w_idx, wire in enumerate(wires):
                solid = build_fill_from_wire(face, wire, thickness, f_idx, w_idx)
                if solid is not None:
                    fills.append(solid)

        holes_detected = len(fills)

        # 6️⃣ Assembly met kleuren → GLTF/GLB
        base_color = Color(0.6, 0.8, 1.0)   # lichtblauw
        fill_color = Color(0.0, 1.0, 0.0)   # groen

        asm = cq.Assembly()
        asm.add(shape, name="base", color=base_color)

        for idx, solid in enumerate(fills):
            try:
                asm.add(solid, name=f"fill_{idx}", color=fill_color)
            except Exception as e:
                print(f"⚠️ Assembly add failed for fill_{idx}: {e}")

        # Belangrijk: CadQuery 2.4 ondersteunt save(..., exportType="GLTF")
        # en beslist aan de extensie of het .gltf of .glb wordt.
        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # 7️⃣ STL export (alleen basisvorm, zonder fills)
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # 8️⃣ Upload naar Supabase
        stl_url = upload_unique(tmp_stl, filename.replace(".step", ".stl"))
        glb_url = upload_unique(tmp_glb, filename.replace(".step", ".glb"))

        # 9️⃣ Wegschrijven in DB
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": filename,
                "dimensions": dims,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "holes_detected": holes_detected,
                "units": "mm",
                "model_url": stl_url,
                "model_url_glb": glb_url,
                "created_at": "now()"
            }).execute()

        # 🔟 Response zoals Lovable het verwacht
        payload = {
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 is not None else None,
            "filename": filename,
            "holesDetected": holes_detected,
            "holes": [],              # eventueel later nog data invullen
            "modelURL": stl_url,
            "modelURL_GLB": glb_url,
        }
        return JSONResponse(content=payload)

    except Exception as e:
        tb = traceback.format_exc(limit=6)
        print("❌ Analysis failed:", e)
        return JSONResponse(content=safe_error_payload(filename, str(e), tb))

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
