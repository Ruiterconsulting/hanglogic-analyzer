from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import os
import traceback

from supabase import create_client, Client
from OCP.ShapeFix import ShapeFix_Wire


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="4.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Supabase setup
# =====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase")
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
        supabase = None
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing, uploads disabled")


def upload_unique(local_path: str, original_name: str) -> str | None:
    """
    Upload file to Supabase met unieke naam.
    Geeft public URL terug of None als supabase niet beschikbaar is.
    """
    if not supabase:
        print("⚠️ Supabase not configured, skipping upload for", original_name)
        return None

    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(original_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    url = storage.get_public_url(remote_path)
    print("🌍 Uploaded:", url)
    return url


# =====================================================
# Helpers
# =====================================================

def ensure_step(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(status_code=415, detail="Upload .STEP or .STP only")


def make_bbox_dims(bbox) -> dict:
    """
    Zelfde logica als eerder:
    sorteer de drie afmetingen, grootste = X, middelste = Y, kleinste = Z.
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
    Zodat Lovable / Postman ALTIJD nette JSON krijgt, ook bij fout.
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
# Geometry helpers: binnencontouren
# =====================================================

def repair_wire(wire):
    """
    Fix rommelige STEP-wires met ShapeFix_Wire.
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


def detect_inner_wires(shape):
    """
    Zoek ALLE gesloten binnencontouren op faces:

    - we checken niet meer op isPlane(), om niets te missen
    - grootste wire (op basis van area) = outer
    - alle andere gesloten wires = inner

    Retourneert: lijst van (face_index, face, [inner_wires])
    """
    result = []
    face_count = 0

    for f_idx, face in enumerate(shape.Faces()):
        face_count += 1
        wires = list(face.Wires())
        if len(wires) <= 1:
            continue

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

        inner = []
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

            inner.append(w)

        if inner:
            result.append((f_idx, face, inner))

    print(f"🔍 Faces checked: {face_count}, faces with inner wires: {len(result)}")
    return result


def build_fill_from_wire(face, wire, thickness, f_idx, w_idx):
    """
    Maak een 'plug' die het gat vult:

    - wire repareren
    - plane = exacte face-plane
    - extrude in normaalrichting over (minstens) plaatdikte
    """
    wire = repair_wire(wire)

    try:
        if wire.isNull() or wire.Length() < 0.1:
            print(f"⚠️ Skipping invalid/tiny wire {w_idx} on face {f_idx}")
            return None
    except Exception:
        return None

    try:
        # Plane van face (werkt voor vlakke faces; voor andere kan het falen)
        plane = face.toPln()
    except Exception as e:
        print(f"⚠️ face.toPln failed on face {f_idx}: {e}")
        return None

    try:
        wp = cq.Workplane(plane).add(wire)
        depth = thickness if thickness > 0 else 1.0

        # Extrude één kant op (positieve normaal)
        solid = wp.toPending().extrude(depth)
        return solid
    except Exception as e:
        print(f"⚠️ Fill extrude failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# Routes
# =====================================================

@app.get("/")
def root():
    return {"message": "HangLogic analyzer live ✅ (v4.3.0)"}


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None
    filename = file.filename

    try:
        ensure_step(filename)

        # 1️⃣ STEP file tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # 2️⃣ STEP inladen
        model = cq.importers.importStep(tmp_step)
        shape = model.val()
        print(f"🧠 Analyzing: {filename}")

        if shape is None or shape.isNull():
            raise RuntimeError("Imported STEP shape is null")

        # 3️⃣ Bounding box & dikte
        bbox = shape.BoundingBox()
        dims = make_bbox_dims(bbox)

        raw_thickness = min(float(bbox.xlen), float(bbox.ylen), float(bbox.zlen))
        thickness = raw_thickness if raw_thickness > 0 else 1.0

        print(f"📏 BoundingBox raw: ({bbox.xlen}, {bbox.ylen}, {bbox.zlen}), "
              f"sorted dims = {dims}, thickness ≈ {thickness}")

        # 4️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 5️⃣ Binnencontouren detecteren en vullen
        inner_faces = detect_inner_wires(shape)
        fills = []

        for f_idx, face, wires in inner_faces:
            for w_idx, wire in enumerate(wires):
                solid = build_fill_from_wire(face, wire, thickness, f_idx, w_idx)
                if solid is not None:
                    fills.append(solid)

        holes_detected = len(fills)
        print(f"🟢 Built {holes_detected} fill solids")

        # 6️⃣ Assembly bouwen met kleuren
        base_color = Color(0.6, 0.8, 1.0, 1.0)   # lichtblauw
        fill_color = Color(0.0, 1.0, 0.0, 1.0)   # felgroen

        asm = cq.Assembly()
        asm.add(shape, name="base", color=base_color)

        for idx, solid in enumerate(fills):
            try:
                asm.add(solid, name=f"fill_{idx}", color=fill_color)
            except Exception as e:
                print(f"⚠️ Assembly add failed for fill_{idx}: {e}")

        # 7️⃣ GLB export – nieuwe API + fallback
        tmp_glb = tmp_step.replace(".step", ".glb")

        try:
            asm.export(tmp_glb, "GLTF")
            print("✅ GLB exported via Assembly.export")
        except AttributeError:
            asm.save(tmp_glb, exportType="GLTF")
            print("✅ GLB exported via Assembly.save (fallback)")
        except Exception as e:
            print("⚠️ GLB export failed:", e)
            tmp_glb = None

        # 8️⃣ STL export (basisdeel)
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")
        print("✅ STL exported")

        # 9️⃣ Upload naar Supabase
        stl_url = upload_unique(tmp_stl, filename.replace(".step", ".stl")) if tmp_stl else None
        glb_url = upload_unique(tmp_glb, filename.replace(".step", ".glb")) if tmp_glb else None

        # 🔟 Wegschrijven in DB
        if supabase:
            try:
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
            except Exception as e:
                print("⚠️ Could not insert into analyzed_parts:", e)

        # 1️⃣1️⃣ Response in bekend formaat
        payload = {
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 is not None else None,
            "filename": filename,
            "holesDetected": holes_detected,
            "holes": [],
            "modelURL": stl_url,
            "modelURL_GLB": glb_url,
        }
        return JSONResponse(content=payload)

    except Exception as e:
        tb = traceback.format_exc(limit=12)
        print("❌ Analysis failed:", e)
        return JSONResponse(content=safe_error_payload(filename, str(e), tb))

    finally:
        # Cleanup
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
