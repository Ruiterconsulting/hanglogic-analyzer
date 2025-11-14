from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cadquery as cq
from cadquery import Color

import tempfile
import os
import traceback

from supabase import create_client, Client
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON


# =====================================================
# FastAPI setup
# =====================================================

app = FastAPI(title="HangLogic Analyzer API", version="5.1.0")

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
    Geeft public URL of None als Supabase niet beschikbaar is.
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
    Sorteer de drie afmetingen: grootste = X, middelste = Y, kleinste = Z.
    (Voor display; níet meer als dikte gebruiken.)
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
    Net JSON foutobject voor Postman / Lovable.
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


def point_inside_solid(shape, pt: cq.Vector, tol: float = 1e-5) -> bool:
    """
    Check of een punt binnen of op de solid ligt via BRepClass3d_SolidClassifier.
    (Momenteel alleen als reserve gebruikt.)
    """
    try:
        classifier = BRepClass3d_SolidClassifier(shape.wrapped)
        p = gp_Pnt(float(pt.x), float(pt.y), float(pt.z))
        classifier.Perform(p, tol)
        state = classifier.State()
        return state in (TopAbs_IN, TopAbs_ON)
    except Exception as e:
        print("⚠️ point_inside_solid failed:", e)
        return False


# =====================================================
# Geometry helpers
# =====================================================

def repair_wire(wire):
    """
    Voor nu: geen ShapeFix, de STEP-wires zijn al bruikbaar.
    Laat dit als passthrough om later eventueel nog te tunen.
    """
    return wire


def detect_inner_wires(shape):
    """
    Zoek ALLE gesloten binnencontouren op faces:

    - gebruik face.innerWires() als CadQuery dat ondersteunt
    - gebruik face.outerWire() om buitencontour expliciet uit te sluiten
    - fallback: grootste wire = outer, rest = inner

    Retourneert: lijst van (face_index, face, [inner_wires])
    """
    result = []
    faces_checked = 0

    for f_idx, face in enumerate(shape.Faces()):
        faces_checked += 1

        inners = []

        # 1️⃣ probeer de “officiële” innerWires + outerWire
        try:
            outer = face.outerWire() if hasattr(face, "outerWire") else None
            if hasattr(face, "innerWires"):
                raw_inners = list(face.innerWires())
                # expliciet beschermen tegen outer die in innerWires belandt
                for w in raw_inners:
                    if outer is not None:
                        try:
                            if w.wrapped.IsSame(outer.wrapped):
                                continue
                        except Exception:
                            pass
                    inners.append(w)
        except Exception as e:
            print(f"⚠️ face.innerWires()/outerWire() failed on face {f_idx}: {e}")
            inners = []

        # 2️⃣ fallback als er geen inners gevonden zijn
        if not inners:
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

            # grootste area is vrijwel zeker de buitencontour
            outer_idx = max(range(len(areas)), key=lambda i: areas[i])
            outer_area = areas[outer_idx]

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

                    # extra safeguard: draad met bijna dezelfde area als outer overslaan
                    if 0.95 * outer_area <= areas[i] <= 1.05 * outer_area:
                        continue
                except Exception:
                    continue
                inners.append(w)

        if inners:
            result.append((f_idx, face, inners))

    print(f"🔍 Faces checked: {faces_checked}, faces with inner wires: {len(result)}")
    return result


def build_fill_from_wire(face, wire, shape, max_depth, f_idx, w_idx):
    """
    Bouw een groen solid die het gat opvult:

    - wire (binnencontour) repareren
    - extrude een lange plug naar binnen (tegenover de face-normal)
    - intersect die plug met de originele solid:
      --> exact het volume binnen het materiaal blijft over,
          vlak met beide oppervlakken.
    """
    wire = repair_wire(wire)

    try:
        if wire.isNull() or wire.Length() < 0.1:
            print(f"⚠️ Skipping invalid/tiny wire {w_idx} on face {f_idx}")
            return None
    except Exception:
        return None

    try:
        wp_face = cq.Workplane(face).add(wire)
        n = cq.Workplane(face).plane.zDir.normalized()
    except Exception as e:
        print(f"⚠️ build_fill_from_wire: workplane/normal failed on face {f_idx}: {e}")
        return None

    # naar binnen = tegen de face-normal in
    direction = -1.0

    try:
        depth = max_depth * 2.0  # ruim genoeg om het hele onderdeel te doorsteken
        long_wp = wp_face.toPending().extrude(direction * depth)  # Workplane

        # expliciet met solids werken voor intersect
        long_solid = long_wp.val()
        wp_long = cq.Workplane("XY").newObject([long_solid])
        wp_shape = cq.Workplane("XY").newObject([shape])

        plug_wp = wp_long.intersect(wp_shape)
        plug = plug_wp.val()

        # ✅ safeguard: als plug bijna even groot is als het hele part → fout, overslaan
        try:
            part_vol = float(shape.Volume())
            plug_vol = float(plug.Volume())
            if part_vol > 0 and plug_vol > 0:
                ratio = plug_vol / part_vol
                if ratio > 0.8:
                    print(
                        f"⚠️ Plug volume ~{ratio:.2f} van part volume "
                        f"(face {f_idx}, wire {w_idx}), skipping (waarschijnlijk outer wire)."
                    )
                    return None
        except Exception as e:
            print("⚠️ Volume check failed:", e)

        return plug
    except Exception as e:
        print(f"⚠️ Fill extrude/intersect failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# Routes
# =====================================================

@app.get("/")
def root():
    return {"message": "HangLogic analyzer live ✅ (v5.1.0)"}


@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None
    filename = file.filename

    try:
        ensure_step(filename)

        # 1️⃣ STEP tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        # 2️⃣ STEP inladen
        model = cq.importers.importStep(tmp_step)
        shape = model.val()
        print(f"🧠 Analyzing: {filename}")

        if shape is None or shape.isNull():
            raise RuntimeError("Imported STEP shape is null")

        # 3️⃣ Bounding box
        bbox = shape.BoundingBox()
        dims = make_bbox_dims(bbox)
        max_dim = max(float(bbox.xlen), float(bbox.ylen), float(bbox.zlen))
        print(
            f"📏 BoundingBox raw: ({bbox.xlen}, {bbox.ylen}, {bbox.zlen}), "
            f"sorted dims = {dims}, max_dim ≈ {max_dim}"
        )

        # 4️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 5️⃣ Binnencontouren detecteren en fills bouwen
        inner_faces = detect_inner_wires(shape)
        fills = []

        # dedupe op center zodat we niet 2× exact dezelfde plug
        seen_centers = set()

        for f_idx, face, wires in inner_faces:
            for w_idx, wire in enumerate(wires):
                try:
                    c = wire.Center()
                    key = (round(c.x, 2), round(c.y, 2), round(c.z, 2))
                    if key in seen_centers:
                        continue
                    seen_centers.add(key)
                except Exception:
                    pass

                solid = build_fill_from_wire(face, wire, shape, max_dim, f_idx, w_idx)
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

        # 7️⃣ GLB export – probeer nieuwe API, anders fallback
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

        # 🔟 Wegschrijven in DB (best effort)
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
