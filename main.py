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

app = FastAPI(title="HangLogic Analyzer API", version="3.5.0")

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
        fixed = fixer.Wire()
        return fixed if not fixed.IsNull() else wire
    except Exception as e:
        print("⚠️ ShapeFix_Wire failed:", e)
        return wire


# =====================================================
# Utility: Detect inner wires (gesloten binnencontouren op vlakke faces)
# =====================================================

def detect_inner_wires(shape):
    """
    Detect gesloten binnencontouren (alleen planar faces).
    Returns list of (face_index, cq.Face, [cq.Wire, ...])
    """
    data = []

    for f_idx, face in enumerate(shape.Faces()):
        try:
            if not face.isPlane():
                continue  # geen extrude op gebogen faces
        except Exception:
            continue

        wires = list(face.Wires())
        if not wires:
            continue

        try:
            outer = face.outerWire()
        except Exception:
            outer = wires[0]

        inner_wires = []
        for w in wires:
            try:
                if w.isNull():
                    continue
                # skip outer boundary
                if w.isSame(outer):
                    continue
                # alleen gesloten binnencontouren (optie A)
                if hasattr(w, "isClosed"):
                    closed = w.isClosed()
                else:
                    closed = True  # fallback
                if not closed:
                    continue
                edges = w.Edges()
                if len(edges) == 0:
                    continue
            except Exception:
                continue

            inner_wires.append(w)

        if inner_wires:
            data.append((f_idx, face, inner_wires))

    print(f"🟢 Planar inner wires found on {len(data)} faces")
    return data


# =====================================================
# Utility: Fill from wire (general inner contour)
# =====================================================

def build_solid_from_wire(face, wire, thickness, f_idx, w_idx):
    """
    Extrudes a repaired wire along the local face normal.
    Geschikt voor vlakke faces (plaat, flens, etc.).
    """
    wire = repair_wire(wire)

    try:
        if wire.isNull() or wire.Length() < 0.1:
            print(f"⚠️ Repaired wire on face {f_idx} still invalid")
            return None
    except Exception:
        return None

    # Normaal van de face
    try:
        normal = face.normalAt()
    except Exception:
        normal = cq.Vector(0, 0, 1)

    # Center van de wire
    try:
        center = wire.Center()
    except Exception:
        center = face.Center()

    plane = cq.Plane((center.x, center.y, center.z), (normal.x, normal.y, normal.z))

    try:
        wp = cq.Workplane(plane).add(wire)

        # Symmetrisch extruderen door plaatdikte
        depth = thickness
        if depth <= 0:
            return None

        pos = wp.toPending().extrude(depth / 2.0)
        neg = wp.toPending().extrude(-depth / 2.0)
        solid = pos.union(neg)
        return solid
    except Exception as e:
        print(f"⚠️ fill failed on face {f_idx}, wire {w_idx}: {e}")
        return None


# =====================================================
# Utility: Detect & fill circular through-holes (any face orientation)
# =====================================================

def detect_circular_hole_fills(shape, thickness_guess):
    """
    Detect analytic circular edges that form through-holes
    and return cq.Solid cylinders that fill these holes.
    """
    circles = []

    # 1️⃣ Verzamel alle volledige cirkel-edges
    for face in shape.Faces():
        for edge in face.Edges():
            if edge.geomType() != "CIRCLE":
                continue
            try:
                ga = edge._geomAdaptor()
                circ = ga.Circle()
                center = circ.Location()   # gp_Pnt
                axis = circ.Axis()         # gp_Ax1
                direction = axis.Direction()
                r = float(circ.Radius())
                if r <= 0:
                    continue

                # Check of edge ongeveer een volledige cirkel is
                L = float(edge.Length())
                full_L = 2.0 * math.pi * r
                if abs(L - full_L) > 0.02 * full_L:
                    continue

                C = cq.Vector(center.X(), center.Y(), center.Z())
                D = cq.Vector(direction.X(), direction.Y(), direction.Z())
                if D.Length == 0:
                    continue
                D = D.normalized()

                circles.append({"center": C, "dir": D, "r": r})
            except Exception as e:
                print("⚠️ circle extraction failed:", e)
                continue

    if not circles:
        return []

    used = set()
    fills = []

    def v_sub(a: cq.Vector, b: cq.Vector) -> cq.Vector:
        return cq.Vector(a.x - b.x, a.y - b.y, a.z - b.z)

    def v_scale(a: cq.Vector, s: float) -> cq.Vector:
        return cq.Vector(a.x * s, a.y * s, a.z * s)

    def v_len(a: cq.Vector) -> float:
        return math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)

    # Verwachte gaten-diepte ligt rond materiaal-dikte
    max_depth = thickness_guess * 3.0  # toleranties

    for i in range(len(circles)):
        if i in used:
            continue
        ci = circles[i]
        best_j = None
        best_depth = 0.0

        for j in range(len(circles)):
            if j == i or j in used:
                continue
            cj = circles[j]

            # As moet ongeveer parallel zijn
            dot_dir = abs(ci["dir"].dot(cj["dir"]))
            if dot_dir < 0.99:
                continue

            diff = v_sub(cj["center"], ci["center"])
            proj = diff.dot(ci["dir"])
            lateral_vec = v_sub(diff, v_scale(ci["dir"], proj))
            lateral = v_len(lateral_vec)

            # Als de centra te ver uit elkaar liggen in laterale richting: geen door-gat
            if lateral > ci["r"] * 0.25:
                continue

            depth = abs(proj)
            # Depth moet grofweg rond plaatdikte liggen (met marge)
            if depth < 0.2 or depth > max_depth:
                continue

            if depth > best_depth:
                best_depth = depth
                best_j = j

        if best_j is None:
            continue

        used.add(i)
        used.add(best_j)

        ci_center = circles[i]["center"]
        dir_vec = circles[i]["dir"]
        r = circles[i]["r"]

        cj_center = circles[best_j]["center"]
        diff = v_sub(cj_center, ci_center)
        proj = diff.dot(dir_vec)
        mid = cq.Vector(
            ci_center.x + 0.5 * proj * dir_vec.x,
            ci_center.y + 0.5 * proj * dir_vec.y,
            ci_center.z + 0.5 * proj * dir_vec.z,
        )

        try:
            plane = cq.Plane((mid.x, mid.y, mid.z), (dir_vec.x, dir_vec.y, dir_vec.z))
            wp = cq.Workplane(plane).circle(r)
            pos = wp.extrude(best_depth / 2.0)
            neg = wp.extrude(-best_depth / 2.0)
            solid = pos.union(neg)
            fills.append(solid)
        except Exception as e:
            print("⚠️ circular fill creation failed:", e)
            continue

    print(f"🟢 Circular fills created: {len(fills)}")
    return fills


# =====================================================
# API Root
# =====================================================

@app.get("/")
def root():
    return {"status": "HangLogic v3.5.0 running"}


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

        # Plaatdikte ≈ kleinste dimensie
        thickness_guess = min(dims["x"], dims["y"], dims["z"])
        if thickness_guess <= 0:
            thickness_guess = 1.0

        # -----------------------------------------
        # 1️⃣ Niet-ronde binnencontouren via planar wires
        # -----------------------------------------
        inner_data = detect_inner_wires(shape)
        filled_solids = []

        for f_idx, face, wires in inner_data:
            for w_idx, wire in enumerate(wires):
                solid = build_solid_from_wire(face, wire, thickness_guess, f_idx, w_idx)
                if solid is not None:
                    filled_solids.append(solid)

        # -----------------------------------------
        # 2️⃣ Ronde doorlopende gaten via circle-detectie
        # -----------------------------------------
        circular_fills = detect_circular_hole_fills(shape, thickness_guess)
        filled_solids.extend(circular_fills)

        # =====================================================
        # Build GLB Assembly
        # =====================================================

        base_color = Color(0.6, 0.8, 1.0)   # lichtblauw
        fill_color = Color(0.0, 1.0, 0.0)   # groen

        asm = cq.Assembly()
        asm.add(shape, name="base", color=base_color)

        for idx, solid in enumerate(filled_solids):
            # extra safety: skip null solids
            try:
                if solid is None:
                    continue
                asm.add(solid, name=f"fill_{idx}", color=fill_color)
            except Exception as e:
                print(f"⚠️ Assembly add failed for fill_{idx}: {e}")

        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # STL
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Upload
        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

        # Save DB entry
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "bounding_box_x": dims["x"],
                "bounding_box_y": dims["y"],
                "bounding_box_z": dims["z"],
                "holes_detected": len(filled_solids),
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
            except Exception:
                pass
