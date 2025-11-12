from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cadquery as cq
import tempfile
import os
import math
import traceback
from supabase import create_client, Client
import httpx


app = FastAPI(title="HangLogic Analyzer API", version="2.2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Supabase connect
# --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    print("⚠️ Supabase env vars missing")


# --------------------------
# Upload helper
# --------------------------
def upload_new_file(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized")

    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    # Maak naam altijd uniek
    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"
    remote_path = f"analyzed/{unique_name}"

    print("📤 Uploading:", remote_path)

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# --------------------------
# Helper: detect inner contours
# --------------------------
def detect_inner_wires(shape):
    """Return list of (face_index, face, inner_wire_list)."""

    result = []

    for f_idx, face in enumerate(shape.Faces()):
        wires = list(face.Wires())

        if len(wires) <= 1:
            continue  # no inner contours

        # eerste wire is altijd outer boundary
        outer_wire = wires[0]
        inner = wires[1:]

        result.append((f_idx, face, inner))

    return result


# --------------------------
# Root test
# --------------------------
@app.get("/")
def root():
    return {"status": "HangLogic API live", "version": "2.2.7"}


# --------------------------
# Check extension
# --------------------------
def ensure_step(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(415, f"Invalid type '{ext}', upload .STEP/.STP only")


# --------------------------
# ANALYZER
# --------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None

    try:
        ensure_step(file.filename)

        # write STEP to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        print(f"🧠 Analyzing {file.filename}")

        bbox = shape.BoundingBox()
        dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3),
        }

        plate_thickness = dims["z"]

        # detect holes / inner contours
        inner_data = detect_inner_wires(shape)

        # generate fills
        filled_solids = []

        for f_idx, face, wire_list in inner_data:
            for w_idx, wire in enumerate(wire_list):

                # 1) Skip null / invalid
                try:
                    if wire.isNull() or wire.Length() < 0.01:
                        print(f"⚠️ Skipping invalid wire (face {f_idx}, wire {w_idx})")
                        continue
                except:
                    print(f"⚠️ wire exception – skipping (face {f_idx})")
                    continue

                # 2) Normaal pakken
                try:
                    normal = face.normalAt()
                except:
                    normal = cq.Vector(0, 0, 1)

                # 3) Center pakken
                try:
                    center = wire.Center()
                except:
                    # fallback op face-center
                    center = face.Center()

                # 4) Workplane op het draadje
                plane = cq.Plane(
                    (center.x, center.y, center.z),
                    (normal.x, normal.y, normal.z)
                )

                try:
                    wp = cq.Workplane(plane).add(wire)

                    # 5) symmetrisch extruderen door volledige plaat
                    pos = wp.toPending().extrude(plate_thickness / 2)
                    neg = wp.toPending().extrude(-plate_thickness / 2)
                    fill = pos.union(neg)

                    filled_solids.append(fill)

                except Exception as e:
                    print(f"⚠️ Fill failed on face {f_idx}, wire {w_idx}: {e}")

        # combine into colored export
        blue = (0.6, 0.8, 1.0)
        green = (0.0, 1.0, 0.0)

        colored = cq.Compound.makeCompound([
            shape.setColor(*blue)
        ] + [s.setColor(*green) for s in filled_solids])

        # export STL
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # export GLB
        tmp_glb = tmp_step.replace(".step", ".glb")
        cq.exporters.export(colored, tmp_glb, exportType="GLTF")  # GLB via GLTF

        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

        # store in Supabase table
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "bounding_box_x": dims["x"],
                "bounding_box_y": dims["y"],
                "bounding_box_z": dims["z"],
                "holes_detected": len(inner_data),
                "model_url": stl_url,
                "model_url_glb": glb_url,
                "units": "mm",
                "created_at": "now()",
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
        # cleanup
        for path in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass
