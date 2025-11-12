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


app = FastAPI(title="HangLogic Analyzer API", version="2.2.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# Supabase connect
# ------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    print("⚠️ Missing Supabase credentials")


# ------------------------------
# Upload helper
# ------------------------------
def upload_new_file(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase not initialized")

    bucket = "cad-models"
    storage = supabase.storage.from_(bucket)

    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{os.urandom(4).hex()}{ext}"

    remote_path = f"analyzed/{unique_name}"
    print("📤 Upload:", remote_path)

    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    return storage.get_public_url(remote_path)


# ------------------------------
# Detect inner wires
# ------------------------------
def detect_inner_wires(shape):
    data = []

    for f_idx, face in enumerate(shape.Faces()):
        wires = list(face.Wires())

        if len(wires) <= 1:
            continue

        outer = wires[0]
        inners = wires[1:]

        data.append((f_idx, face, inners))

    return data


# ------------------------------
@app.get("/")
def root():
    return {"status": "ok", "version": "2.2.8"}


def ensure_step(filename):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".stp", ".step"]:
        raise HTTPException(415, "Upload only .STEP or .STP files")


# ------------------------------
# MAIN ANALYZER
# ------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_step = tmp_stl = tmp_glb = None

    try:
        ensure_step(file.filename)

        # save STEP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as t:
            t.write(await file.read())
            tmp_step = t.name

        model = cq.importers.importStep(tmp_step)
        shape = model.val()

        bbox = shape.BoundingBox()
        dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3)
        }

        thickness = dims["z"]

        inner_data = detect_inner_wires(shape)

        filled_solids = []

        # -----------------------------
        # Generate fills
        # -----------------------------
        for f_idx, face, wires in inner_data:
            for w_idx, wire in enumerate(wires):

                # Skip invalid wires
                try:
                    if wire.isNull() or wire.Length() < 0.5:
                        print(f"⚠️ skipping null/short wire {w_idx}")
                        continue
                except:
                    continue

                try:
                    normal = face.normalAt()
                except:
                    normal = cq.Vector(0, 0, 1)

                try:
                    center = wire.Center()
                except:
                    center = face.Center()

                plane = cq.Plane(
                    (center.x, center.y, center.z),
                    (normal.x, normal.y, normal.z)
                )

                try:
                    wp = cq.Workplane(plane).add(wire)

                    pos = wp.toPending().extrude(thickness / 2)
                    neg = wp.toPending().extrude(-thickness / 2)
                    fill = pos.union(neg)

                    filled_solids.append(fill)

                except Exception as e:
                    print(f"⚠️ fill failed: face {f_idx}, wire {w_idx}: {e}")

        # -----------------------------
        # Build GLB assembly
        # -----------------------------
        blue = (0.6, 0.8, 1.0)
        green = (0.0, 1.0, 0.0)

        asm = cq.Assembly()

        asm.add(shape, name="base", color=blue)

        for idx, solid in enumerate(filled_solids):
            asm.add(solid, name=f"fill_{idx}", color=green)

        tmp_glb = tmp_step.replace(".step", ".glb")
        asm.save(tmp_glb, exportType="GLTF")

        # STL export
        tmp_stl = tmp_step.replace(".step", ".stl")
        cq.exporters.export(shape, tmp_stl, "STL")

        # Upload
        stl_url = upload_new_file(tmp_stl, file.filename.replace(".step", ".stl"))
        glb_url = upload_new_file(tmp_glb, file.filename.replace(".step", ".glb"))

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
            "modelURL_GLB": glb_url
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

    finally:
        for p in [tmp_step, tmp_stl, tmp_glb]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
