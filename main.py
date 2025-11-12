from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import math
import traceback
from supabase import create_client, Client
import httpx

# ============================================================
# 🌍 App configuratie
# ============================================================
app = FastAPI(title="HangLogic Analyzer API", version="1.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔑 Supabase setup
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase.")
        try:
            buckets = supabase.storage.list_buckets()
            print("📦 Buckets found:", [b.name for b in buckets])
        except Exception as e:
            print("⚠️ Could not list buckets:", e)
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# ============================================================
# 🕳️ Detecteer binnencontouren (edges met inner loops)
# ============================================================
def detect_inner_contours(shape):
    contours = []
    contour_id = 0
    for f_id, face in enumerate(shape.Faces()):
        try:
            wires = face.Wires()
            if len(wires) <= 1:
                continue
            # Eerste wire = buitenrand, de rest = binnencontouren
            for inner_wire in wires[1:]:
                pts = []
                for v in inner_wire.Vertices():
                    p = v.toTuple()
                    pts.append([round(p[0], 3), round(p[1], 3), round(p[2], 3)])
                if len(pts) >= 3:
                    contour_id += 1
                    contours.append({"id": contour_id, "face_id": f_id, "points": pts})
        except Exception as e:
            print(f"⚠️ Failed on face {f_id}: {e}")
    print(f"✅ Found {len(contours)} inner contours")
    return contours

# ============================================================
# 📦 Upload helper
# ============================================================
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")
    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)
    try:
        # Verwijder bestaand bestand om duplicates te voorkomen
        try:
            storage.remove([remote_path])
        except Exception:
            pass

        with open(local_path, "rb") as f:
            storage.upload(remote_path, f)

        public_url = storage.get_public_url(remote_path)
        print(f"🌍 Uploaded {remote_name}")
        return public_url
    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")

# ============================================================
# 🔍 Basis endpoints
# ============================================================
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (v1.8.0 - inner contours + colors)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# 🧰 Bestandstype check
# ============================================================
def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP",
        )

# ============================================================
# 🧮 Analyzer endpoint
# ============================================================
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        _ensure_step_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"🧠 Analyzing: {file.filename}")
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # 📏 Bounding box
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3),
        }
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 📦 Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 🟢 Detect binnencontouren
        inner_contours = detect_inner_contours(shape)

        # 🎨 Maak blauw basismodel
        blue_model = cq.Workplane("XY").add(shape).setColor(cq.Color(0, 0, 1))  # blauw

        # 🟢 Voeg groene patches toe (zichtbare binnencontouren)
        for contour in inner_contours:
            try:
                pts = contour["points"]
                if not pts or len(pts) < 3:
                    continue
                avg_z = sum(p[2] for p in pts) / len(pts)
                projected_pts = [cq.Vector(p[0], p[1], avg_z + 0.1) for p in pts]

                if projected_pts[0] != projected_pts[-1]:
                    projected_pts.append(projected_pts[0])

                wire = cq.Wire.makePolygon(projected_pts)
                face = cq.Face.makeFromWires(wire)
                blue_model.add(face, color=cq.Color(0, 1, 0))  # groen
            except Exception as e:
                print(f"⚠️ Failed to create green patch for contour {contour.get('id', '?')}: {e}")

        # 📤 Exporteer STL
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 📤 Exporteer GLB (met kleuren)
        glb_path = tmp_path.replace(".step", ".glb")
        cq.exporters.export(blue_model, glb_path, "GLTF")

        # ☁️ Uploaden
        stl_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))
        glb_url = upload_to_supabase(glb_path, file.filename.replace(".step", ".glb"))

        # 🧾 Opslaan in database
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "units": "mm",
                "volume_mm3": volume_mm3,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "inner_contours": inner_contours,
                "model_url": glb_url,
                "created_at": "now()",
            }).execute()

        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": volume_mm3,
            "filename": file.filename,
            "innerContours": inner_contours,
            "modelURL": glb_url,
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        print("❌ Error during analyze:", e)
        return JSONResponse(status_code=500, content={
            "error": f"Analysis failed: {e}",
            "trace": tb
        })
    finally:
        for path in [tmp_path,
                     tmp_path.replace(".step", ".stl") if tmp_path else None,
                     tmp_path.replace(".step", ".glb") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

# ============================================================
# 🌐 Proxy endpoint voor CORS-vrije toegang
# ============================================================
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    url = f"https://sywnjytfygvotskufvzs.supabase.co/storage/v1/object/public/{path}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": r.headers.get("content-type", "application/octet-stream"),
        }
        return StreamingResponse(iter([r.content]), headers=headers)
    except Exception as e:
        print("⚠️ Proxy fetch failed:", e)
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {str(e)}"})
