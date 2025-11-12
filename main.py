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

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="2.0.0 - GLB colors")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🔑 Supabase setup
# -------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase.")
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# -------------------------------
# 🧠 Binnencontour detectie
# -------------------------------
def extract_inner_contours(shape):
    contours = []
    for face_id, face in enumerate(shape.Faces()):
        try:
            inner_wires = face.Wires()
            for w in inner_wires:
                edges = w.Edges()
                if not edges:
                    continue
                pts = []
                for e in edges:
                    for v in e.Vertices():
                        p = v.toTuple()
                        pts.append([round(p[0], 3), round(p[1], 3), round(p[2], 3)])
                if len(pts) >= 3:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    cz = sum(p[2] for p in pts) / len(pts)
                    contours.append({
                        "id": len(contours) + 1,
                        "face_id": face_id,
                        "points": pts,
                        "center": [round(cx, 3), round(cy, 3), round(cz, 3)]
                    })
        except Exception:
            continue
    print(f"✅ Found {len(contours)} inner contours")
    return contours

# -------------------------------
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")
    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)

    try:
        # verwijder oud bestand indien aanwezig
        try:
            storage.remove([remote_path])
        except Exception as e:
            print("ℹ️ No previous file to remove:", e)

        # upload nieuwe file
        with open(local_path, "rb") as f:
            res = storage.upload(remote_path, f)
        print("✅ Upload:", remote_name)

        public_url = storage.get_public_url(remote_path)
        return public_url
    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".step", ".stp"]:
            raise HTTPException(status_code=415, detail="Upload .STEP or .STP only")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # Bounding box
        bbox = shape.BoundingBox()
        dims = {
            "X": round(bbox.xlen, 3),
            "Y": round(bbox.ylen, 3),
            "Z": round(bbox.zlen, 3),
        }

        # Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # Detect binnencontouren
        inner_contours = extract_inner_contours(shape)

        # -------------------------------
        # 🔵 Maak blauw hoofdmodel
        # -------------------------------
        blue_model = cq.Assembly()
        blue_model.add(shape, color=cq.Color(0, 0, 1))  # blauw RGB(0,0,1)

        # -------------------------------
        # 🟢 Voeg groene vlakken toe op binnencontouren
        # -------------------------------
        for contour in inner_contours:
            pts = [cq.Vector(p[0], p[1], p[2]) for p in contour["points"]]
            try:
                wire = cq.Wire.makePolygon(pts, True)
                face = cq.Face.makeFromWires(wire)
                blue_model.add(face, color=cq.Color(0, 1, 0))  # groen RGB(0,1,0)
            except Exception as e:
                print(f"⚠️ Failed to create green patch for contour {contour['id']}: {e}")

        # -------------------------------
        # 📦 Exporteer STL en GLB
        # -------------------------------
        stl_path = tmp_path.replace(".step", ".stl")
        glb_path = tmp_path.replace(".step", ".glb")

        cq.exporters.export(shape, stl_path, "STL")
        blue_model.save(glb_path)  # kleurinformatie inbegrepen

        # -------------------------------
        # ☁️ Upload beide bestanden
        # -------------------------------
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))
        glb_public_url = upload_to_supabase(glb_path, file.filename.replace(".step", ".glb"))

        # -------------------------------
        # 💾 Opslaan in Supabase database
        # -------------------------------
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "volume_mm3": volume_mm3,
                "inner_contours": inner_contours,
                "model_url_stl": stl_public_url,
                "model_url_glb": glb_public_url,
            }).execute()

        # -------------------------------
        # ✅ Response
        # -------------------------------
        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "filename": file.filename,
            "innerContours": inner_contours,
            "modelURL_STL": stl_public_url,
            "modelURL_GLB": glb_public_url,
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})
    finally:
        for path in [tmp_path, tmp_path.replace(".step", ".stl"), tmp_path.replace(".step", ".glb")]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

# -------------------------------
# 🌐 Proxy endpoint
# -------------------------------
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
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {e}"})
