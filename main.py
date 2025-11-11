from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os
import traceback
from supabase import create_client, Client
import httpx
import math

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # eventueel beperken tot Lovable domeinen
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
        try:
            buckets = supabase.storage.list_buckets()
            print("📦 Buckets found:", [b.name for b in buckets])
        except Exception as e:
            print("⚠️ Could not list buckets:", e)
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")


# -------------------------------
# 🧩 Binnencontour detectie
# -------------------------------
def detect_inner_contours(shape):
    """
    Detecteert ALLE binnencontouren (rond, sleuf, vierkant, complex)
    en negeert buitencontouren of randafrondingen.
    """
    contours = []
    contour_id = 0

    try:
        bbox = shape.BoundingBox()
        max_dim = max(bbox.xlen, bbox.ylen, bbox.zlen)

        for face_id, face in enumerate(shape.Faces()):
            wires = face.Wires()
            if len(wires) <= 1:
                continue  # geen binnencontouren

            outer_wire = wires[0]
            inner_wires = wires[1:]

            for wire in inner_wires:
                contour_id += 1
                points = []
                for edge in wire.Edges():
                    for vertex in edge.Vertices():
                        v = vertex.toTuple()
                        points.append([
                            round(float(v[0]), 3),
                            round(float(v[1]), 3),
                            round(float(v[2]), 3)
                        ])

                # Sla contouren over die maar 1 of 2 punten hebben
                if len(points) < 3:
                    continue

                # Gemiddelde positie = centrum
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                cz = sum(p[2] for p in points) / len(points)

                # Grootte (schatting)
                dx = max(p[0] for p in points) - min(p[0] for p in points)
                dy = max(p[1] for p in points) - min(p[1] for p in points)
                dz = max(p[2] for p in points) - min(p[2] for p in points)
                diag = (dx**2 + dy**2 + dz**2) ** 0.5

                # Alleen contouren die klein genoeg zijn (max 50% van grootste dimensie)
                if diag > (max_dim * 0.5):
                    continue

                # Vormclassificatie
                edge_types = [e.geomType() for e in wire.Edges()]
                if all(t == "CIRCLE" for t in edge_types):
                    contour_type = "round"
                elif any(t == "ELLIPSE" for t in edge_types):
                    contour_type = "oval"
                elif edge_types.count("LINE") >= 4:
                    contour_type = "rectangular"
                else:
                    contour_type = "complex"

                contours.append({
                    "id": contour_id,
                    "face_id": face_id,
                    "type": contour_type,
                    "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
                    "points": points,
                    "edgeCount": len(wire.Edges()),
                    "size_estimate": round(diag, 3)
                })

        print(f"🧩 Detected {len(contours)} filtered inner contours.")
    except Exception as e:
        print("⚠️ Inner contour detection failed:", e)

    return contours


# -------------------------------
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """Upload bestand naar Supabase Storage en geef publieke URL terug."""
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)

    try:
        # Verwijder oude versie als die al bestaat
        try:
            files = storage.list(path="analyzed")
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    print(f"⚠️ Bestand {remote_name} bestaat al — verwijderen...")
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden of kon niet lezen:", e)

        with open(local_path, "rb") as f:
            res = storage.upload(remote_path, f)
        print("✅ Upload gelukt:", res)

        public_url = storage.get_public_url(remote_path)
        print("🌍 Public URL:", public_url)
        return public_url

    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")


# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (inner contour detection + Supabase)"}


@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Validatie
        ext = (os.path.splitext(file.filename)[1] or "").lower()
        if ext not in [".step", ".stp"]:
            raise HTTPException(status_code=415, detail="Upload .STEP or .STP file")

        # Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Inladen in CadQuery
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # Bounding box
        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3),
        }

        # Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # Binnencontouren detecteren
        inner_contours = detect_inner_contours(shape)

        # STL exporteren
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # Uploaden naar Supabase
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # Opslaan in database
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": dims,
                "inner_contours": inner_contours,
                "units": "mm",
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "volume_mm3": volume_mm3,
                "model_url": stl_public_url,
                "created_at": "now()",
            }).execute()

        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "filename": file.filename,
            "innerContours": inner_contours,
            "modelURL": stl_public_url,
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={
            "error": f"Analysis failed: {str(e)}",
            "trace": tb
        })
    finally:
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


# -------------------------------
# 🌐 Proxy endpoint voor Lovable (CORS fix)
# -------------------------------
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    """Proxy voor Lovable 3D viewer zodat CORS headers worden toegevoegd."""
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
