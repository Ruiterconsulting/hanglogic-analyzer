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
import datetime

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # eventueel beperken tot jouw Lovable domeinen
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
# 🕳️ Strikte Gatdetectie
# -------------------------------
def detect_analytic_holes_strict(shape,
                                 d_min=1.0, d_max=1000.0,
                                 xy_tol=0.2, d_tol=0.2,
                                 z_pair_min=0.5, z_pair_max=100.0,
                                 full_circle_tol_ratio=0.02):
    """
    Detecteert echte doorlopende ronde gaten (volledige 360° cirkels)
    en filtert:
      - Boogsegmenten (geen volledige 360°)
      - Countersinks of ruimingen
      - Losse cirkels zonder boven/onder paar
    Retourneert lijst met {x, y, z, diameter}
    """
    circles = []  # (cx, cy, cz, diameter)
    for face in shape.Faces():
        for edge in face.Edges():
            if edge.geomType() != "CIRCLE":
                continue
            try:
                circ = edge._geomAdaptor().Circle()
                loc = circ.Location()
                r = float(circ.Radius())
                if r <= 0:
                    continue

                # check volledige cirkel: lengte ≈ 2πr
                L = float(edge.Length())
                if L <= 0:
                    continue
                full_L = 2.0 * math.pi * r
                if abs(L - full_L) > full_circle_tol_ratio * full_L:
                    continue

                cx, cy, cz = float(loc.X()), float(loc.Y()), float(loc.Z())
                d = 2.0 * r

                if not (d_min <= d <= d_max):
                    continue

                circles.append((round(cx, 3), round(cy, 3), round(cz, 3), round(d, 3)))
            except Exception:
                continue

    if not circles:
        print("🕳️ No full-circle edges found.")
        return []

    # 2️⃣ Groepeer op XY & diameter
    groups = []
    for cx, cy, cz, d in circles:
        placed = False
        for g in groups:
            if (abs(g["x"] - cx) <= xy_tol and
                abs(g["y"] - cy) <= xy_tol and
                abs(g["d"] - d) <= d_tol):
                g["zs"].append(cz)
                placed = True
                break
        if not placed:
            groups.append({"x": cx, "y": cy, "d": d, "zs": [cz]})

    # 3️⃣ Zoek echte doorlopende paren (boven/onderzijde)
    holes = []
    for g in groups:
        zs = sorted(g["zs"])
        if len(zs) < 2:
            continue
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                dz = abs(zs[j] - zs[i])
                if z_pair_min <= dz <= z_pair_max:
                    z_avg = round((zs[i] + zs[j]) / 2.0, 3)
                    holes.append({
                        "x": round(g["x"], 3),
                        "y": round(g["y"], 3),
                        "z": z_avg,
                        "diameter": round(g["d"], 3)
                    })
                    break

    # 4️⃣ Duplicaten verwijderen
    deduped = []
    for h in holes:
        if not any(abs(h["x"] - u["x"]) <= xy_tol and
                   abs(h["y"] - u["y"]) <= xy_tol and
                   abs(h["diameter"] - u["diameter"]) <= d_tol for u in deduped):
            deduped.append(h)

    print(f"🕳️ Detected {len(deduped)} clean through-holes (strict).")
    return deduped


# -------------------------------
# 📦 Upload helper (unieke bestandsnamen)
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """Upload bestand naar Supabase met unieke timestamp."""
    if not supabase:
        raise RuntimeError("Supabase not initialized.")

    bucket = "cad-models"
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(remote_name)
    unique_name = f"{base}_{timestamp}{ext}"
    remote_path = f"analyzed/{unique_name}"

    storage = supabase.storage.from_(bucket)
    with open(local_path, "rb") as f:
        storage.upload(remote_path, f)

    public_url = storage.get_public_url(remote_path)
    print(f"🌍 Upload gelukt: {public_url}")
    return public_url


# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (v1.8.0, unique uploads + color prep)"}

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------
# 🧰 Bestandstype check
# -------------------------------
def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP"
        )


# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        _ensure_step_extension(file.filename)

        # tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # STEP inladen
        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # bounding box
        bbox = shape.BoundingBox()
        dims = {
            "X": round(bbox.xlen, 3),
            "Y": round(bbox.ylen, 3),
            "Z": round(bbox.zlen, 3)
        }

        # volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # detect holes
        holes = detect_analytic_holes_strict(shape)
        holes_detected = len(holes)

        # export STL
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # upload
        stl_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # opslaan in supabase
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "volume_mm3": volume_mm3,
                "holes_detected": holes_detected,
                "holes_data": holes,
                "model_url": stl_url,
                "units": "mm",
                "created_at": "now()"
            }).execute()

        # resultaat terug
        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "filename": file.filename,
            "holesDetected": holes_detected,
            "holes": holes,
            "modelURL": stl_url
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})

    finally:
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


# -------------------------------
# 🌐 Proxy voor CORS
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
        print("⚠️ Proxy fetch failed:", e)
        return JSONResponse(status_code=500, content={"error": f"Proxy failed: {str(e)}"})
