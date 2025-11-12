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
import trimesh

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
                                 d_min=2.0, d_max=30.0,
                                 xy_tol=0.2, d_tol=0.2,
                                 z_pair_min=1.0, z_pair_max=40.0,
                                 full_circle_tol_ratio=0.02):
    """
    Detecteert echte doorlopende ronde gaten en filtert:
    - boogsegmenten (géén volledige 360°)
    - countersinks / ruimingen
    - losse cirkels zonder tegenhanger
    Retourneert lijst met {x,y,z,diameter} (z = gemiddelde van boven/onder).
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
                    # hoogstwaarschijnlijk een boogdeel → overslaan
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

    # 3️⃣ Zoek echte paren (boven/onderzijde)
    holes = []
    for g in groups:
        zs = sorted(g["zs"])
        if len(zs) < 2:
            continue

        found = False
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
                    found = True
                    break
            if found:
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
# 📦 Upload helper
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")
    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)
    try:
        # verwijder oud bestand
        try:
            files = storage.list(path="analyzed")
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    print(f"⚠️ Bestand {remote_name} bestaat al — verwijderen...")
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden:", e)

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
    return {"message": "STEP analyzer live ✅ (v1.8.0 strict holes + STL+GLB export)"}

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
    stl_path = None
    glb_path = None

    try:
        _ensure_step_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1️⃣ STEP importeren
        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # 2️⃣ Bounding box
        bbox = shape.BoundingBox()
        bbox_dims = {
            "x": round(bbox.xlen, 3),
            "y": round(bbox.ylen, 3),
            "z": round(bbox.zlen, 3)
        }
        sorted_dims = sorted(bbox_dims.values(), reverse=True)
        dims_sorted = {
            "X": sorted_dims[0],
            "Y": sorted_dims[1],
            "Z": sorted_dims[2]
        }

        # 3️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 4️⃣ Hole detection
        holes = detect_analytic_holes_strict(shape)
        holes_detected = len(holes)

        # 5️⃣ STL exporteren
        base_name, _ext = os.path.splitext(file.filename)
        safe_base_name = base_name  # evt. later sanitizen
        stl_filename = f"{safe_base_name}.stl"
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 6️⃣ STL uploaden
        stl_public_url = upload_to_supabase(stl_path, stl_filename)

        # 7️⃣ GLB genereren (lichtblauw part + groene gaten)
        glb_public_url = None
        try:
            # hoofdmesh uit STL
            mesh = trimesh.load_mesh(stl_path)
            if mesh.is_empty:
                raise RuntimeError("Loaded STL mesh is empty.")

            # hoofdmesh lichtblauw kleuren (RGBA 0-255)
            light_blue = [150, 200, 255, 255]
            mesh.visual.vertex_colors = [light_blue] * len(mesh.vertices)

            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="body")

            # maak per gat een groene cilinder die het gat 'opvult'
            green = [0, 255, 0, 255]
            height = float(bbox_dims["z"]) * 1.5  # iets langer dan dikte, zodat hij door het part steekt

            for idx, hole in enumerate(holes):
                radius = float(hole["diameter"]) / 2.0
                # cilinder in Z-richting, gecentreerd op (0,0,0)
                cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
                cyl.visual.vertex_colors = [green] * len(cyl.vertices)

                # verplaats naar gat-positie; we gebruiken hole["z"] als middelpunt in Z
                cx = float(hole["x"])
                cy = float(hole["y"])
                cz = float(hole["z"])
                cyl.apply_translation((cx, cy, cz))

                scene.add_geometry(cyl, node_name=f"hole_{idx}")

            # exporteer scene naar GLB
            glb_bytes = scene.export(file_type="glb")
            glb_path = tmp_path.replace(".step", ".glb")
            with open(glb_path, "wb") as f:
                f.write(glb_bytes)

            glb_filename = f"{safe_base_name}.glb"
            glb_public_url = upload_to_supabase(glb_path, glb_filename)
        except Exception as e:
            print("⚠️ GLB generation/upload failed:", e)
            glb_public_url = None

        # 8️⃣ Opslaan in Supabase database
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {
                    "x": dims_sorted["X"],
                    "y": dims_sorted["Y"],
                    "z": dims_sorted["Z"]
                },
                "holes_detected": holes_detected,
                "holes_data": holes,
                "units": "mm",
                "bounding_box_x": dims_sorted["X"],
                "bounding_box_y": dims_sorted["Y"],
                "bounding_box_z": dims_sorted["Z"],
                "model_url": stl_public_url,
                "model_url_glb": glb_public_url,
            }).execute()

        # 9️⃣ Resultaat terugsturen
        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims_sorted,
            "boundingBoxAxes": bbox_dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 is not None else None,
            "filename": file.filename,
            "holesDetected": holes_detected,
            "holes": holes,
            "modelURL": stl_public_url,
            "modelURLGLB": glb_public_url,
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        # alle tijdelijke bestanden opruimen
        for path in [tmp_path, stl_path, glb_path]:
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
    # TIP: eventueel SUPABASE_URL gebruiken i.p.v. hardcoded
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
