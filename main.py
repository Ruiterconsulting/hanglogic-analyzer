from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cadquery as cq
import tempfile
import os
import traceback
from supabase import create_client, Client

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later kun je dit beperken tot jouw Lovable domeinen
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
# 🕳️ Gatdetectie helper
# -------------------------------
def detect_holes(shape):
    """
    Detecteert ronde gaten in het model.
    Herkent ook doorlopende gaten (boven- en onderzijde).
    Retourneert lijst met {x, y, z, diameter}.
    """
    holes = []
    try:
        # Verzamel cirkelranden
        circular_edges = []
        for edge in shape.Edges():
            try:
                t = edge.geomType()
                if t == "CIRCLE":
                    circ = edge._geomAdaptor().Circle()
                    radius = float(circ.Radius())
                    center = circ.Location().toTuple()[0][:3]
                    circular_edges.append({"center": center, "radius": radius})
            except Exception:
                continue

        # Combineer cirkels die op ±zelfde XY zitten (boven- en onderzijde)
        matched = []
        for i, c1 in enumerate(circular_edges):
            for j, c2 in enumerate(circular_edges):
                if i >= j:
                    continue
                dx = abs(c1["center"][0] - c2["center"][0])
                dy = abs(c1["center"][1] - c2["center"][1])
                dr = abs(c1["radius"] - c2["radius"])
                if dx < 0.5 and dy < 0.5 and dr < 0.5:
                    mid_z = (c1["center"][2] + c2["center"][2]) / 2
                    holes.append({
                        "x": round(c1["center"][0], 3),
                        "y": round(c1["center"][1], 3),
                        "z": round(mid_z, 3),
                        "diameter": round(c1["radius"] * 2, 3)
                    })
                    matched.append(i)
                    matched.append(j)
                    break

        # Als er geen matches zijn, voeg losse cilinders toe
        if not holes:
            for face in shape.Faces():
                if face.geomType() == "CYLINDER":
                    try:
                        surf = face.Surface()
                        radius = float(surf.Radius())
                        loc = surf.Location().toTuple()[0][:3]
                        holes.append({
                            "x": round(loc[0], 3),
                            "y": round(loc[1], 3),
                            "z": round(loc[2], 3),
                            "diameter": round(radius * 2, 3)
                        })
                    except Exception:
                        continue

        # Dubbele entries verwijderen
        unique = []
        for h in holes:
            if not any(abs(h["x"] - u["x"]) < 0.1 and abs(h["y"] - u["y"]) < 0.1 for u in unique):
                unique.append(h)
        holes = unique

    except Exception as e:
        print("⚠️ Hole detection failed:", e)
    return holes

# -------------------------------
# 📦 Upload helper (definitieve versie)
# -------------------------------
def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """
    Upload bestand naar Supabase Storage.
    Bestaat het al → dan verwijderen we het eerst.
    Retourneert de publieke URL.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)

    try:
        # 🔹 Bestaat het bestand al? -> verwijder het
        try:
            files = storage.list(path="analyzed")
            for f in files:
                # f kan dict of object zijn afhankelijk van libversie
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    print(f"⚠️ Bestand {remote_name} bestaat al — verwijderen...")
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden of kon niet lezen:", e)

        # 🔹 Upload nieuw bestand
        with open(local_path, "rb") as f:
            res = storage.upload(remote_path, f)
        print("✅ Upload gelukt:", res)

        # 🔹 Publieke URL ophalen
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
    return {"message": "STEP analyzer live ✅ (CadQuery + STL export + holes + Supabase)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧰 Hulpfunctie
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
    """
    Ontvangt een .STEP/.STP bestand, berekent bounding box en volume,
    detecteert gaten, exporteert STL, uploadt naar Supabase, en slaat metadata op.
    """
    tmp_path = None
    try:
        # 1️⃣ Validatie
        _ensure_step_extension(file.filename)

        # 2️⃣ Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3️⃣ Inladen in CadQuery
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # 4️⃣ Bounding box
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(float(bbox.xlen), 3),
            "y": round(float(bbox.ylen), 3),
            "z": round(float(bbox.zlen), 3),
        }
        # Sorteer op grootte: X=langste, Y=middelste, Z=kleinste
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 5️⃣ Volume
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 6️⃣ Gaten detectie
        holes = detect_holes(shape)
        holes_detected = len(holes)

        # 7️⃣ STL-export
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 8️⃣ Upload naar Supabase (Storage)
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # 9️⃣ Opslaan in database (Table: analyzed_parts)
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "holes_detected": holes_detected,
                "holes_data": holes,              # JSONB kolom
                "created_at": "now()",
                "units": "mm",
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "model_url": stl_public_url,
            }).execute()

        # 🔟 Antwoord naar frontend
        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
                "filename": file.filename,
                "holesDetected": holes_detected,
                "holes": holes,
                "modelURL": stl_public_url,
            }
        )

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}", "trace": tb},
        )
    finally:
        # Verwijder tijdelijke bestanden
        for path in [tmp_path, tmp_path.replace(".step", ".stl") if tmp_path else None]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
