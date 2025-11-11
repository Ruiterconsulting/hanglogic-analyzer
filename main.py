# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import os
import math
import traceback
import numpy as np

# CAD / Mesh
import cadquery as cq
import trimesh
from shapely.geometry import Polygon

# Supabase + HTTP proxy
from supabase import create_client, Client
import httpx

# -------------------------------
# 🌍 App configuratie
# -------------------------------
app = FastAPI(title="HangLogic Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # beperk later tot jouw Lovable domeinen
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
# 🧰 Helpers
# -------------------------------
def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP"
        )

def upload_to_supabase(local_path: str, remote_name: str) -> str:
    """
    Upload bestand naar Supabase Storage (bucket 'cad-models').
    Bestaat het al → verwijder eerst. Retourneert public URL.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")

    bucket = "cad-models"
    remote_path = f"analyzed/{remote_name}"
    storage = supabase.storage.from_(bucket)

    try:
        # Verwijder bestaande met dezelfde naam
        try:
            files = storage.list(path="analyzed")
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name == remote_name:
                    storage.remove([f"analyzed/{remote_name}"])
                    break
        except Exception as e:
            print("ℹ️ Geen bestaande bestanden gevonden of list() faalde:", e)

        # Upload
        with open(local_path, "rb") as f:
            storage.upload(remote_path, f)

        # Public URL
        public_url = storage.get_public_url(remote_path)
        return public_url
    except Exception as e:
        raise RuntimeError(f"Upload to Supabase failed: {e}")

# -------------------------------
# 🕳️ Mesh-snede-gebaseerde binnencontour & gatdetectie
# -------------------------------
def _equiv_diameter_from_area(area: float) -> float:
    """Equivalent cirkeldiameter uit polygon-oppervlak."""
    if area <= 0:
        return 0.0
    return 2.0 * math.sqrt(area / math.pi)

def _principal_slice_axis(extents: np.ndarray) -> int:
    """Kies de as met de kleinste extent als snederichting (typisch plaatdikte)."""
    # extents: [ex, ey, ez]
    return int(np.argmin(extents))

def _plane_normal_for_axis(axis: int) -> np.ndarray:
    normals = [
        np.array([1.0, 0.0, 0.0]),  # snede loodrecht op X (dus normal = X)
        np.array([0.0, 1.0, 0.0]),  # snede loodrecht op Y
        np.array([0.0, 0.0, 1.0])   # snede loodrecht op Z
    ]
    return normals[axis]

def detect_through_holes_from_mesh(mesh: trimesh.Trimesh,
                                   samples: int = 17,
                                   min_seen_ratio: float = 0.4,
                                   xy_tol: float = 0.6,
                                   d_tol: float = 0.6,
                                   margin_ratio: float = 0.1):
    """
    Snijdt het mesh op meerdere hoogtes (langs kleinste bbox-as).
    Voor elke snede:
      - Neem grootste polygon als buitencontour
      - Alle kleinere polygons zijn binnencontouren
      - Verzamel (centroid, area → equiv. diameter)
    Daarna:
      - Cluster per (x,y,diameter) met tolerantie
      - Een cluster telt als doorlopend gat als het in genoeg snedes gezien is.

    Retourneert:
      holes: [{x,y,z,diameter}]
      inner_contours: [{id, face_id: 0, points:[[x,y,z],...], center:[x,y,z]}]  (één representatieve polygon per gat)
    """
    if mesh.is_empty:
        return [], []

    # Bepaal snederichting
    bounds = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
    extents = bounds[1] - bounds[0]
    axis = _principal_slice_axis(extents)
    normal = _plane_normal_for_axis(axis)

    # Snedes ván min+margin tot max-margin
    min_v = bounds[0][axis]
    max_v = bounds[1][axis]
    span = float(max_v - min_v)
    if span <= 0:
        return [], []

    offset = margin_ratio * span
    start = min_v + offset
    end = max_v - offset
    if end <= start:
        # te dun of marge te groot → geen snedes
        return [], []

    planes = np.linspace(start, end, samples)

    # Verzamelen van observaties per snede
    observations = []  # list of dicts: {x,y,z,diameter, poly_points}
    for v in planes:
        origin = np.array([0.0, 0.0, 0.0])
        origin[axis] = v
        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
            if section is None:
                continue

            # Breng naar 2D vlak
            planar = section.to_planar()
            # Haal shapely polygons op
            polys = planar.polygons_full  # lijst van shapely.Polygon
            if not polys:
                continue

            # Sorteer op area (grootste = buiten)
            polys = sorted(polys, key=lambda p: p.area, reverse=True)

            if len(polys) == 1:
                # alleen buitencontour → geen binnencontouren
                continue

            outer = polys[0]
            inners = polys[1:]  # mogelijke gaten / binnencontouren

            for poly in inners:
                if not isinstance(poly, Polygon):
                    continue
                if poly.area <= 1e-6:
                    continue

                # centroid in 2D → terug naar 3D wereldcoördinaten
                c2d = np.array([poly.centroid.x, poly.centroid.y, 0.0])
                # Inverse mapping naar 3D:
                # planar.to_3D(np.column_stack([...])) verwacht Nx2; we voeren één punt
                c3d = planar.to_3D(np.array([[c2d[0], c2d[1]]]))[0]

                # polygon punten nemen (exterior)
                coords2d = np.array(poly.exterior.coords)  # Nx2
                pts3d = planar.to_3D(coords2d[:, :2])     # Nx3

                eq_d = _equiv_diameter_from_area(poly.area)

                observations.append({
                    "x": float(round(c3d[0], 3)),
                    "y": float(round(c3d[1], 3)),
                    "z": float(round(c3d[2], 3)),  # snedevlakcoördinaat
                    "diameter": float(round(eq_d, 3)),
                    "points": pts3d.tolist()
                })
        except Exception as e:
            # Snede kan soms falen; gewoon overslaan
            print("ℹ️ section failed at v=", v, "err:", e)
            continue

    if not observations:
        print("🕳️ No inner contours detected across slices.")
        return [], []

    # Cluster op (x,y,diameter) met toleranties
    clusters = []  # each: {"x","y","d","zs":[...],"samples":[observation indices]}
    for idx, ob in enumerate(observations):
        placed = False
        for cl in clusters:
            if (abs(cl["x"] - ob["x"]) <= xy_tol and
                abs(cl["y"] - ob["y"]) <= xy_tol and
                abs(cl["d"] - ob["diameter"]) <= d_tol):
                cl["zs"].append(ob["z"])
                cl["samples"].append(idx)
                placed = True
                break
        if not placed:
            clusters.append({
                "x": ob["x"], "y": ob["y"], "d": ob["diameter"],
                "zs": [ob["z"]], "samples": [idx]
            })

    # Filter clusters die doorlopend zijn (genoeg snedes gezien)
    min_seen = max(2, int(math.ceil(min_seen_ratio * samples)))
    holes = []
    inner_contours = []
    next_id = 1

    for cl in clusters:
        seen = len(set([round(z, 3) for z in cl["zs"]]))
        if seen < min_seen:
            continue

        z_avg = round(float(np.mean(cl["zs"])), 3)
        hole = {
            "x": round(cl["x"], 3),
            "y": round(cl["y"], 3),
            "z": z_avg,
            "diameter": round(cl["d"], 3)
        }
        holes.append(hole)

        # Neem één representatieve polygon uit deze cluster
        # (het sample met z dichtst bij z_avg)
        cand_idx = min(
            cl["samples"],
            key=lambda i: abs(observations[i]["z"] - z_avg)
        )
        poly_pts = observations[cand_idx]["points"]
        inner_contours.append({
            "id": next_id,
            "face_id": 0,
            "points": poly_pts,
            "center": [hole["x"], hole["y"], z_avg]
        })
        next_id += 1

    print(f"🕳️ Detected {len(holes)} through-holes from mesh slices.")
    return holes, inner_contours

# -------------------------------
# 🔍 Basis endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (mesh slicing holes + inner contours + STL export)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# 🧮 Analyzer endpoint
# -------------------------------
@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """
    - Valideert en slaat STEP tijdelijk op
    - Berekent bbox/volume (CadQuery, mm)
    - Exporteert STL
    - Detecteert gaten & binnencontouren via mesh-slices (trimesh + shapely)
    - Uploadt STL naar Supabase
    - Schrijft metadata (incl. holes & inner_contours) naar DB
    """
    tmp_path = None
    stl_path = None
    try:
        # 1) Validatie
        _ensure_step_extension(file.filename)

        # 2) Tijdelijk opslaan
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3) STEP → CadQuery shape
        model = cq.importers.importStep(tmp_path)
        shape = model.val()

        # 4) Bounding box (mm)
        bbox = shape.BoundingBox()
        raw_dims = {
            "x": round(float(bbox.xlen), 3),
            "y": round(float(bbox.ylen), 3),
            "z": round(float(bbox.zlen), 3),
        }
        sorted_dims = sorted(raw_dims.values(), reverse=True)
        dims = {"X": sorted_dims[0], "Y": sorted_dims[1], "Z": sorted_dims[2]}

        # 5) Volume (mm³)
        try:
            volume_mm3 = float(shape.Volume())
        except Exception:
            volume_mm3 = None

        # 6) Export STL
        stl_path = tmp_path.replace(".step", ".stl")
        cq.exporters.export(shape, stl_path, "STL")

        # 7) Mesh laden + gaten/contouren detecteren
        mesh = trimesh.load_mesh(stl_path, process=True)
        holes, inner_contours = detect_through_holes_from_mesh(
            mesh,
            samples=17,          # meer samples = robuuster
            min_seen_ratio=0.4,  # % snedes waarin de opening moet terugkomen
            xy_tol=0.6,          # clustertolerantie (mm)
            d_tol=0.6,           # diameter-tolerantie (mm)
            margin_ratio=0.1     # snedes niet vlak tegen huid
        )
        holes_detected = len(holes)

        # 8) Upload STL
        stl_public_url = upload_to_supabase(stl_path, file.filename.replace(".step", ".stl"))

        # 9) Wegschrijven in Supabase DB
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "units": "mm",
                "volume_mm3": volume_mm3,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "holes_detected": holes_detected,
                "holes_data": holes,            # JSONB
                "inner_contours": inner_contours,  # JSONB
                "model_url": stl_public_url,
                "created_at": "now()",
            }).execute()

        # 10) Response
        return JSONResponse(content={
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": round(volume_mm3, 3) if volume_mm3 else None,
            "filename": file.filename,
            "holesDetected": holes_detected,
            "holes": holes,
            "innerContours": inner_contours,
            "modelURL": stl_public_url,
        })

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        # opruimen
        for path in [tmp_path, stl_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

# -------------------------------
# 🌐 Proxy voor CORS (Lovable)
# -------------------------------
@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    url = f"https://{SUPABASE_URL.split('//')[-1]}/storage/v1/object/public/{path}" if SUPABASE_URL else \
          f"https://sywnjytfygvotskufvzs.supabase.co/storage/v1/object/public/{path}"
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
