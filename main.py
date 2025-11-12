from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cadquery as cq
import tempfile
import os, math, traceback
from supabase import create_client, Client
import httpx

app = FastAPI(title="HangLogic Analyzer API", version="1.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase.")
        try:
            buckets = supabase.storage.list_buckets()
            print("📦 Buckets:", [b.name for b in buckets])
        except Exception as e:
            print("⚠️ Could not list buckets:", e)
    except Exception as e:
        print("⚠️ Could not initialize Supabase client:", e)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY missing in environment.")

# ---------- helpers ----------

def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP",
        )

def upload_public(bucket: str, local_path: str, remote_path: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase client not initialized.")
    sto = supabase.storage.from_(bucket)
    # upsert via remove-if-exists
    try:
        files = sto.list(path=os.path.dirname(remote_path) or "")
        for f in files:
            name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            if name == os.path.basename(remote_path):
                sto.remove([remote_path])
                break
    except Exception:
        pass
    with open(local_path, "rb") as fh:
        sto.upload(remote_path, fh)
    return sto.get_public_url(remote_path)

# -------------- inner-contour → groene patches --------------

def _face_normal(face: cq.Face) -> cq.Vector:
    # robuuste normale: middelpunt van boundbox
    bb = face.BoundingBox()
    u = 0.5
    try:
        # CQ 2.4: normalAt(u, v=None)
        n = face.normalAt(u)
    except TypeError:
        # oudere signatuur: normalAt(u, v)
        n = face.normalAt(u, 0.5)
    return n

def _wire_center_approx(wire: cq.Wire):
    vs = [v.toTuple() for v in wire.Vertices()]
    if not vs:
        return (0.0, 0.0, 0.0)
    cx = sum(p[0] for p in vs) / len(vs)
    cy = sum(p[1] for p in vs) / len(vs)
    cz = sum(p[2] for p in vs) / len(vs)
    return (cx, cy, cz)

def build_green_patches_from_inner_wires(shape: cq.Shape, thickness: float = 0.8):
    """
    Voor elk vlak pakt dit de inner wires (binnencontouren) en
    maakt een dun groen 'patchje' door de contour te extruderen.
    Werkt ook bij non-planar of niet-perfecte wires.
    """
    green_solids = []
    total_wires = 0
    failed = 0

    for face in shape.Faces():
        wires = face.Wires()
        if len(wires) <= 1:
            continue
        inner = wires[1:]
        total_wires += len(inner)
        n = _face_normal(face)
        if n.Length == 0:
            continue
        n = n.normalized()

        for w in inner:
            try:
                # probeer normale face-build
                patch_face = cq.Face.makeFromWires(w)
                patch_solid = patch_face.extrude(n * thickness)
                green_solids.append(patch_solid)
            except Exception:
                try:
                    # fallback: maak cirkelpatch via gemiddelde center + radius
                    pts = [v.toTuple() for v in w.Vertices()]
                    if len(pts) >= 3:
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = sum(p[1] for p in pts) / len(pts)
                        cz = sum(p[2] for p in pts) / len(pts)
                        # schatting radius
                        r = sum(math.dist(p, (cx, cy, cz)) for p in pts) / len(pts)
                        patch = cq.Workplane("XY").moveTo(cx, cy).circle(r).extrude(thickness)
                        green_solids.append(patch.val())
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    continue

    print(f"✅ Found {total_wires} inner wires; built {len(green_solids)} green patches; failed {failed}")
    if not green_solids:
        return None
    return cq.Compound.makeCompound(green_solids)

# -------------- endpoint basics --------------

@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (v1.8.0 — GLTF with green inner patches)"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------- main analyze --------------

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    tmp_path = None
    try:
        _ensure_step_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # STEP → shape
        model = cq.importers.importStep(tmp_path)
        shape = model.val()
        print(f"🧠 Analyzing: {file.filename}")

        # dims & volume
        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3),
        }
        try:
            volume = round(float(shape.Volume()), 3)
        except Exception:
            volume = None

        # bouw groene patches
        green = build_green_patches_from_inner_wires(shape, thickness=0.8)

        # bouw GLTF scene: blauw body + groen patches
        asm = cq.Assembly()
        asm.add(shape, name="body", color=cq.Color(0.02, 0.06, 0.28))  # donkerblauw
        if green:
            asm.add(green, name="inner_patches", color=cq.Color(0.05, 0.85, 0.2))  # felgroen

        # export STL (legacy) + GLTF (visueel met groen)
        stl_path = tmp_path.replace(".step", ".stl")
        gltf_path = tmp_path.replace(".step", ".gltf")
        cq.exporters.export(shape, stl_path, "STL")
        cq.exporters.export(asm, gltf_path, "GLTF")

        # upload
        base = os.path.splitext(file.filename)[0]
        stl_url = upload_public("cad-models", stl_path, f"analyzed/{base}.stl")
        gltf_url = upload_public("cad-models", gltf_path, f"analyzed/{base}.gltf")

        # sla meta op
        if supabase:
            supabase.table("analyzed_parts").insert({
                "filename": file.filename,
                "dimensions": {"x": dims["X"], "y": dims["Y"], "z": dims["Z"]},
                "units": "mm",
                "volume_mm3": volume,
                "bounding_box_x": dims["X"],
                "bounding_box_y": dims["Y"],
                "bounding_box_z": dims["Z"],
                "model_url": stl_url,
                "model_url_gltf": gltf_url,
                "created_at": "now()",
            }).execute()

        return JSONResponse({
            "status": "success",
            "units": "mm",
            "boundingBoxMM": dims,
            "volumeMM3": volume,
            "filename": file.filename,
            "modelURL": stl_url,
            "modelURL_gltf": gltf_url
        })

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}", "trace": tb})
    finally:
        if tmp_path:
            for p in (tmp_path, tmp_path.replace(".step", ".stl"), tmp_path.replace(".step", ".gltf"), tmp_path.replace(".step", ".bin")):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# --------- public proxy (blijft handig) ---------

@app.get("/proxy/{path:path}")
async def proxy_file(path: str):
    url = f"https://{SUPABASE_URL.split('//')[-1]}/storage/v1/object/public/{path}"
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
