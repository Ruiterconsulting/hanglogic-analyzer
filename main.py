# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cadquery as cq
import tempfile
import os
import traceback

app = FastAPI(title="HangLogic Analyzer API", version="1.0.0")

# CORS voor je frontend/Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # zet hier je duidelijke domeinen als je wilt beperken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery mode)"}

@app.get("/health")
def health():
    return {"status": "ok"}

def _ensure_step_extension(filename: str):
    ext = (os.path.splitext(filename)[1] or "").lower()
    if ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Please upload .STEP or .STP"
        )

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    """
    Ontvangt een .STEP/.STP bestand, laadt het met CadQuery,
    en geeft bounding box dimensies in millimeters terug.
    """
    try:
        # 1) Validatie bestandstype
        _ensure_step_extension(file.filename)

        # 2) Sla tijdelijk op (CadQuery leest vanaf pad)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 3) Importeren met CadQuery
        #    cq.importers.importStep retourneert een Workplane/Shape-collectie
        model = cq.importers.importStep(tmp_path)

        # Zorg dat we een valide shape hebben
        try:
            shape = model.val()
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Could not derive a valid shape from the STEP file."
            )

        # 4) Bounding box bepalen (CadQuery is in mm)
        bbox = shape.BoundingBox()
        dims = {
            "X": round(float(bbox.xlen), 3),
            "Y": round(float(bbox.ylen), 3),
            "Z": round(float(bbox.zlen), 3),
        }

        # 5) Optioneel: volume (kan bij dunne platen 0 zijn als niet solide)
        try:
            volume_mm3 = float(shape.Volume())  # mm³
        except Exception:
            volume_mm3 = None

        # 6) Opruimen temp
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # --- Save result to Supabase ---
    try:
        supabase.table("analyzed_parts").insert({
            "filename": filename,
            "dimensions": {
                "x": bbox["X"],
                "y": bbox["Y"],
                "z": bbox["Z"]
            },
            "volume_mm3": volume,
            "units": "mm",
            "holes_detected": 0  # (later wordt dit automatisch bepaald)
        }).execute()
    except Exception as e:
        print("⚠️ Error saving to Supabase:", e)

        
        return JSONResponse(
            content={
                "status": "success",
                "units": "mm",
                "boundingBoxMM": dims,
                "volumeMM3": round(volume_mm3, 3) if volume_mm3 is not None else None,
                "filename": file.filename,
            }
        )

    except HTTPException as he:
        # Bekende, nette fout (bijv. wrong filetype)
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        # Onbekende fout → log stacktrace en geef generieke melding
        tb = traceback.format_exc(limit=3)
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}", "trace": tb},
        )
