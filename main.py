from fastapi import FastAPI
import cadquery as cq
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"message": "STEP analyzer live ✅ (CadQuery mode)"}

@app.post("/analyze/")
def analyze_step(file_url: str):
    # voorbeeldanalyse
    try:
        model = cq.importers.importStep(file_url)
        bbox = model.val().BoundingBox()
        dims = {
            "x": round(bbox.xlen, 2),
            "y": round(bbox.ylen, 2),
            "z": round(bbox.zlen, 2)
        }
        return {"status": "ok", "bounding_box_mm": dims}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
