import cadquery as cq

@app.post("/analyze")
async def analyze_step(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Gebruik cadquery om de STEP in te lezen
        shape = cq.importers.importStep(tmp_path)

        # Bereken bounding box
        bb = shape.val().BoundingBox()
        result = {
            "id": str(uuid.uuid4()),
            "file_name": file.filename,
            "bounding_box_mm": {
                "x": round(bb.xlen, 2),
                "y": round(bb.ylen, 2),
                "z": round(bb.zlen, 2),
            },
        }

        supabase.table("analyzed_parts").insert(result).execute()
        os.remove(tmp_path)
        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "details": str(e)}
