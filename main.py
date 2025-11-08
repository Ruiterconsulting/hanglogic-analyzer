from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
import tempfile
import uuid

# ✅ Supabase keys (je mag hier je eigen projectgegevens invullen)
SUPABASE_URL = "https://sywnjytfygvotskufvzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d25qeXRmeWd2b3Rza3VmdnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1ODAzMTAsImV4cCI6MjA3ODE1NjMxMH0.rwdyRnjOAG5pUrPufoZL13_O0HAQhuP8E2O_Al2kqMY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="HangLogic Analyzer API")

# ✅ CORS – zodat Lovable frontend contact mag maken
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "HangLogic Analyzer API running!"}

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Dummy analyzer – leest bestand, berekent grootte en slaat het resultaat op in Supabase
    """
    try:
        # Sla tijdelijk bestand op
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Simpele 'analyse'
        file_size_kb = round(len(content) / 1024, 2)
        file_ext = os.path.splitext(file.filename)[1].lower()
        part_id = str(uuid.uuid4())

        result = {
            "id": part_id,
            "file_name": file.filename,
            "file_type": file_ext,
            "size_kb": file_size_kb,
        }

        # Sla resultaat op in Supabase
        data, count = supabase.table("analyzed_parts").insert(result).execute()

        # Opruimen
        os.remove(tmp_path)

        return {
            "status": "success",
            "analyzed": result,
            "supabase_response": data,
        }

    except Exception as e:
        return {"status": "error", "details": str(e)}
