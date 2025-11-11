# ============================================================
# 🧰 Base image
# ============================================================
FROM python:3.11-slim

# ============================================================
# ⚙️ System dependencies for CadQuery / OCC / Shapely / Trimesh
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    libfreetype6 \
    libxft2 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 📦 Set working directory
# ============================================================
WORKDIR /app

# ============================================================
# 📥 Copy requirements
# ============================================================
COPY requirements.txt .

# ============================================================
# 📦 Install Python dependencies
# ============================================================
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ============================================================
# 📂 Copy application files
# ============================================================
COPY . .

# ============================================================
# 🌍 Environment variables (for Render)
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# ============================================================
# 🚀 Run the FastAPI app
# ============================================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
