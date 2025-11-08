# Gebruik een basisimage dat CadQuery en OpenCascade aankan
FROM python:3.11-slim

# Voorkom interactiviteit
ENV DEBIAN_FRONTEND=noninteractive

# Installeer systeemdependencies die CadQuery/OpenCascade nodig heeft
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Werkmap
WORKDIR /app

# Kopieer requirements en installeer Python-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer de app
COPY main.py .

# Expose de poort voor Render
EXPOSE 8080

# Start de FastAPI-app met Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
