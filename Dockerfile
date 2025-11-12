# Gebruik een stabiele base image
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Installeer systeemafhankelijkheden voor CadQuery en OpenCascade
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    libfreetype6 \
    libjpeg-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Werkdirectory
WORKDIR /app

# Kopieer dependencies
COPY requirements.txt .

# Installeer Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer app
COPY main.py .

# Poort openstellen
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
