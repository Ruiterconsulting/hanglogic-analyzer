# Gebruik een volledige image met meer systeemondersteuning
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Installeer Linux dependencies die CadQuery / OpenCascade nodig hebben
RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stel de werkdirectory in
WORKDIR /app

# Kopieer de vereisten
COPY requirements.txt .

# Installeer Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer de applicatie
COPY main.py .

# Poort openstellen
EXPOSE 8080

# Start de FastAPI-server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
