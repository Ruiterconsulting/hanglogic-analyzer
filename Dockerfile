FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

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
    curl wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Gebruik modern pip-resolver met fallback
RUN pip install --upgrade pip setuptools wheel
RUN PIP_CONSTRAINT="" pip install --no-cache-dir -r requirements.txt

COPY main.py .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
