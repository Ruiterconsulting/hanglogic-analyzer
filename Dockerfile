# Gebruik Python 3.10 zodat pythonocc-core werkt
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# Installeer systeempakketten die nodig zijn voor OpenCascade
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Installeer Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
