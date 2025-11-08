FROM python:3.11-slim

# System libs for pythonocc
RUN apt-get update && apt-get install -y \
    libgl1 libxrender1 libxext6 libx11-6 libglu1-mesa libfreetype6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
