# Gebruik Python 3.10 met ingebouwde OpenGL dependencies
FROM debian:bullseye-slim

# Installeer Python handmatig zodat pythonocc-core werkt
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev \
    libgl1 libglu1-mesa libxrender1 libxext6 libsm6 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Installeer dependencies (geen cache = sneller builden)
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 10000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
