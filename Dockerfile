FROM python:3.11-slim

# System deps for OpenCV and rawpy (libraw)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libraw-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT at runtime (default 8080)
EXPOSE 8080

CMD ["python3", "main.py"]
