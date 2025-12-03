# Use an official PyTorch image with CUDA runtime
# See https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY app ./app

# Environment for FastAPI / uvicorn
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose internal port
EXPOSE 8000

# Start uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
