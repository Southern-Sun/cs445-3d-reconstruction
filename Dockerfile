# Use an official PyTorch image with CUDA runtime.
# Check https://hub.docker.com/r/pytorch/pytorch/tags and pick a recent CUDA tag.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps (git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (layer cache friendly)
COPY requirements.txt .

# Install deps; install PyTorch extras only if needed (base already has it)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY app ./app

# Environment for FastAPI / uvicorn
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
# RunPod will usually set PORT; default to 8000 if not set
ENV PORT=8000

# Expose internal port (RunPod will map this externally)
EXPOSE 8000

# Start uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
