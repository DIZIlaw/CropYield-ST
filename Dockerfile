# CropYield-ST: Universal Zero-Installation Container
# Runs seamlessly on Render, Hugging Face, Linux, macOS, and Windows
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install system dependencies for geospatial & scientific wheels (including OpenMP for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch first (saves ~2GB vs default CUDA wheel), then requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-compiled frontend distribution
COPY backend/ backend/
COPY data/ data/
COPY configs/ configs/
COPY experiments/ experiments/
COPY models/ models/
COPY scripts/ scripts/
COPY src/ src/
COPY frontend/dist/ frontend/dist/
COPY run_app.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

CMD ["python", "run_app.py"]
