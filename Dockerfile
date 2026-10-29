# CropYield-ST: Universal Zero-Installation Container
# Runs seamlessly on Windows, macOS (Intel & Apple Silicon), and Linux
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

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
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "run_app.py"]
