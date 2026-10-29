"""
CropYield-ST: Main FastAPI Application Entry Point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import (
    health,
    dashboard,
    parcels,
    predict,
    models_api,
    experiments,
    evaluation_api,
    datasets
)

app = FastAPI(
    title="CropYield-ST API",
    description="Multi-Spectral Satellite & SAR Fusion for Hyperlocal Crop Yield Forecasting Using Spatio-Temporal Graph Neural Networks",
    version="1.0.0"
)

# Configure CORS for local development with Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(parcels.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(models_api.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(evaluation_api.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")


@app.get("/api")
def api_info():
    return {
        "project": "CropYield-ST",
        "description": "Multi-Spectral Satellite & SAR Fusion for Hyperlocal Crop Yield Forecasting Using Spatio-Temporal Graph Neural Networks",
        "course": "Artificial Intelligence — B.Tech CSE, Fall Semester 2026–27",
        "team": [
            {"name": "Aditya Pandey", "registration_no": "24BCE2830"},
            {"name": "Rishav Raj", "registration_no": "24BCE2093"},
            {"name": "Divyansh Singh", "registration_no": "24BCE2821"}
        ],
        "api_docs": "/docs",
        "health_check": "/api/health"
    }


# Serve built React frontend if available for unified, single-port zero-Node execution
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "index.html")):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Serve exact file if it exists (e.g. favicon.svg, icons.svg)
        candidate = os.path.join(frontend_dist, full_path)
        if full_path and os.path.exists(candidate) and os.path.isfile(candidate):
            return FileResponse(candidate)
        # Otherwise serve index.html for React SPA client-side routing
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    def root():
        return api_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
