"""
Unit tests for Real Satellite Data Ingestion Pipeline & Provider Diagnostic Checks.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.data.real_data_pipeline import RealDataPipeline, real_data_pipeline
from backend.app.data.storage import storage

client = TestClient(app)


def test_provider_status_structure():
    """Verify check_satellite_providers returns structured diagnostics."""
    pipeline = RealDataPipeline()
    status = pipeline.check_satellite_providers()
    
    assert "real_data_ready" in status
    assert "active_mode" in status
    assert "provenance" in status
    assert "providers" in status
    assert "google_earth_engine" in status["providers"]
    assert "copernicus_data_space" in status["providers"]
    assert "public_stac" in status["providers"]


def test_real_ingestion_without_credentials_fails_honestly():
    """Verify that attempting real ingestion without credentials raises PermissionError, refusing fake data."""
    pipeline = RealDataPipeline()
    # Ensure no credentials set in test runner
    orig_gee = pipeline.check_satellite_providers()
    if not orig_gee["real_data_ready"]:
        with pytest.raises(PermissionError, match="No authenticated provider found"):
            pipeline.ingest_real_satellite_observations(
                gdf=storage.gdf,
                start_date="2025-10-15",
                end_date="2026-04-15"
            )


def test_api_providers_endpoint():
    """Verify GET /api/datasets/providers endpoint."""
    response = client.get("/api/datasets/providers")
    assert response.status_code == 200
    data = response.json()
    assert "real_data_ready" in data
    assert "provenance" in data


def test_api_real_ingestion_endpoint_diagnostics():
    """Verify POST /api/datasets/real-ingestion returns honest 403 when unauthenticated."""
    providers = client.get("/api/datasets/providers").json()
    if not providers["real_data_ready"]:
        response = client.post(
            "/api/datasets/real-ingestion",
            json={
                "start_date": "2025-10-15",
                "end_date": "2026-04-15",
                "cloud_max_percent": 30.0,
                "study_area_name": "test_area"
            }
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        assert "No authenticated provider found" in detail
