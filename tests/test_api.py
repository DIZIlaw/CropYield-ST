"""
Unit tests for FastAPI REST API endpoints.
"""
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["torch_available"] is True


def test_dashboard_summary():
    """Verify GET /api/dashboard/summary."""
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_parcels" in data
    assert data["total_parcels"] > 0
    assert data["is_demonstration"] is True


def test_list_parcels():
    """Verify GET /api/parcels."""
    response = client.get("/api/parcels")
    assert response.status_code == 200
    parcels = response.json()
    assert isinstance(parcels, list)
    assert len(parcels) > 0
    assert "parcel_id" in parcels[0]
    assert "centroid_lat" in parcels[0]


def test_parcel_detail_and_neighbors():
    """Verify GET /api/parcels/{id} and /api/parcels/{id}/neighbors."""
    p_res = client.get("/api/parcels")
    first_pid = p_res.json()[0]["parcel_id"]
    
    det_res = client.get(f"/api/parcels/{first_pid}")
    assert det_res.status_code == 200
    detail = det_res.json()
    assert detail["parcel_id"] == first_pid
    assert "neighbors" in detail
    
    nbr_res = client.get(f"/api/parcels/{first_pid}/neighbors")
    assert nbr_res.status_code == 200
    assert "neighbors_count" in nbr_res.json()


def test_predict_endpoint():
    """Verify POST /api/predict."""
    p_res = client.get("/api/parcels")
    first_pid = p_res.json()[0]["parcel_id"]
    
    payload = {
        "parcel_id": first_pid,
        "model_type": "st_gnn",
        "feature_set": "fusion",
        "crop_type": "Wheat - HD 2967"
    }
    
    pred_res = client.post("/api/predict", json=payload)
    assert pred_res.status_code == 200
    data = pred_res.json()
    assert data["parcel_id"] == first_pid
    assert "predicted_yield_tha" in data
    assert data["predicted_yield_tha"] > 0.0
    assert len(data["confidence_interval"]) == 2


def test_models_catalog():
    """Verify GET /api/models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert any(m["id"] == "st_gnn" for m in models)
