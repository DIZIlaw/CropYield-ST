"""
Dashboard Summary Endpoints
"""
from fastapi import APIRouter
from backend.app.schemas.models import DashboardSummary
from backend.app.data.storage import storage

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
def get_dashboard_summary():
    return storage.get_summary()
