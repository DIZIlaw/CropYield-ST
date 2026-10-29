"""
CropYield-ST: Pydantic Schemas for REST API Validation
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    torch_available: bool
    device: str
    sample_data_loaded: bool


class DashboardSummary(BaseModel):
    total_parcels: int
    total_observations: int
    latest_observation_date: Optional[str]
    trained_models_count: int
    experiments_count: int
    mean_predicted_yield: Optional[float]
    mean_actual_yield: Optional[float]
    provenance: str
    is_demonstration: bool
    disclaimer: str


class ParcelMetadata(BaseModel):
    parcel_id: str
    crop_type: Optional[str] = "Wheat"
    area_ha: Optional[float] = 10.0
    soil_quality: Optional[float] = 0.8
    centroid_lat: float
    centroid_lon: float
    neighbors_count: int
    predicted_yield: Optional[float] = None
    actual_yield: Optional[float] = None
    geometry: Dict[str, Any]


class ParcelDetail(ParcelMetadata):
    neighbors: List[str]
    observation_count: int
    optical_availability_pct: float
    sar_availability_pct: float


class TimeSeriesObservation(BaseModel):
    date: str
    timestep: int
    ndvi: float
    evi: Optional[float] = None
    ndre: Optional[float] = None
    ndwi: Optional[float] = None
    sar_vv_db: float
    sar_vh_db: float
    sar_vh_vv_ratio: float
    sar_rvi: float
    cloud_cover_percent: float
    cloud_masked: bool
    optical_valid: bool


class ParcelTimeSeriesResponse(BaseModel):
    parcel_id: str
    observations: List[TimeSeriesObservation]
    provenance: str


class TrainRequest(BaseModel):
    model_type: str = Field(default="st_gnn", description="st_gnn, pure_temporal_rnn, or xgboost")
    feature_set: str = Field(default="fusion", description="fusion, optical_only, or sar_only")
    epochs: int = Field(default=35, ge=5, le=200)
    learning_rate: float = Field(default=0.005, gt=0.0)
    hidden_dim: int = Field(default=64, ge=16, le=256)
    dropout: float = Field(default=0.2, ge=0.0, le=0.7)
    random_seed: int = Field(default=42)


class PredictRequest(BaseModel):
    parcel_id: str
    model_type: str = Field(default="st_gnn")
    feature_set: str = Field(default="fusion")
    crop_type: Optional[str] = None


class PredictResponse(BaseModel):
    parcel_id: str
    predicted_yield_tha: float
    actual_yield_tha: Optional[float] = None
    label_type: Optional[str] = "synthetic_reference"
    crop_type: str
    model_used: str
    model_version: Optional[str] = "st_gnn_v1.0"
    unit: Optional[str] = "tonnes/hectare"
    feature_set: str
    latest_observation_date: str
    data_availability: str
    is_demonstration: bool
    provenance: str
    neighbors: List[str]
    confidence_interval: List[float]
    uncertainty: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = Field(default_factory=list)
