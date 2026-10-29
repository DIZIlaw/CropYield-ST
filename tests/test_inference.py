"""
Unit tests for CropYield-ST InferenceService and genuine PyTorch neural execution.
"""
import pytest
import numpy as np
import torch

from backend.app.models.inference import InferenceService
from backend.app.data.storage import storage


def test_inference_service_initialization():
    """Verify InferenceService initializes and contains active model."""
    service = InferenceService()
    assert service.model is not None
    assert service.active_model_type in ("st_gnn", "pure_temporal_rnn")
    assert service.conformal_q95 > 0.0


def test_inference_service_predict_parcel_st_gnn():
    """Verify authentic forward inference using CropYieldSTGNN."""
    service = InferenceService()
    p_id = storage.gdf["parcel_id"].iloc[0]
    
    res = service.predict_parcel(
        parcel_id=p_id,
        timeseries_df=storage.timeseries_df,
        gdf=storage.gdf,
        graph_data=storage.graph_data,
        labels_df=storage.labels_df,
        model_type="st_gnn",
        feature_set="fusion"
    )
    
    assert res["parcel_id"] == p_id
    assert isinstance(res["predicted_yield_tha"], float)
    assert 0.1 <= res["predicted_yield_tha"] <= 12.0
    assert res["unit"] == "tonnes/hectare"
    assert res["model_used"] == "st_gnn"
    assert len(res["confidence_interval"]) == 2
    assert res["confidence_interval"][0] <= res["predicted_yield_tha"] <= res["confidence_interval"][1]
    assert "uncertainty" in res
    assert res["uncertainty"]["interval_type"] in ("conformal_prediction_interval", "split_conformal_prediction_interval")
    assert "empirical_test_coverage_pct" in res["uncertainty"]
    assert "calibration_sample_size" in res["uncertainty"]
    assert res["label_type"] in ("synthetic_reference", "measured_harvest")


def test_inference_service_predict_parcel_pure_rnn():
    """Verify inference using PureTemporalRNN baseline (no graph convolution)."""
    service = InferenceService()
    p_id = storage.gdf["parcel_id"].iloc[1]
    
    res = service.predict_parcel(
        parcel_id=p_id,
        timeseries_df=storage.timeseries_df,
        gdf=storage.gdf,
        graph_data=storage.graph_data,
        labels_df=storage.labels_df,
        model_type="pure_temporal_rnn",
        feature_set="optical_only"
    )
    
    assert res["parcel_id"] == p_id
    assert res["model_used"] == "pure_temporal_rnn"
    assert res["predicted_yield_tha"] > 0.0


def test_inference_service_xgboost_baseline():
    """Verify inference using XGBoost tabular baseline."""
    service = InferenceService()
    p_id = storage.gdf["parcel_id"].iloc[2]
    
    res = service.predict_parcel(
        parcel_id=p_id,
        timeseries_df=storage.timeseries_df,
        gdf=storage.gdf,
        graph_data=storage.graph_data,
        labels_df=storage.labels_df,
        model_type="xgboost",
        feature_set="fusion"
    )
    
    assert res["parcel_id"] == p_id
    assert res["model_used"] == "xgboost"
    assert res["predicted_yield_tha"] > 0.0


def test_inference_nonexistent_parcel_raises():
    """Verify invalid parcel ID raises an informative ValueError."""
    service = InferenceService()
    with pytest.raises(ValueError, match="not found in active study area"):
        service.predict_parcel(
            parcel_id="NON_EXISTENT_999",
            timeseries_df=storage.timeseries_df,
            gdf=storage.gdf,
            graph_data=storage.graph_data
        )
