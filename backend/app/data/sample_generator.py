"""
CropYield-ST: Deterministic Geospatial and Multi-Sensor Sample Data Generator
Generates realistic contiguous parcel polygons, multi-temporal optical & SAR time series,
and calibrated ground truth yield labels for any geographic region or preset.
"""
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping

from backend.app.features.indices import compute_ndvi, compute_evi, compute_ndre, compute_ndwi
from backend.app.features.sar_features import compute_all_sar_features


# Built-in Regional Presets covering diverse global agricultural zones
REGION_PRESETS: Dict[str, Dict[str, Any]] = {
    "punjab": {
        "id": "punjab",
        "name": "Punjab, India (Ludhiana Wheat Belt)",
        "lat": 30.9010,
        "lon": 75.8573,
        "description": "High-intensity irrigated Indo-Gangetic plain. Major rabi wheat & kharif rice rotation.",
        "crops": [
            ("Wheat - HD 2967", 5.2, 0.4),
            ("Wheat - PBW 550", 4.8, 0.35),
            ("Wheat - DBW 187", 5.6, 0.45),
            ("Mustard - Pusa Bold", 2.4, 0.25)
        ]
    },
    "haryana": {
        "id": "haryana",
        "name": "Haryana, India (Karnal Rice-Wheat Belt)",
        "lat": 29.6857,
        "lon": 76.9905,
        "description": "Fertile alluvial plains with extensive tube-well irrigation and high crop vigor.",
        "crops": [
            ("Basmati Rice - Pusa 1121", 4.4, 0.35),
            ("Wheat - DBW 303", 5.4, 0.40),
            ("Wheat - HD 3226", 5.1, 0.35),
            ("Mustard - Giriraj", 2.5, 0.22)
        ]
    },
    "madhya_pradesh": {
        "id": "madhya_pradesh",
        "name": "Madhya Pradesh, India (Malwa Plateau)",
        "lat": 23.1765,
        "lon": 75.7885,
        "description": "Black cotton soil (Vertisols) plateau with high water retention and soybean-wheat systems.",
        "crops": [
            ("Sharbati Wheat - C-306", 4.2, 0.35),
            ("Soybean - JS 335", 2.6, 0.25),
            ("Chickpea - JG 11", 2.2, 0.20),
            ("Durum Wheat - HI 8663", 4.9, 0.38)
        ]
    },
    "iowa_usa": {
        "id": "iowa_usa",
        "name": "Iowa, USA (Midwest Corn & Soybean Belt)",
        "lat": 41.5868,
        "lon": -93.6250,
        "description": "Large-scale rainfed Mollisols prairie soil with high organic matter and mechanized farming.",
        "crops": [
            ("Field Corn / Maize", 11.2, 0.8),
            ("Soybean - Pioneer", 3.8, 0.3),
            ("Winter Wheat", 4.5, 0.4),
            ("Sweet Corn", 9.8, 0.6)
        ]
    },
    "california_usa": {
        "id": "california_usa",
        "name": "California, USA (Central Valley)",
        "lat": 36.7468,
        "lon": -119.7726,
        "description": "Mediterranean climate with intensive canal and drip irrigation systems.",
        "crops": [
            ("Processing Tomato", 75.0, 6.0),
            ("Alfalfa Hay", 16.5, 1.2),
            ("Durum Wheat", 6.2, 0.5),
            ("Pima Cotton", 3.4, 0.3)
        ]
    },
    "france": {
        "id": "france",
        "name": "Nouvelle-Aquitaine, France (Cereal Plain)",
        "lat": 44.8378,
        "lon": -0.5792,
        "description": "Temperate maritime cereal plains with soft winter wheat and rapeseed rotations.",
        "crops": [
            ("Soft Winter Wheat", 7.4, 0.5),
            ("Grain Maize", 10.2, 0.7),
            ("Winter Barley", 6.8, 0.4),
            ("Rapeseed / Canola", 3.6, 0.3)
        ]
    }
}


def generate_synthetic_agricultural_study_area(
    n_rows: int = 6,
    n_cols: int = 6,
    base_lat: float = 30.9010,
    base_lon: float = 75.8573,
    field_width_deg: float = 0.0045,  # ~450 meters
    field_height_deg: float = 0.0035, # ~350 meters
    random_seed: int = 42,
    region_name: str = "Punjab Wheat Belt",
    custom_crops: Optional[List[Tuple[str, float, float]]] = None
) -> gpd.GeoDataFrame:
    """
    Generates n_rows * n_cols contiguous agricultural field polygons.
    Fields share borders so that standard topological adjacency (touches) creates a real planar graph.
    """
    np.random.seed(random_seed)
    features = []
    
    crop_varieties = custom_crops or [
        ("Wheat - HD 2967", 5.2, 0.4),
        ("Wheat - PBW 550", 4.8, 0.35),
        ("Wheat - DBW 187", 5.6, 0.45),
        ("Mustard - Pusa Bold", 2.4, 0.25)
    ]
    
    parcel_count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            parcel_count += 1
            pid = f"PARCEL_{parcel_count:03d}"
            
            # Field coordinates with slight realistic organic jitter at shared boundary
            min_lon = base_lon + c * field_width_deg
            max_lon = min_lon + field_width_deg
            min_lat = base_lat + r * field_height_deg
            max_lat = min_lat + field_height_deg
            
            poly = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat)
            ])
            
            # Assign crop variety
            variety_idx = (r * 2 + c) % len(crop_varieties)
            crop_name, base_yld, std_yld = crop_varieties[variety_idx]
            area_ha = round(float(poly.area * (111000 ** 2) / 10000.0), 2)
            soil_quality = round(float(0.7 + 0.3 * np.sin(r * 0.8) * np.cos(c * 0.8) + np.random.normal(0, 0.05)), 2)
            soil_quality = float(np.clip(soil_quality, 0.4, 1.0))
            
            features.append({
                "parcel_id": pid,
                "crop_type": crop_name,
                "area_ha": area_ha,
                "soil_quality": soil_quality,
                "row_idx": r,
                "col_idx": c,
                "region_name": region_name,
                "geometry": poly
            })
            
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    return gdf


def generate_multitemporal_satellite_timeseries(
    gdf: gpd.GeoDataFrame,
    start_date: str = "2025-11-01",
    n_timesteps: int = 14,
    step_days: int = 12,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates multi-sensor time-series (Sentinel-2 optical + Sentinel-1 SAR)
    simulating a growing season with realistic phenology curves.
    Includes simulated cloud cover days to illustrate SAR cloud-penetration value!
    """
    np.random.seed(random_seed)
    
    dates = pd.date_range(start=start_date, periods=n_timesteps, freq=f"{step_days}D")
    records = []
    
    for idx, row in gdf.iterrows():
        pid = row["parcel_id"]
        soil = row["soil_quality"]
        crop = row["crop_type"]
        
        # Base canopy potential
        peak_factor = 0.85 * soil + np.random.normal(0, 0.03)
        peak_factor = float(np.clip(peak_factor, 0.5, 0.95))
        
        for t, dt in enumerate(dates):
            date_str = dt.strftime("%Y-%m-%d")
            tau = t / (n_timesteps - 1)
            
            # Asymmetric phenological greenness curve
            greenness = np.exp(-((tau - 0.55) ** 2) / 0.05)
            biomass = 1.0 / (1.0 + np.exp(-12.0 * (tau - 0.35)))
            
            # Simulated cloudy timesteps
            is_cloudy = (t in (4, 8)) and (np.random.rand() > 0.15)
            cloud_pct = float(np.random.uniform(70.0, 95.0)) if is_cloudy else float(np.random.uniform(0.0, 12.0))
            
            # Optical bands (Surface Reflectance scale 0.0 - 1.0)
            if is_cloudy:
                b4_red = float(np.clip(0.35 + np.random.normal(0, 0.05), 0.25, 0.5))
                b8_nir = float(np.clip(0.40 + np.random.normal(0, 0.05), 0.30, 0.55))
                b2_blue = float(np.clip(0.30 + np.random.normal(0, 0.04), 0.20, 0.45))
                b5_rededge = float(np.clip(0.38 + np.random.normal(0, 0.04), 0.28, 0.50))
                b11_swir = float(np.clip(0.32 + np.random.normal(0, 0.04), 0.22, 0.48))
                opt_valid = False
            else:
                b4_red = float(np.clip(0.18 - 0.14 * greenness * peak_factor + np.random.normal(0, 0.008), 0.02, 0.25))
                b8_nir = float(np.clip(0.20 + 0.55 * greenness * peak_factor + np.random.normal(0, 0.015), 0.15, 0.85))
                b2_blue = float(np.clip(0.12 - 0.07 * greenness * peak_factor + np.random.normal(0, 0.006), 0.02, 0.18))
                b5_rededge = float(np.clip(0.16 + 0.35 * greenness * peak_factor + np.random.normal(0, 0.012), 0.10, 0.65))
                b11_swir = float(np.clip(0.22 - 0.10 * greenness * peak_factor + np.random.normal(0, 0.010), 0.08, 0.30))
                opt_valid = True
                
            # Optical indices
            ndvi_val = float(compute_ndvi(np.array([b8_nir]), np.array([b4_red]))[0])
            evi_val = float(compute_evi(np.array([b8_nir]), np.array([b4_red]), np.array([b2_blue]))[0])
            ndre_val = float(compute_ndre(np.array([b8_nir]), np.array([b5_rededge]))[0])
            ndwi_val = float(compute_ndwi(np.array([b8_nir]), np.array([b11_swir]))[0])
            
            # SAR features (C-band Sentinel-1 radar)
            base_vv_lin = 0.04 + 0.08 * biomass * peak_factor + np.random.normal(0, 0.005)
            base_vh_lin = 0.005 + 0.035 * biomass * peak_factor + np.random.normal(0, 0.002)
            
            vv_lin = float(np.clip(base_vv_lin, 0.01, 0.25))
            vh_lin = float(np.clip(base_vh_lin, 0.002, 0.10))
            
            sar_dict = compute_all_sar_features(np.array([vv_lin]), np.array([vh_lin]), is_db=False)
            
            records.append({
                "parcel_id": pid,
                "date": date_str,
                "timestep": t,
                "b2_blue": round(b2_blue, 4),
                "b4_red": round(b4_red, 4),
                "b5_rededge": round(b5_rededge, 4),
                "b8_nir": round(b8_nir, 4),
                "b11_swir": round(b11_swir, 4),
                "ndvi": round(ndvi_val, 4),
                "evi": round(evi_val, 4),
                "ndre": round(ndre_val, 4),
                "ndwi": round(ndwi_val, 4),
                "sar_vv_linear": round(vv_lin, 5),
                "sar_vh_linear": round(vh_lin, 5),
                "sar_vv_db": round(float(sar_dict["vv_db"][0]), 2),
                "sar_vh_db": round(float(sar_dict["vh_db"][0]), 2),
                "sar_vh_vv_ratio": round(float(sar_dict["vh_vv_ratio"][0]), 4),
                "sar_rvi": round(float(sar_dict["rvi"][0]), 4),
                "cloud_cover_percent": round(cloud_pct, 1),
                "cloud_masked": bool(is_cloudy),
                "optical_valid": bool(opt_valid)
            })
            
    df = pd.DataFrame(records)
    return df


def generate_ground_truth_yield_labels(
    gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Computes calibrated ground truth yield (tonnes/hectare) for each parcel.
    Incorporates peak vegetation, volume backscatter, soil quality, and crop type potential.
    """
    np.random.seed(random_seed)
    
    valid_ts = timeseries_df[timeseries_df["optical_valid"] == True]
    agg_opt = valid_ts.groupby("parcel_id").agg({
        "ndvi": ["max", "mean"],
        "ndre": ["max"]
    }).reset_index()
    agg_opt.columns = ["parcel_id", "ndvi_max", "ndvi_mean", "ndre_max"]
    
    agg_sar = timeseries_df.groupby("parcel_id").agg({
        "sar_vh_linear": ["max", "mean"],
        "sar_rvi": ["max"]
    }).reset_index()
    agg_sar.columns = ["parcel_id", "vh_max", "vh_mean", "rvi_max"]
    
    merged = gdf.merge(agg_opt, on="parcel_id").merge(agg_sar, on="parcel_id")
    
    labels = []
    for idx, row in merged.iterrows():
        pid = row["parcel_id"]
        crop = row["crop_type"]
        soil = row["soil_quality"]
        ndvi_m = row["ndvi_max"]
        vh_m = row["vh_max"]
        
        # Determine crop potential baseline
        if "Corn" in crop or "Maize" in crop:
            base = 8.5 + 4.0 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 0.4)
            yield_tha = float(np.clip(base + noise, 7.0, 14.5))
        elif "Tomato" in crop:
            base = 65.0 + 30.0 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 2.5)
            yield_tha = float(np.clip(base + noise, 55.0, 95.0))
        elif "Alfalfa" in crop:
            base = 12.0 + 6.0 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 0.6)
            yield_tha = float(np.clip(base + noise, 10.0, 20.0))
        elif "Sugarcane" in crop:
            base = 55.0 + 30.0 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 2.5)
            yield_tha = float(np.clip(base + noise, 45.0, 85.0))
        elif "Rice" in crop:
            base = 3.2 + 2.2 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 0.18)
            yield_tha = float(np.clip(base + noise, 2.8, 6.2))
        elif "Soybean" in crop:
            base = 2.4 + 1.8 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 0.15)
            yield_tha = float(np.clip(base + noise, 1.8, 4.5))
        elif "Mustard" in crop or "Rapeseed" in crop:
            base = 1.8 + 1.2 * (ndvi_m * 0.6 + (vh_m / 0.04) * 0.4) * soil
            noise = np.random.normal(0, 0.12)
            yield_tha = float(np.clip(base + noise, 1.5, 3.6))
        else: # Wheat default
            base = 3.6 + 2.2 * (ndvi_m * 0.5 + (vh_m / 0.04) * 0.5) * soil
            noise = np.random.normal(0, 0.18)
            yield_tha = float(np.clip(base + noise, 3.2, 7.2))
            
        labels.append({
            "parcel_id": pid,
            "crop_type": crop,
            "area_ha": row["area_ha"],
            "soil_quality": soil,
            "yield_tha": round(yield_tha, 2),
            "provenance": "synthetic_demonstration",
            "is_ground_truth": True
        })
        
    return pd.DataFrame(labels)


def export_sample_dataset(
    output_dir: str = "data/sample",
    n_rows: int = 6,
    n_cols: int = 6,
    base_lat: Optional[float] = None,
    base_lon: Optional[float] = None,
    region_name: Optional[str] = None,
    custom_crops: Optional[List[Tuple[str, float, float]]] = None,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generates and saves:
      - parcels.geojson
      - timeseries.csv
      - yield_labels.csv
      - sample_schema.json
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lat = base_lat if base_lat is not None else 30.9010
    lon = base_lon if base_lon is not None else 75.8573
    reg_name = region_name or "Punjab Agricultural Study Region"
    
    # 1. Generate Parcels
    gdf = generate_synthetic_agricultural_study_area(
        n_rows=n_rows,
        n_cols=n_cols,
        base_lat=lat,
        base_lon=lon,
        region_name=reg_name,
        custom_crops=custom_crops,
        random_seed=random_seed
    )
    geojson_path = os.path.join(output_dir, "parcels.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    
    # 2. Generate Multi-temporal time-series
    ts_df = generate_multitemporal_satellite_timeseries(
        gdf=gdf, random_seed=random_seed
    )
    ts_path = os.path.join(output_dir, "timeseries.csv")
    ts_df.to_csv(ts_path, index=False)
    
    # 3. Generate Yield Labels
    labels_df = generate_ground_truth_yield_labels(
        gdf=gdf, timeseries_df=ts_df, random_seed=random_seed
    )
    labels_path = os.path.join(output_dir, "yield_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    
    # 4. Generate Schema Documentation JSON
    schema_info = {
        "dataset_name": f"CropYield-ST {reg_name} Field Dataset",
        "description": f"Multi-sensor Sentinel-1/2 time series and yield labels for {n_rows*n_cols} contiguous parcels",
        "spatial_coverage": f"{lat:.4f} N, {lon:.4f} E ({reg_name})",
        "temporal_coverage": "2025-11-01 to 2026-04-06 (14 bi-weekly acquisition timesteps)",
        "sensors": {
            "optical": "Sentinel-2 MSI Level-2A (B2 Blue, B4 Red, B5 RedEdge, B8 NIR, B11 SWIR)",
            "radar": "Sentinel-1 C-band SAR GRD (VV, VH dual polarization)"
        },
        "indices_included": ["NDVI", "EVI", "NDRE", "NDWI", "SAR VV/VH Ratio", "SAR RVI", "SAR dB"],
        "target_variable": "yield_tha (Crop yield in tonnes per hectare)",
        "provenance": "synthetic_demonstration",
        "disclaimer": "This sample dataset is calibrated for algorithm verification and faculty evaluation."
    }
    schema_path = os.path.join(output_dir, "sample_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)
        
    return {
        "parcels_geojson": geojson_path,
        "timeseries_csv": ts_path,
        "yield_labels_csv": labels_path,
        "schema_json": schema_path
    }


if __name__ == "__main__":
    out = export_sample_dataset()
    print("Sample dataset successfully generated at:", out)
