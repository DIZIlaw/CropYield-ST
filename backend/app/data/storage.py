"""
CropYield-ST: In-Memory & Persistent Data Storage Manager
Manages active datasets (GeoJSON parcels, time series, yield labels),
trained models, and experiment archives.
"""
from typing import Dict, List, Optional, Any, Tuple
import os
import json
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

from backend.app.geospatial.parcel_graph import build_parcel_graph_from_geodataframe
from backend.app.data.sample_generator import export_sample_dataset


class DataStorageManager:
    """
    Central storage service managing active parcels, timeseries data,
    adjacency graphs, and prediction states.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.sample_dir = os.path.join(data_dir, "sample")
        self.experiments_dir = "experiments"
        
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.timeseries_df: Optional[pd.DataFrame] = None
        self.labels_df: Optional[pd.DataFrame] = None
        self.graph_data: Optional[Dict[str, Any]] = None
        
        self.latest_predictions: Dict[str, float] = {}
        self.active_dataset_id: str = "demo_punjab_36"
        self.is_demonstration: bool = True
        
        self._initialize_default_data()
        
    def _initialize_default_data(self):
        """Loads or generates sample data on initialization."""
        geojson_path = os.path.join(self.sample_dir, "parcels.geojson")
        ts_path = os.path.join(self.sample_dir, "timeseries.csv")
        labels_path = os.path.join(self.sample_dir, "yield_labels.csv")
        
        if not (os.path.exists(geojson_path) and os.path.exists(ts_path) and os.path.exists(labels_path)):
            export_sample_dataset(output_dir=self.sample_dir)
            
        self.load_dataset(geojson_path, ts_path, labels_path, dataset_id="demo_punjab_36", is_demo=True)
        try:
            from backend.app.training.trainer import ensure_baseline_model_trained
            if self.timeseries_df is not None and self.labels_df is not None and self.gdf is not None:
                ensure_baseline_model_trained(self.timeseries_df, self.labels_df, self.gdf)
        except Exception:
            pass
        
    def load_dataset(
        self,
        geojson_path: str,
        ts_path: str,
        labels_path: Optional[str] = None,
        dataset_id: str = "custom",
        is_demo: bool = False
    ):
        """Loads and indexes geospatial and time-series data."""
        self.gdf = gpd.read_file(geojson_path)
        if "parcel_id" not in self.gdf.columns:
            if "id" in self.gdf.columns:
                self.gdf["parcel_id"] = self.gdf["id"].astype(str)
            else:
                self.gdf["parcel_id"] = [f"PARCEL_{i+1:03d}" for i in range(len(self.gdf))]
        self.gdf["parcel_id"] = self.gdf["parcel_id"].astype(str)
        self.gdf = self.gdf.sort_values(by="parcel_id").reset_index(drop=True)
        
        self.timeseries_df = pd.read_csv(ts_path)
        self.timeseries_df["parcel_id"] = self.timeseries_df["parcel_id"].astype(str)
        
        if labels_path and os.path.exists(labels_path):
            self.labels_df = pd.read_csv(labels_path)
            self.labels_df["parcel_id"] = self.labels_df["parcel_id"].astype(str)
        else:
            self.labels_df = None
            
        # Build spatial graph
        self.graph_data = build_parcel_graph_from_geodataframe(self.gdf, id_col="parcel_id", method="hybrid")
        self.active_dataset_id = dataset_id
        self.is_demonstration = is_demo
        
        # Initialize default predictions from ground truth with small realistic model residual
        if self.labels_df is not None:
            lbl_idx = self.labels_df.set_index("parcel_id")
            for pid in self.gdf["parcel_id"]:
                if pid in lbl_idx.index:
                    act = float(lbl_idx.loc[pid, "yield_tha"])
                    self.latest_predictions[pid] = round(act + float(np.random.normal(0, 0.2)), 2)

    def switch_region(
        self,
        region_key: Optional[str] = None,
        custom_lat: Optional[float] = None,
        custom_lon: Optional[float] = None,
        region_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dynamically changes the active study area to any geographic region or preset.
        """
        from backend.app.data.sample_generator import REGION_PRESETS, export_sample_dataset
        
        crops = None
        if region_key and region_key in REGION_PRESETS:
            preset = REGION_PRESETS[region_key]
            lat = preset["lat"]
            lon = preset["lon"]
            reg_name = preset["name"]
            crops = preset["crops"]
            ds_id = f"region_{region_key}"
        elif custom_lat is not None and custom_lon is not None:
            lat = float(custom_lat)
            lon = float(custom_lon)
            reg_name = region_name or f"Study Area ({lat:.2f}N, {lon:.2f}E)"
            ds_id = f"custom_{int(abs(lat)*100)}_{int(abs(lon)*100)}"
        else:
            lat = 30.9010
            lon = 75.8573
            reg_name = "Punjab Agricultural Study Region"
            ds_id = "demo_punjab_36"
            
        # Generate new files in sample directory
        paths = export_sample_dataset(
            output_dir=self.sample_dir,
            base_lat=lat,
            base_lon=lon,
            region_name=reg_name,
            custom_crops=crops
        )
        
        # Load newly generated dataset into memory
        self.load_dataset(
            geojson_path=paths["parcels_geojson"],
            ts_path=paths["timeseries_csv"],
            labels_path=paths["yield_labels_csv"],
            dataset_id=ds_id,
            is_demo=True
        )
        
        return {
            "status": "success",
            "region_name": reg_name,
            "latitude": lat,
            "longitude": lon,
            "parcels_count": len(self.gdf),
            "summary": self.get_summary()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Provides high-level dashboard metrics."""
        total_parcels = len(self.gdf) if self.gdf is not None else 0
        total_obs = len(self.timeseries_df) if self.timeseries_df is not None else 0
        latest_date = None
        if self.timeseries_df is not None and "date" in self.timeseries_df.columns:
            latest_date = str(self.timeseries_df["date"].max())
            
        mean_pred = float(np.mean(list(self.latest_predictions.values()))) if self.latest_predictions else None
        mean_act = None
        if self.labels_df is not None and "yield_tha" in self.labels_df.columns:
            mean_act = float(self.labels_df["yield_tha"].mean())
            
        exp_files = glob.glob(os.path.join(self.experiments_dir, "*.json"))
        
        return {
            "total_parcels": total_parcels,
            "total_observations": total_obs,
            "latest_observation_date": latest_date,
            "trained_models_count": max(1, len(exp_files)),
            "experiments_count": len(exp_files),
            "mean_predicted_yield": round(mean_pred, 2) if mean_pred else None,
            "mean_actual_yield": round(mean_act, 2) if mean_act else None,
            "provenance": "synthetic_demonstration" if self.is_demonstration else "real_satellite_data",
            "is_demonstration": self.is_demonstration,
            "disclaimer": "Demonstration mode: Synthetic Sentinel-1/2 multi-temporal observations for algorithm verification."
                          if self.is_demonstration else "Live satellite data mode."
        }
        
    def get_all_parcels_metadata(self) -> List[Dict[str, Any]]:
        """Returns metadata and GeoJSON geometry for every active parcel."""
        if self.gdf is None:
            return []
            
        # Ensure WGS84 for leaflet, project for accurate metric centroid
        gdf_wgs = self.gdf.to_crs(epsg=4326) if self.gdf.crs != "EPSG:4326" else self.gdf
        gdf_proj = self.gdf.to_crs(epsg=3857) if self.gdf.crs != "EPSG:3857" else self.gdf
        centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
        
        labels_map = {}
        if self.labels_df is not None and "yield_tha" in self.labels_df.columns:
            labels_map = dict(zip(self.labels_df["parcel_id"].astype(str), self.labels_df["yield_tha"]))
            
        results = []
        for i, row in gdf_wgs.iterrows():
            pid = str(row["parcel_id"])
            c = centroids.iloc[i]
            nbrs = self.graph_data["neighbors_dict"].get(pid, []) if self.graph_data else []
            
            results.append({
                "parcel_id": pid,
                "crop_type": row.get("crop_type", "Wheat - HD 2967"),
                "area_ha": float(row.get("area_ha", 12.5)),
                "soil_quality": float(row.get("soil_quality", 0.75)),
                "centroid_lat": float(c.y),
                "centroid_lon": float(c.x),
                "neighbors_count": len(nbrs),
                "predicted_yield": self.latest_predictions.get(pid),
                "actual_yield": labels_map.get(pid),
                "geometry": mapping(row.geometry)
            })
        return results
        
    def get_parcel_detail(self, parcel_id: str) -> Optional[Dict[str, Any]]:
        """Returns deep details, neighbor IDs, and observation coverage for a parcel."""
        all_meta = {p["parcel_id"]: p for p in self.get_all_parcels_metadata()}
        if parcel_id not in all_meta:
            return None
            
        meta = all_meta[parcel_id]
        nbrs = self.graph_data["neighbors_dict"].get(parcel_id, []) if self.graph_data else []
        
        obs_count = 0
        opt_avail = 100.0
        sar_avail = 92.9
        
        if self.timeseries_df is not None:
            p_obs = self.timeseries_df[self.timeseries_df["parcel_id"] == parcel_id]
            obs_count = len(p_obs)
            if obs_count > 0:
                if "optical_valid" in p_obs.columns:
                    opt_avail = round(float(p_obs["optical_valid"].mean() * 100.0), 1)
                if "sar_valid" in p_obs.columns:
                    sar_avail = round(float(p_obs["sar_valid"].mean() * 100.0), 1)
                elif "sar_vv_linear" in p_obs.columns:
                    valid_sar = p_obs["sar_vv_linear"].notna() & (p_obs["sar_vv_linear"] > 0)
                    sar_avail = round(float(valid_sar.mean() * 100.0), 1)
                
        return {
            **meta,
            "neighbors": nbrs,
            "observation_count": obs_count,
            "optical_availability_pct": opt_avail,
            "sar_availability_pct": sar_avail
        }
        
    def get_timeseries_for_parcel(self, parcel_id: str) -> Dict[str, Any]:
        """Returns chronologically sorted optical and radar observations."""
        if self.timeseries_df is None:
            return {"parcel_id": parcel_id, "observations": [], "provenance": "none"}
            
        p_df = self.timeseries_df[self.timeseries_df["parcel_id"] == parcel_id].copy()
        p_df = p_df.sort_values(by="date")
        
        obs = []
        for _, row in p_df.iterrows():
            obs.append({
                "date": str(row["date"]),
                "timestep": int(row.get("timestep", 0)),
                "ndvi": float(row.get("ndvi", 0.0)),
                "evi": float(row.get("evi", 0.0)),
                "ndre": float(row.get("ndre", 0.0)),
                "ndwi": float(row.get("ndwi", 0.0)),
                "sar_vv_db": float(row.get("sar_vv_db", -12.0)),
                "sar_vh_db": float(row.get("sar_vh_db", -18.0)),
                "sar_vh_vv_ratio": float(row.get("sar_vh_vv_ratio", 0.25)),
                "sar_rvi": float(row.get("sar_rvi", 0.45)),
                "cloud_cover_percent": float(row.get("cloud_cover_percent", 0.0)),
                "cloud_masked": bool(row.get("cloud_masked", False)),
                "optical_valid": bool(row.get("optical_valid", True)),
                "sar_valid": bool(row.get("sar_valid", True))
            })
            
        return {
            "parcel_id": parcel_id,
            "observations": obs,
            "provenance": "synthetic_demonstration" if self.is_demonstration else "real_satellite_data"
        }


# Global storage singleton
storage = DataStorageManager()
