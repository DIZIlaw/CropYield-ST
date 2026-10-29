"""
CropYield-ST: Temporal Feature Preparation and Alignment Module
Aligns multi-sensor time-series into 3D tensors [num_parcels, timesteps, num_features].
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


DEFAULT_FEATURE_COLUMNS = [
    "ndvi",
    "evi",
    "ndre",
    "ndwi",
    "sar_vv_linear",
    "sar_vh_linear",
    "sar_vh_vv_ratio",
    "sar_rvi",
    "sar_vv_db",
    "sar_vh_db"
]

OPTICAL_FEATURE_COLUMNS = [
    "ndvi",
    "evi",
    "ndre",
    "ndwi"
]

SAR_FEATURE_COLUMNS = [
    "sar_vv_linear",
    "sar_vh_linear",
    "sar_vh_vv_ratio",
    "sar_rvi",
    "sar_vv_db",
    "sar_vh_db"
]


class TemporalFeatureProcessor:
    """
    Transforms tabular multi-sensor timeseries into aligned 3D tensors.
    Fits normalization statistics strictly on training parcel subsets.
    """
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        impute_cloudy: bool = True,
        scaler_type: str = "standard"
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
        self.impute_cloudy = impute_cloudy
        self.scaler_type = scaler_type
        self.scaler: Optional[StandardScaler] = None
        self.fitted_parcel_ids: List[str] = []
        self.time_dates: List[str] = []
        self.n_timesteps: int = 0
        
    def fit(self, df: pd.DataFrame, train_parcel_ids: Optional[List[str]] = None) -> "TemporalFeatureProcessor":
        """
        Fits the scaler on features from training parcels only to prevent leakage.
        """
        df_copy = df.copy()
        if train_parcel_ids is not None:
            df_fit = df_copy[df_copy["parcel_id"].isin(train_parcel_ids)]
        else:
            df_fit = df_copy
            
        fit_values = df_fit[self.feature_cols].values
        
        # Replace non-finite with column means
        nan_mask = ~np.isfinite(fit_values)
        if np.any(nan_mask):
            col_means = np.nanmean(np.where(nan_mask, np.nan, fit_values), axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for c in range(fit_values.shape[1]):
                fit_values[nan_mask[:, c], c] = col_means[c]
                
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.scaler.fit(fit_values)
        if train_parcel_ids is not None:
            self.fitted_parcel_ids = list(train_parcel_ids)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        ordered_parcel_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the dataframe into a 3D tensor:
          [N_parcels, T_timesteps, F_features]
        Also returns a 2D optical validity mask:
          [N_parcels, T_timesteps]
        """
        df_copy = df.copy()
        
        # Ensure ordered unique dates
        dates = sorted(df_copy["date"].unique().tolist())
        self.time_dates = dates
        self.n_timesteps = len(dates)
        n_parcels = len(ordered_parcel_ids)
        n_features = len(self.feature_cols)
        
        # Sort values
        df_copy = df_copy.sort_values(by=["parcel_id", "date"]).reset_index(drop=True)
        
        tensor_3d = np.zeros((n_parcels, self.n_timesteps, n_features), dtype=np.float32)
        validity_mask = np.ones((n_parcels, self.n_timesteps), dtype=bool)
        
        # Index dataframe by (parcel_id, date) for O(1) lookup
        df_indexed = df_copy.set_index(["parcel_id", "date"])
        
        for i, pid in enumerate(ordered_parcel_ids):
            for t, dt in enumerate(dates):
                if (pid, dt) in df_indexed.index:
                    row = df_indexed.loc[(pid, dt)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    
                    feats = row[self.feature_cols].values.astype(np.float32)
                    tensor_3d[i, t, :] = np.nan_to_num(feats, nan=0.0)
                    
                    if "optical_valid" in row:
                        validity_mask[i, t] = bool(row["optical_valid"])
                else:
                    # Missing observation on this date
                    validity_mask[i, t] = False
                    
            # Handle cloudy/missing imputation per parcel
            if self.impute_cloudy:
                # Forward-fill optical indices for cloudy timesteps
                for c, col_name in enumerate(self.feature_cols):
                    if col_name in OPTICAL_FEATURE_COLUMNS:
                        val_series = tensor_3d[i, :, c]
                        is_valid = validity_mask[i, :]
                        if not np.all(is_valid) and np.any(is_valid):
                            # Interpolate invalid optical values
                            valid_indices = np.where(is_valid)[0]
                            all_indices = np.arange(self.n_timesteps)
                            tensor_3d[i, :, c] = np.interp(
                                all_indices,
                                valid_indices,
                                val_series[valid_indices]
                            )

        # Scale features using fitted scaler
        if self.scaler is not None:
            flat_view = tensor_3d.reshape(-1, n_features)
            scaled_flat = self.scaler.transform(flat_view)
            tensor_3d = scaled_flat.reshape(n_parcels, self.n_timesteps, n_features)
            
        return tensor_3d.astype(np.float32), validity_mask


def extract_tabular_summary_features(
    df: pd.DataFrame,
    ordered_parcel_ids: List[str]
) -> pd.DataFrame:
    """
    Extracts tabular summary statistics (mean, max, min, std) for classical ML baselines.
    """
    summary_cols = ["ndvi", "evi", "ndre", "sar_vv_linear", "sar_vh_linear", "sar_rvi", "sar_vh_vv_ratio"]
    existing_cols = [c for c in summary_cols if c in df.columns]
    
    grouped = df.groupby("parcel_id")[existing_cols].agg(["mean", "max", "min", "std"]).reset_index()
    grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
    
    # Align to ordered_parcel_ids
    df_ordered = pd.DataFrame({"parcel_id": ordered_parcel_ids})
    merged = df_ordered.merge(grouped, on="parcel_id", how="left").fillna(0.0)
    return merged.drop(columns=["parcel_id"])
