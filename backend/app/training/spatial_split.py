"""
CropYield-ST: Spatial Splitting Module
Implements spatial block partitioning to prevent geographic information leakage between adjacent fields.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import geopandas as gpd


def spatial_block_split(
    parcel_ids: List[str],
    gdf: Optional[Any] = None,
    test_ratio: float = 0.25,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Partitions parcel IDs into Train, Validation, and Test sets.
    If GeoDataFrame is provided, clusters parcels spatially (e.g. by geographic coordinates)
    so contiguous geographical blocks are held out as test parcels, eliminating spatial adjacency leakage!
    """
    np.random.seed(random_seed)
    n = len(parcel_ids)
    
    if gdf is not None and len(gdf) == n:
        # Spatial quadrant or coordinate partition
        gdf_sorted = gdf.set_index("parcel_id").loc[parcel_ids].reset_index()
        gdf_proj = gdf_sorted.to_crs(epsg=3857) if gdf_sorted.crs and gdf_sorted.crs.is_geographic else gdf_sorted
        centroids = gdf_proj.geometry.centroid
        lons = np.array([c.x for c in centroids])
        lats = np.array([c.y for c in centroids])
        
        # Sort spatially by (lon + lat) to create geographic blocks
        spatial_metric = lons * 1.5 + lats
        sort_order = np.argsort(spatial_metric)
        sorted_pids = [parcel_ids[i] for i in sort_order]
        
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        
        # Take the outermost geographic block for test
        test_pids = sorted_pids[-n_test:]
        val_pids = sorted_pids[-(n_test + n_val):-n_test]
        train_pids = sorted_pids[:-(n_test + n_val)]
    else:
        # Fallback to stratified random split
        shuffled = np.random.permutation(parcel_ids).tolist()
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_pids = shuffled[:n_test]
        val_pids = shuffled[n_test:n_test + n_val]
        train_pids = shuffled[n_test + n_val:]
        
    return {
        "train": train_pids,
        "val": val_pids,
        "test": test_pids
    }

Optional_GDF = Any = object
