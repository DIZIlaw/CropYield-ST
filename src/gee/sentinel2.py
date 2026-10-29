"""
Sentinel-2 Optical Satellite Data Acquisition and Preprocessing Pipeline
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Standard Sentinel-2 L2A Bands
S2_BANDS = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "red_edge_1",
    "B6": "red_edge_2",
    "B7": "red_edge_3",
    "B8": "nir",
    "B8A": "nir_narrow",
    "B11": "swir_1",
    "B12": "swir_2",
    "SCL": "scene_classification"
}


def mask_s2_clouds_gee(image: Any) -> Any:
    """
    Apply cloud and shadow masking on Sentinel-2 Level-2A surface reflectance image in GEE.
    Uses Scene Classification Layer (SCL).
    
    SCL classes masked out:
        3 = Cloud shadow
        8 = Cloud medium probability
        9 = Cloud high probability
        10 = Thin cirrus
        11 = Snow/ice
    """
    try:
        scl = image.select("SCL")
        # Keep vegetation (4), bare soil (5), water (6), unclassified (7), etc.
        cloud_shadow = scl.eq(3)
        cloud_med = scl.eq(8)
        cloud_high = scl.eq(9)
        cirrus = scl.eq(10)
        snow = scl.eq(11)
        
        mask = cloud_shadow.Or(cloud_med).Or(cloud_high).Or(cirrus).Or(snow).Not()
        # Scale reflectance by 10000 (standard ESA L2A scale factor)
        return image.updateMask(mask).divide(10000.0)
    except Exception as e:
        logger.error(f"Error applying GEE cloud mask: {e}")
        return image


def fetch_sentinel2_collection_gee(
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_cloud_percent: float = 20.0
) -> Optional[Any]:
    """
    Query Copernicus Sentinel-2 Harmonized Surface Reflectance (L2A) in Google Earth Engine.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        max_cloud_percent: Maximum allowable granule cloud coverage percentage
        
    Returns:
        ee.ImageCollection or None if GEE is not initialized
    """
    try:
        import ee
        roi = ee.Geometry.Rectangle(bbox)
        
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
            .map(mask_s2_clouds_gee)
        )
        return collection
    except Exception as e:
        logger.warning(f"GEE Sentinel-2 fetch unavailable: {e}")
        return None


def generate_synthetic_sentinel2_sample(
    shape: Tuple[int, int] = (128, 128),
    seed: int = 42,
    cloud_fraction: float = 0.05
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Generate a physically grounded sample multi-spectral Sentinel-2 scene for testing and 
    offline development when GEE authentication is not configured.
    
    This adheres to standard Sentinel-2 Surface Reflectance optical physics:
    - Soil: moderate visible reflectance, low NIR, rising SWIR
    - Healthy Crop: low Blue/Red (chlorophyll absorption), moderate Green, high NIR (mesophyll), high Red-Edge
    - Water: low NIR/SWIR, high Blue/Green
    
    Returns:
        Tuple of (bands_dict, cloud_mask, metadata)
    """
    rng = np.random.RandomState(seed)
    h, w = shape
    
    # Generate spatial agricultural parcel patterns (Voronoi/perlin-like crop fields)
    y_grid, x_grid = np.ogrid[:h, :w]
    num_fields = 16
    centers_x = rng.randint(10, w - 10, size=num_fields)
    centers_y = rng.randint(10, h - 10, size=num_fields)
    
    # Compute nearest field index for each pixel
    field_map = np.zeros((h, w), dtype=int)
    min_dist = np.full((h, w), np.inf)
    for i, (cx, cy) in enumerate(zip(centers_x, centers_y)):
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        mask = dist < min_dist
        field_map[mask] = i
        min_dist[mask] = dist[mask]
        
    # Assign crop growth vigor (NDVI potential) to each field (0.3 to 0.85)
    field_vigors = rng.uniform(0.35, 0.85, size=num_fields)
    pixel_vigor = field_vigors[field_map] + rng.normal(0, 0.03, size=(h, w))
    pixel_vigor = np.clip(pixel_vigor, 0.1, 0.95)
    
    # Generate realistic surface reflectance values (0.0 to 1.0)
    # Band 2 (Blue ~ 490 nm): 0.02 - 0.06 for vegetation
    blue = np.clip(0.04 - 0.02 * pixel_vigor + rng.normal(0, 0.005, size=(h, w)), 0.01, 0.3)
    # Band 3 (Green ~ 560 nm): 0.04 - 0.10 for vegetation
    green = np.clip(0.05 + 0.05 * pixel_vigor + rng.normal(0, 0.008, size=(h, w)), 0.02, 0.4)
    # Band 4 (Red ~ 665 nm): 0.02 - 0.08 for healthy vegetation, high for bare soil
    red = np.clip(0.12 - 0.10 * pixel_vigor + rng.normal(0, 0.006, size=(h, w)), 0.01, 0.5)
    # Band 5 (Red Edge 1 ~ 705 nm): Transition zone
    red_edge_1 = np.clip(0.15 + 0.20 * pixel_vigor + rng.normal(0, 0.01, size=(h, w)), 0.05, 0.6)
    # Band 8 (NIR ~ 842 nm): Strong reflection (0.35 - 0.65)
    nir = np.clip(0.20 + 0.45 * pixel_vigor + rng.normal(0, 0.015, size=(h, w)), 0.05, 0.9)
    # Band 11 (SWIR 1 ~ 1610 nm): 0.15 - 0.30
    swir_1 = np.clip(0.25 - 0.12 * pixel_vigor + rng.normal(0, 0.01, size=(h, w)), 0.05, 0.6)
    
    # SCL (Scene Classification Layer)
    # 4 = Vegetation, 5 = Bare soil, 8/9 = Cloud, 3 = Cloud shadow
    scl = np.full((h, w), 4, dtype=np.uint8)
    scl[pixel_vigor < 0.25] = 5 # Bare soil
    
    # Cloud simulation
    cloud_mask = np.zeros((h, w), dtype=bool)
    if cloud_fraction > 0:
        cloud_center_x, cloud_center_y = rng.randint(0, w), rng.randint(0, h)
        cloud_radius = int(np.sqrt(h * w * cloud_fraction / np.pi))
        cloud_dist = np.sqrt((x_grid - cloud_center_x)**2 + (y_grid - cloud_center_y)**2)
        cloud_mask = cloud_dist < cloud_radius
        
        # In cloud pixels, visible and NIR reflectances are very high
        blue[cloud_mask] = np.clip(rng.normal(0.65, 0.05, size=np.sum(cloud_mask)), 0.5, 0.9)
        green[cloud_mask] = np.clip(rng.normal(0.65, 0.05, size=np.sum(cloud_mask)), 0.5, 0.9)
        red[cloud_mask] = np.clip(rng.normal(0.65, 0.05, size=np.sum(cloud_mask)), 0.5, 0.9)
        nir[cloud_mask] = np.clip(rng.normal(0.68, 0.05, size=np.sum(cloud_mask)), 0.5, 0.9)
        scl[cloud_mask] = 9 # High probability cloud
    
    bands = {
        "B2": blue.astype(np.float32),
        "B3": green.astype(np.float32),
        "B4": red.astype(np.float32),
        "B5": red_edge_1.astype(np.float32),
        "B8": nir.astype(np.float32),
        "B11": swir_1.astype(np.float32),
        "SCL": scl
    }
    
    metadata = {
        "source": "Sentinel-2 Physical Optical Simulation (Reproducible Prototype Fallback)",
        "spatial_resolution_m": 10.0,
        "width": w,
        "height": h,
        "num_fields": num_fields,
        "cloud_pixel_count": int(np.sum(cloud_mask)),
        "cloud_fraction": float(np.mean(cloud_mask)),
        "center_lat": 41.5868,
        "center_lon": -93.6250
    }
    
    return bands, cloud_mask, metadata
