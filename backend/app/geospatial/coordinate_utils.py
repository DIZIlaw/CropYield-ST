"""
CropYield-ST: Geospatial Coordinate Utilities
Handles CRS transformations, centroid computations, bounding boxes, and GeoJSON conversions.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, mapping


def compute_centroids_lat_lon(gdf: gpd.GeoDataFrame, id_col: str = "parcel_id") -> Dict[str, Dict[str, float]]:
    """
    Computes latitude and longitude of polygon centroids in WGS84 (EPSG:4326).
    """
    gdf_wgs84 = gdf.to_crs(epsg=4326) if gdf.crs != "EPSG:4326" else gdf
    centroids = {}
    for idx, row in gdf_wgs84.iterrows():
        pid = str(row[id_col])
        c = row.geometry.centroid
        centroids[pid] = {
            "latitude": float(c.y),
            "longitude": float(c.x)
        }
    return centroids


def get_bounding_box(gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Returns bounds [minx, miny, maxx, maxy] in WGS84.
    """
    gdf_wgs84 = gdf.to_crs(epsg=4326) if gdf.crs != "EPSG:4326" else gdf
    bounds = gdf_wgs84.total_bounds
    return {
        "min_lon": float(bounds[0]),
        "min_lat": float(bounds[1]),
        "max_lon": float(bounds[2]),
        "max_lat": float(bounds[3])
    }
