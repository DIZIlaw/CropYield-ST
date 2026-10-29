"""
CropYield-ST: Agricultural Parcel Graph Construction Module
Constructs spatial adjacency graphs using Shapely geometries and GeoPandas.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform
import pyproj


def build_parcel_graph_from_geodataframe(
    gdf: gpd.GeoDataFrame,
    id_col: str = "parcel_id",
    method: str = "hybrid",
    distance_threshold_m: float = 1200.0,
    k_neighbors: int = 4
) -> Dict[str, Any]:
    """
    Constructs a spatial graph from a GeoPandas GeoDataFrame.
    
    Methods:
      - 'touches': Adjacent fields sharing a common border / boundary (Queen/Rook contiguity).
      - 'distance': Fields whose centroids or minimum distance is within `distance_threshold_m`.
      - 'knn': Each field is connected to its k nearest centroids.
      - 'hybrid': Fields that touch are connected, plus ensuring every field has at least 1 neighbor
                  via nearest centroid if isolated.
                  
    Returns:
      Dict with:
        - parcel_ids: List[str] aligned with node indices 0 .. N-1
        - edge_index: np.ndarray shape [2, E] (undirected pairs)
        - adjacency_matrix: np.ndarray shape [N, N] binary or weighted
        - normalized_adjacency: np.ndarray shape [N, N] (D^-0.5 * (A + I) * D^-0.5)
        - neighbors_dict: Dict[str, List[str]] mapping parcel_id -> list of adjacent parcel_ids
        - edge_distances: Dict[Tuple[int, int], float] distance in meters
    """
    if id_col not in gdf.columns:
        raise ValueError(f"Column '{id_col}' not found in GeoDataFrame columns: {list(gdf.columns)}")
        
    # Ensure parcel IDs are strings and sorted deterministically
    gdf = gdf.copy()
    gdf[id_col] = gdf[id_col].astype(str)
    gdf = gdf.sort_values(by=id_col).reset_index(drop=True)
    
    n_nodes = len(gdf)
    parcel_ids = gdf[id_col].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(parcel_ids)}
    
    # Ensure projected CRS for accurate metric distance calculations if needed
    if gdf.crs is None or gdf.crs.is_geographic:
        # Reproject to local UTM or World Mercator for distance metric in meters
        gdf_proj = gdf.to_crs(epsg=3857)
    else:
        gdf_proj = gdf
        
    centroids = gdf_proj.geometry.centroid
    
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    edge_list = set()
    edge_distances = {}
    
    for i in range(n_nodes):
        poly_i = gdf_proj.geometry.iloc[i]
        c_i = centroids.iloc[i]
        
        # Check boundary sharing or intersections
        for j in range(i + 1, n_nodes):
            poly_j = gdf_proj.geometry.iloc[j]
            c_j = centroids.iloc[j]
            dist_m = float(c_i.distance(c_j))
            
            is_connected = False
            
            if method in ("touches", "hybrid"):
                # Check if polygons share a border or intersect at boundary
                if poly_i.touches(poly_j) or poly_i.intersects(poly_j):
                    is_connected = True
            
            if method in ("distance", "hybrid") and not is_connected:
                if dist_m <= distance_threshold_m:
                    is_connected = True
                    
            if is_connected:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
                edge_list.add((i, j))
                edge_list.add((j, i))
                edge_distances[(i, j)] = dist_m
                edge_distances[(j, i)] = dist_m

    # If knn is requested or hybrid requires fallback for isolated nodes
    for i in range(n_nodes):
        deg = np.sum(adj_matrix[i])
        if (method == "knn" and deg < k_neighbors) or (method in ("touches", "hybrid") and deg == 0):
            # Find nearest neighbors by centroid
            c_i = centroids.iloc[i]
            dists = [
                (j, float(c_i.distance(centroids.iloc[j])))
                for j in range(n_nodes) if j != i
            ]
            dists.sort(key=lambda x: x[1])
            needed = k_neighbors if method == "knn" else 1
            for j, d_m in dists[:needed]:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
                edge_list.add((i, j))
                edge_list.add((j, i))
                edge_distances[(i, j)] = d_m
                edge_distances[(j, i)] = d_m

    # Build edge_index tensor format [2, E]
    if len(edge_list) > 0:
        edges_sorted = sorted(list(edge_list))
        edge_index = np.array(edges_sorted, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        
    # Calculate symmetric normalized adjacency matrix: D_tilde^-0.5 * (A + I) * D_tilde^-0.5
    A_tilde = adj_matrix + np.eye(n_nodes, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_tilde_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adjacency = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    
    # Build neighbors dict
    neighbors_dict = {}
    for i, pid in enumerate(parcel_ids):
        nbr_indices = np.where(adj_matrix[i] > 0)[0]
        neighbors_dict[pid] = [parcel_ids[j] for j in nbr_indices]
        
    return {
        "parcel_ids": parcel_ids,
        "n_nodes": n_nodes,
        "n_edges": len(edge_list) // 2,
        "edge_index": edge_index,
        "adjacency_matrix": adj_matrix,
        "normalized_adjacency": normalized_adjacency,
        "neighbors_dict": neighbors_dict,
        "edge_distances": edge_distances
    }


def build_parcel_graph_from_geojson(
    geojson_path_or_dict: Any,
    id_col: str = "parcel_id",
    method: str = "hybrid",
    distance_threshold_m: float = 1200.0
) -> Dict[str, Any]:
    """
    Constructs parcel graph from a GeoJSON filepath or dictionary.
    """
    if isinstance(geojson_path_or_dict, dict):
        gdf = gpd.GeoDataFrame.from_features(geojson_path_or_dict["features"])
        if "crs" in geojson_path_or_dict:
            gdf.set_crs(geojson_path_or_dict["crs"], inplace=True)
        else:
            gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gpd.read_file(geojson_path_or_dict)
        
    if id_col not in gdf.columns:
        # Fallback to 'id' or first string column
        if "id" in gdf.columns:
            gdf[id_col] = gdf["id"]
        elif "ID" in gdf.columns:
            gdf[id_col] = gdf["ID"]
        else:
            gdf[id_col] = [f"PARCEL_{i+1:03d}" for i in range(len(gdf))]
            
    return build_parcel_graph_from_geodataframe(
        gdf=gdf,
        id_col=id_col,
        method=method,
        distance_threshold_m=distance_threshold_m
    )
