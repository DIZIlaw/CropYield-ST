"""
Unit tests for agricultural parcel graph construction and spatial Laplacian normalization.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pytest

from backend.app.geospatial.parcel_graph import (
    build_parcel_graph_from_geodataframe,
    build_parcel_graph_from_geojson
)


def create_dummy_grid_parcels():
    """Creates a 2x2 grid of contiguous polygons."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    pids = ["P01", "P02", "P03", "P04"]
    gdf = gpd.GeoDataFrame({"parcel_id": pids, "geometry": polys}, crs="EPSG:3857")
    return gdf


def test_parcel_graph_adjacency():
    """Verify shared-boundary adjacency for 2x2 grid."""
    gdf = create_dummy_grid_parcels()
    res = build_parcel_graph_from_geodataframe(gdf, id_col="parcel_id", method="touches")
    
    assert res["n_nodes"] == 4
    assert res["adjacency_matrix"].shape == (4, 4)
    
    # Check symmetry A = A^T
    adj = res["adjacency_matrix"]
    assert np.allclose(adj, adj.T)
    
    # P01 (index 0) touches P02 (index 1) and P03 (index 2)
    assert adj[0, 1] == 1.0
    assert adj[0, 2] == 1.0


def test_normalized_laplacian_properties():
    """Verify normalized adjacency matrix D^-0.5 * (A + I) * D^-0.5."""
    gdf = create_dummy_grid_parcels()
    res = build_parcel_graph_from_geodataframe(gdf, id_col="parcel_id", method="touches")
    norm_adj = res["normalized_adjacency"]
    
    # Must be symmetric
    assert np.allclose(norm_adj, norm_adj.T)
    
    # Diagonal must be positive (self-loops)
    diag = np.diag(norm_adj)
    assert np.all(diag > 0.0)
    assert np.all(diag <= 1.0)


def test_neighbors_lookup():
    """Verify neighbors dictionary correctly populates adjacent parcel IDs."""
    gdf = create_dummy_grid_parcels()
    res = build_parcel_graph_from_geodataframe(gdf, id_col="parcel_id", method="touches")
    nbrs = res["neighbors_dict"]
    
    assert "P01" in nbrs
    assert "P02" in nbrs["P01"]
    assert "P03" in nbrs["P01"]
