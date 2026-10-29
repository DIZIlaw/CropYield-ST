"""
Parcels and Geospatial Explorer Endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from backend.app.schemas.models import ParcelMetadata, ParcelDetail, ParcelTimeSeriesResponse
from backend.app.data.storage import storage

router = APIRouter(tags=["Parcels"])


@router.get("/parcels", response_model=List[ParcelMetadata])
def list_parcels():
    """Lists all parcels with geometry, centroids, and prediction data."""
    return storage.get_all_parcels_metadata()


@router.get("/parcels/graph")
def get_parcel_graph():
    """Returns spatial adjacency graph nodes, edges, and connectivity."""
    if storage.graph_data is None:
        raise HTTPException(status_code=404, detail="Parcel graph not constructed.")
        
    all_parcels = storage.get_all_parcels_metadata()
    nodes = [
        {
            "id": p["parcel_id"],
            "crop_type": p["crop_type"],
            "lat": p["centroid_lat"],
            "lon": p["centroid_lon"],
            "yield": p["predicted_yield"]
        }
        for p in all_parcels
    ]
    
    edges = []
    pid_to_idx = {p["id"]: i for i, p in enumerate(nodes)}
    for pid, nbrs in storage.graph_data["neighbors_dict"].items():
        if pid in pid_to_idx:
            for nbr in nbrs:
                if nbr in pid_to_idx and pid_to_idx[pid] < pid_to_idx[nbr]:
                    edges.append({
                        "source": pid,
                        "target": nbr,
                        "distance_m": round(storage.graph_data["edge_distances"].get((pid_to_idx[pid], pid_to_idx[nbr]), 500.0), 1)
                    })
                    
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }


@router.get("/parcels/{parcel_id}", response_model=ParcelDetail)
def get_parcel_detail(parcel_id: str):
    """Returns deep details for a specific agricultural parcel."""
    detail = storage.get_parcel_detail(parcel_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found.")
    return detail


@router.get("/parcels/{parcel_id}/neighbors")
def get_parcel_neighbors(parcel_id: str):
    """Returns list of immediately adjacent parcel IDs and metadata."""
    if not storage.graph_data:
        raise HTTPException(status_code=404, detail="Graph not initialized.")
        
    nbr_ids = storage.graph_data["neighbors_dict"].get(parcel_id, [])
    all_meta = {p["parcel_id"]: p for p in storage.get_all_parcels_metadata()}
    
    neighbors = [all_meta[nid] for nid in nbr_ids if nid in all_meta]
    return {
        "parcel_id": parcel_id,
        "neighbors_count": len(neighbors),
        "neighbors": neighbors
    }


@router.get("/parcels/{parcel_id}/timeseries", response_model=ParcelTimeSeriesResponse)
def get_parcel_timeseries(parcel_id: str):
    """Returns multi-temporal Sentinel-1 and Sentinel-2 feature observations."""
    detail = storage.get_parcel_detail(parcel_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found.")
    return storage.get_timeseries_for_parcel(parcel_id)
