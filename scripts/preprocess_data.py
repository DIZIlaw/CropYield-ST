"""
Command-line utility to preprocess raw satellite observations and parcel geometries.
"""
import sys
import os
import argparse
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.geospatial.parcel_graph import build_parcel_graph_from_geodataframe
from backend.app.features.temporal_features import TemporalFeatureProcessor


def main():
    parser = argparse.ArgumentParser(description="CropYield-ST Data Preprocessing CLI")
    parser.add_argument("--geojson", type=str, default="data/sample/parcels.geojson", help="Path to parcel GeoJSON")
    parser.add_argument("--timeseries", type=str, default="data/sample/timeseries.csv", help="Path to timeseries CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed tensors")
    args = parser.parse_args()

    print(f"[1/3] Loading parcel geometries from {args.geojson}...")
    gdf = gpd.read_file(args.geojson)
    if "parcel_id" not in gdf.columns:
        gdf["parcel_id"] = [f"PARCEL_{i+1:03d}" for i in range(len(gdf))]
    gdf = gdf.sort_values(by="parcel_id").reset_index(drop=True)
    pids = gdf["parcel_id"].tolist()
    print(f"      Loaded {len(gdf)} parcels.")

    print(f"[2/3] Building parcel topological adjacency graph...")
    graph = build_parcel_graph_from_geodataframe(gdf, id_col="parcel_id")
    print(f"      Constructed graph with {graph['n_nodes']} nodes and {graph['n_edges']} edges.")

    print(f"[3/3] Aligning multi-sensor temporal sequences from {args.timeseries}...")
    ts_df = pd.read_csv(args.timeseries)
    processor = TemporalFeatureProcessor(impute_cloudy=True)
    processor.fit(ts_df)
    tensor_3d, mask = processor.transform(ts_df, ordered_parcel_ids=pids)
    print(f"      Feature tensor shape: {tensor_3d.shape} [Parcels, Timesteps, Features]")

    os.makedirs(args.output_dir, exist_ok=True)
    import numpy as np
    out_file = os.path.join(args.output_dir, "preprocessed_features.npz")
    np.savez_compressed(
        out_file,
        features=tensor_3d,
        validity_mask=mask,
        adjacency=graph["adjacency_matrix"],
        normalized_adjacency=graph["normalized_adjacency"],
        parcel_ids=np.array(pids)
    )
    print(f"Successfully exported preprocessed tensors to: {out_file}")


if __name__ == "__main__":
    main()
