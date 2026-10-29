"""
Command-line utility to train CropYield-ST or baseline models from terminal.
"""
import sys
import os
import argparse
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.training.trainer import train_model_experiment


def main():
    parser = argparse.ArgumentParser(description="CropYield-ST Model Training CLI")
    parser.add_argument("--model", type=str, default="st_gnn", choices=["st_gnn", "pure_temporal_rnn", "xgboost", "random_forest"])
    parser.add_argument("--features", type=str, default="fusion", choices=["fusion", "optical_only", "sar_only"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--geojson", type=str, default="data/sample/parcels.geojson")
    parser.add_argument("--timeseries", type=str, default="data/sample/timeseries.csv")
    parser.add_argument("--labels", type=str, default="data/sample/yield_labels.csv")
    args = parser.parse_args()

    print(f"Loading data files for training...")
    gdf = gpd.read_file(args.geojson) if os.path.exists(args.geojson) else None
    ts_df = pd.read_csv(args.timeseries)
    lbl_df = pd.read_csv(args.labels)

    print(f"Starting training run: Model={args.model}, Features={args.features}, Epochs={args.epochs}...")
    res = train_model_experiment(
        timeseries_df=ts_df,
        labels_df=lbl_df,
        gdf=gdf,
        model_type=args.model,
        feature_set=args.features,
        epochs=args.epochs,
        lr=args.lr
    )

    print("\n" + "=" * 50)
    print(f" TRAINING COMPLETE: {res['experiment_id']}")
    print("=" * 50)
    print(f"  Duration:   {res['duration_sec']} seconds")
    print(f"  Test RMSE:  {res['test_metrics']['rmse']} tonnes/ha")
    print(f"  Test MAE:   {res['test_metrics']['mae']} tonnes/ha")
    print(f"  Test R²:    {res['test_metrics']['r2']}")
    print(f"  Test MAPE:  {res['test_metrics']['mape_percent']}%")
    print(f"  Checkpoints saved to: {res['save_dir']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
