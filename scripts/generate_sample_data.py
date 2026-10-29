"""
Command-line utility to generate reproducible CropYield-ST sample datasets for any region.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.data.sample_generator import export_sample_dataset, REGION_PRESETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropYield-ST Location & Dataset Generator")
    parser.add_argument(
        "--region",
        type=str,
        default="punjab",
        choices=list(REGION_PRESETS.keys()) + ["custom"],
        help="Preset agricultural region"
    )
    parser.add_argument("--lat", type=float, default=None, help="Custom latitude (degrees)")
    parser.add_argument("--lon", type=float, default=None, help="Custom longitude (degrees)")
    parser.add_argument("--name", type=str, default=None, help="Custom region name")
    parser.add_argument("--rows", type=int, default=6, help="Grid rows of parcels")
    parser.add_argument("--cols", type=int, default=6, help="Grid cols of parcels")
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample"))

    if args.region in REGION_PRESETS:
        preset = REGION_PRESETS[args.region]
        lat = args.lat if args.lat is not None else preset["lat"]
        lon = args.lon if args.lon is not None else preset["lon"]
        name = args.name or preset["name"]
        crops = preset["crops"]
    else:
        lat = args.lat if args.lat is not None else 30.9010
        lon = args.lon if args.lon is not None else 75.8573
        name = args.name or f"Study Region ({lat:.2f}N, {lon:.2f}E)"
        crops = None

    print(f"Generating field parcels & satellite time series for: {name}")
    print(f"Location coordinates: {lat:.4f}° N, {lon:.4f}° E | Grid: {args.rows}x{args.cols} parcels")
    
    paths = export_sample_dataset(
        output_dir=out_dir,
        n_rows=args.rows,
        n_cols=args.cols,
        base_lat=lat,
        base_lon=lon,
        region_name=name,
        custom_crops=crops
    )
    
    print("\nGeneration complete:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")
