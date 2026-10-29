"""
Publication-Quality Remote Sensing Visualization Module (Optical & SAR)
"""
import os
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_rgb_composite(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Sentinel-2 True Color (RGB: B4, B3, B2)",
    percentile_stretch: Tuple[float, float] = (2, 98)
) -> plt.Figure:
    """Generate True Color RGB composite map."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    rgb = np.dstack([red, green, blue])
    
    p_low, p_high = np.nanpercentile(rgb, percentile_stretch)
    rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
    rgb_stretched = np.nan_to_num(rgb_stretched, nan=0.9)
    
    ax.imshow(rgb_stretched)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pixel X (10m Resolution)", fontsize=10)
    ax.set_ylabel("Pixel Y (10m Resolution)", fontsize=10)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_single_index_map(
    index_array: np.ndarray,
    index_name: str,
    output_path: Optional[str] = None,
    cmap: str = "RdYlGn",
    vmin: float = -0.2,
    vmax: float = 0.9,
    unit_label: str = "Index Value"
) -> plt.Figure:
    """Generate spatial heatmap for a single vegetation index."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    
    im = ax.imshow(index_array, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(f"Sentinel-2 {index_name.upper()} Spatial Map", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pixel X (10m / pixel)", fontsize=10)
    ax.set_ylabel("Pixel Y (10m / pixel)", fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{index_name.upper()} {unit_label}", fontsize=10)
    
    valid_data = index_array[~np.isnan(index_array)]
    if len(valid_data) > 0:
        stat_text = f"Mean: {np.mean(valid_data):.3f}\nMedian: {np.median(valid_data):.3f}\nStd: {np.std(valid_data):.3f}"
        ax.text(
            0.03, 0.95, stat_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
        )
        
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multi_indices_grid(
    indices: Dict[str, np.ndarray],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate a 2x2 grid map of optical indices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    configs = [
        ("ndvi", "NDVI (Normalized Difference Vegetation Index)", "RdYlGn", 0.0, 0.9),
        ("ndre", "NDRE (Red Edge Chlorophyll Index)", "YlGn", 0.0, 0.8),
        ("evi", "EVI (Enhanced Vegetation Index)", "Greens", 0.0, 1.0),
        ("ndwi", "NDWI (Normalized Difference Water Index)", "PuBuGn", -0.6, 0.4)
    ]
    for ax, (key, title, cmap, vmin, vmax) in zip(axes.flatten(), configs):
        if key in indices:
            arr = indices[key]
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="lightgray")
            im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Sentinel-2 Optical Vegetation & Water Indices Suite", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_index_distributions(
    indices: Dict[str, np.ndarray],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate density histograms for optical vegetation indices."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = {"ndvi": "#2ca02c", "ndre": "#1f77b4", "evi": "#9467bd", "ndwi": "#17becf"}
    for key, arr in indices.items():
        valid_vals = arr[~np.isnan(arr)].flatten()
        if len(valid_vals) > 0:
            ax.hist(
                valid_vals,
                bins=50,
                density=True,
                alpha=0.45,
                color=colors.get(key, "#333333"),
                label=f"{key.upper()} (\u03bc={np.mean(valid_vals):.2f}, \u03c3={np.std(valid_vals):.2f})"
            )
    ax.set_title("Vegetation Index Frequency Distributions (Study Area)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calculated Index Value", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sar_composite(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Sentinel-1 SAR False-Color (R: VV, G: VH, B: |VV-VH|)"
) -> plt.Figure:
    """
    Generate a 3-channel SAR false color composite representing backscatter and depolarization.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Normalize channels to [0, 1] range based on standard C-band agricultural bounds
    # VV: [-20, -5] dB, VH: [-28, -10] dB, Ratio/Diff: [4, 16] dB
    r = np.clip((vv_db - (-20.0)) / ((-5.0) - (-20.0)), 0.0, 1.0)
    g = np.clip((vh_db - (-28.0)) / ((-10.0) - (-28.0)), 0.0, 1.0)
    b_diff = np.abs(vv_db - vh_db)
    b = np.clip((b_diff - 4.0) / (16.0 - 4.0), 0.0, 1.0)
    
    sar_rgb = np.dstack([r, g, b])
    ax.imshow(sar_rgb)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pixel X (10m Resolution)", fontsize=10)
    ax.set_ylabel("Pixel Y (10m Resolution)", fontsize=10)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sar_backscatter_maps(
    sar_features: Dict[str, np.ndarray],
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate dual side-by-side maps for VV and VH backscatter (in dB) and RVI index.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    im0 = axes[0].imshow(sar_features["vv_db"], cmap="bone", vmin=-20, vmax=-6)
    axes[0].set_title("Sentinel-1 VV Backscatter (dB)\n[Surface Roughness & Vertical Structure]", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Backscatter \u03c3\u2070 (dB)")
    
    im1 = axes[1].imshow(sar_features["vh_db"], cmap="bone", vmin=-28, vmax=-11)
    axes[1].set_title("Sentinel-1 VH Backscatter (dB)\n[Canopy Volume Scattering & Biomass]", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Backscatter \u03c3\u2070 (dB)")
    
    im2 = axes[2].imshow(sar_features["rvi"], cmap="YlGn", vmin=0.1, vmax=0.9)
    axes[2].set_title("Radar Vegetation Index (RVI)\n[4*VH / (VV + VH)]", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="RVI Value")
    
    for ax in axes:
        ax.set_xlabel("Pixel X (10m)")
        ax.set_ylabel("Pixel Y (10m)")
        
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_optical_vs_sar_comparison(
    optical_rgb: np.ndarray,
    optical_masked_ndvi: np.ndarray,
    sar_vh_db: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Publication figure comparing Optical (Cloud-impacted) vs SAR (Cloud-penetrating) sensors.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # 1. Cloudy Optical RGB
    axes[0].imshow(optical_rgb)
    axes[0].set_title("Sentinel-2 Optical (Cloud-Affected)\n[Solar Reflectance Blocked]", fontsize=10, fontweight="bold")
    
    # 2. Masked Optical NDVI (with missing NaN gaps)
    cmap_ndvi = plt.get_cmap("RdYlGn").copy()
    cmap_ndvi.set_bad(color="#dcdcdc")
    im1 = axes[1].imshow(optical_masked_ndvi, cmap=cmap_ndvi, vmin=0.0, vmax=0.9)
    axes[1].set_title("Sentinel-2 NDVI (Masked)\n[Gaps / Missing Observations]", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="NDVI")
    
    # 3. Sentinel-1 SAR (Unimpeded continuous observation)
    im2 = axes[2].imshow(sar_vh_db, cmap="magma", vmin=-28, vmax=-11)
    axes[2].set_title("Sentinel-1 SAR VH Backscatter\n[Microwave Penetrates Clouds 100%]", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="VH Backscatter (dB)")
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle("Sensor Complementarity: Optical Cloud Degradation vs. Active Microwave Radar Invariance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
