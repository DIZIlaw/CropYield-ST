"""
Spectral Indices Calculation Module for Sentinel-2 Optical Imagery
"""
from typing import Dict, Optional, Union
import numpy as np


def compute_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    eps: float = 1e-8,
    clip_valid: bool = True
) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    Formula:
        NDVI = (NIR - RED) / (NIR + RED)
        
    Physical Relevance:
        Measures photosynthetic active biomass and chlorophyll content. Healthy vegetation 
        strongly absorbs visible red light (for photosynthesis) and strongly reflects 
        near-infrared (NIR) light due to mesophyll cell structure.
        
    Args:
        nir: Near-infrared band reflectance array (Sentinel-2 Band 8 or 8A).
        red: Red band reflectance array (Sentinel-2 Band 4).
        eps: Small constant to avoid division by zero.
        clip_valid: If True, clip results to valid physical range [-1.0, 1.0].
        
    Returns:
        np.ndarray: Computed NDVI values.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    
    numerator = nir - red
    denominator = nir + red
    
    # Avoid zero or near-zero division
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(np.abs(denominator) > eps, numerator / denominator, np.nan)
        
    if clip_valid:
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
    return ndvi


def compute_ndre(
    nir: np.ndarray,
    red_edge: np.ndarray,
    eps: float = 1e-8,
    clip_valid: bool = True
) -> np.ndarray:
    """
    Calculate Normalized Difference Red Edge Index (NDRE).
    
    Formula:
        NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        
    Physical Relevance:
        NDRE utilizes the narrow transition band between red absorption and NIR reflectance 
        (Sentinel-2 Band 5/6/7). Unlike standard NDVI, NDRE does NOT easily saturate in dense, 
        mid-to-late season crop canopies, making it superior for nitrogen monitoring and 
        late-stage yield forecasting.
        
    Args:
        nir: Near-infrared band reflectance (Band 8/8A).
        red_edge: Red edge band reflectance (Band 5, 705 nm).
        eps: Small constant to avoid division by zero.
        clip_valid: If True, clip to [-1.0, 1.0].
        
    Returns:
        np.ndarray: Computed NDRE values.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red_edge = np.asarray(red_edge, dtype=np.float32)
    
    numerator = nir - red_edge
    denominator = nir + red_edge
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndre = np.where(np.abs(denominator) > eps, numerator / denominator, np.nan)
        
    if clip_valid:
        ndre = np.clip(ndre, -1.0, 1.0)
        
    return ndre


def compute_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
    eps: float = 1e-8,
    clip_valid: bool = True
) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index (EVI).
    
    Formula:
        EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
        
    Physical Relevance:
        EVI optimizes the vegetation signal with improved sensitivity in high biomass regions 
        and reduces atmospheric aerosol scattering influences using the blue band (Band 2).
        
    Args:
        nir: Near-infrared band (Band 8).
        red: Red band (Band 4).
        blue: Blue band (Band 2).
        g: Gain factor (default 2.5).
        c1: Red band aerosol resistance coefficient (default 6.0).
        c2: Blue band aerosol resistance coefficient (default 7.5).
        l: Canopy background adjustment factor (default 1.0).
        eps: Small epsilon to prevent division by zero.
        clip_valid: If True, clip to valid range [-1.0, 1.5].
        
    Returns:
        np.ndarray: Computed EVI values.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)
    
    numerator = g * (nir - red)
    denominator = nir + (c1 * red) - (c2 * blue) + l
    
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = np.where(np.abs(denominator) > eps, numerator / denominator, np.nan)
        
    if clip_valid:
        evi = np.clip(evi, -1.0, 1.5)
        
    return evi


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    eps: float = 1e-8,
    clip_valid: bool = True
) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI - McFeeters).
    
    Formula:
        NDWI = (GREEN - NIR) / (GREEN + NIR)
        
    Physical Relevance:
        Delineates open water bodies and tracks soil surface moisture / moisture stress 
        in agricultural landscapes.
        
    Args:
        green: Green band reflectance (Band 3).
        nir: Near-infrared band reflectance (Band 8).
        eps: Small constant to avoid zero division.
        clip_valid: If True, clip to [-1.0, 1.0].
        
    Returns:
        np.ndarray: Computed NDWI values.
    """
    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    
    numerator = green - nir
    denominator = green + nir
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(np.abs(denominator) > eps, numerator / denominator, np.nan)
        
    if clip_valid:
        ndwi = np.clip(ndwi, -1.0, 1.0)
        
    return ndwi


def compute_all_optical_indices(
    bands: Dict[str, np.ndarray],
    mask_cloud: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all essential agricultural optical indices from a dictionary of reflectance bands.
    
    Expected Band Keys:
        'B2' or 'blue': Blue reflectance
        'B3' or 'green': Green reflectance
        'B4' or 'red': Red reflectance
        'B5' or 'red_edge': Red edge reflectance
        'B8' or 'nir': NIR reflectance
        
    Args:
        bands: Dictionary mapping band name to numpy 2D/3D array.
        mask_cloud: Optional boolean mask where True indicates clouds/shadows (to be masked as NaN).
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'ndvi', 'ndre', 'evi', 'ndwi'.
    """
    blue = bands.get("B2", bands.get("blue"))
    green = bands.get("B3", bands.get("green"))
    red = bands.get("B4", bands.get("red"))
    red_edge = bands.get("B5", bands.get("red_edge_1", bands.get("red_edge")))
    nir = bands.get("B8", bands.get("nir"))
    
    if red is None or nir is None:
        raise ValueError("Both Red and NIR bands must be provided to compute vegetation indices.")
        
    indices = {
        "ndvi": compute_ndvi(nir, red)
    }
    
    if red_edge is not None:
        indices["ndre"] = compute_ndre(nir, red_edge)
        
    if blue is not None:
        indices["evi"] = compute_evi(nir, red, blue)
        
    if green is not None:
        indices["ndwi"] = compute_ndwi(green, nir)
        
    if mask_cloud is not None:
        for k in indices:
            indices[k][mask_cloud] = np.nan
            
    return indices
