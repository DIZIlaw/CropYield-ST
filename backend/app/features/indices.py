"""
CropYield-ST: Optical Spectral Indices Module
Calculates physical vegetation indices from Sentinel-2 surface reflectance bands.
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
    Normalized Difference Vegetation Index (NDVI).
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    Physical relevance: Reflects chlorophyll absorption in red and mesophyll scattering in NIR.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    
    numerator = nir - red
    denominator = nir + red
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(np.abs(denominator) > eps, numerator / denominator, 0.0)
        
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
    Normalized Difference Red Edge Index (NDRE).
    Formula: NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    Physical relevance: Red-edge band (B5, 705 nm) penetrates deeper into dense canopies
    without the early saturation common to NDVI.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red_edge = np.asarray(red_edge, dtype=np.float32)
    
    numerator = nir - red_edge
    denominator = nir + red_edge
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndre = np.where(np.abs(denominator) > eps, numerator / denominator, 0.0)
        
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
    Enhanced Vegetation Index (EVI).
    Formula: EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    Physical relevance: Decouples background soil canopy scattering and corrects aerosol resistance.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)
    
    numerator = nir - red
    denominator = nir + (c1 * red) - (c2 * blue) + l
    
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = np.where(np.abs(denominator) > eps, g * (numerator / denominator), 0.0)
        
    if clip_valid:
        evi = np.clip(evi, -1.0, 1.5)
        
    return evi


def compute_ndwi(
    nir: np.ndarray,
    swir: np.ndarray,
    eps: float = 1e-8,
    clip_valid: bool = True
) -> np.ndarray:
    """
    Normalized Difference Water Index (NDWI / NDII - Gao 1996).
    Formula: NDWI = (NIR - SWIR) / (NIR + SWIR)
    Physical relevance: Reflects liquid water content in plant leaves.
    """
    nir = np.asarray(nir, dtype=np.float32)
    swir = np.asarray(swir, dtype=np.float32)
    
    numerator = nir - swir
    denominator = nir + swir
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(np.abs(denominator) > eps, numerator / denominator, 0.0)
        
    if clip_valid:
        ndwi = np.clip(ndwi, -1.0, 1.0)
        
    return ndwi


def compute_all_optical_indices(
    bands: Dict[str, np.ndarray],
    mask_cloud: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> Dict[str, np.ndarray]:
    """
    Computes all eligible optical vegetation and water indices from a band dictionary.
    Keys expected (case-insensitive):
      - 'B8' or 'nir'
      - 'B4' or 'red'
      - 'B2' or 'blue' (optional for EVI)
      - 'B5' or 'red_edge' (optional for NDRE)
      - 'B11' or 'swir' or 'B3' / 'green' (optional for NDWI)
    """
    lookup = {k.upper(): v for k, v in bands.items()}
    
    nir = lookup.get("B8") if "B8" in lookup else lookup.get("NIR")
    red = lookup.get("B4") if "B4" in lookup else lookup.get("RED")
    blue = lookup.get("B2") if "B2" in lookup else lookup.get("BLUE")
    red_edge = lookup.get("B5") if "B5" in lookup else lookup.get("RED_EDGE")
    swir = lookup.get("B11") if "B11" in lookup else lookup.get("SWIR", lookup.get("B3", lookup.get("GREEN")))
    
    results = {}
    if nir is not None and red is not None:
        results["ndvi"] = compute_ndvi(nir, red, eps=eps)
        if blue is not None:
            results["evi"] = compute_evi(nir, red, blue, eps=eps)
        if red_edge is not None:
            results["ndre"] = compute_ndre(nir, red_edge, eps=eps)
        if swir is not None:
            results["ndwi"] = compute_ndwi(nir, swir, eps=eps)
            
    if mask_cloud is not None:
        for k in results:
            results[k] = np.where(mask_cloud, np.nan, results[k])
            
    return results
