"""
Google Earth Engine Authentication & Initialization Module
"""
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def check_gee_authenticated() -> Tuple[bool, str]:
    """
    Check if Google Earth Engine is authenticated and initialized on the local system.
    
    Returns:
        Tuple[bool, str]: (is_authenticated, message)
    """
    try:
        import ee
    except ImportError:
        return False, "Google Earth Engine Python API (earthengine-api) is not installed. Install with `pip install earthengine-api`."

    try:
        ee.Initialize()
        return True, "Google Earth Engine successfully initialized."
    except Exception as e:
        msg = (
            "Google Earth Engine is not authenticated on this machine.\n"
            "To authenticate, run: `earthengine authenticate` or in Python `import ee; ee.Authenticate()`.\n"
            f"Details: {str(e)}"
        )
        return False, msg

def initialize_gee(project_id: Optional[str] = None) -> bool:
    """
    Attempt to initialize Google Earth Engine with optional cloud project ID.
    
    Args:
        project_id: Optional GCP Project ID registered with Earth Engine.
        
    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    try:
        import ee
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.info("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        logger.warning(f"GEE Initialization notice: {e}")
        return False
