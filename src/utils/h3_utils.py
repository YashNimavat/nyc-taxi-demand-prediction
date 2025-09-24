#!/usr/bin/env python3
"""
H3 Utilities
Helper functions for H3 spatial indexing operations
"""

import h3
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def lat_lon_to_h3(lat: float, lon: float, resolution: int = 8) -> str:
    """Convert latitude/longitude to H3 cell"""
    try:
        return h3.geo_to_h3(lat, lon, resolution)
    except:
        return None

def h3_to_lat_lon(h3_cell: str) -> Tuple[float, float]:
    """Convert H3 cell to latitude/longitude"""
    try:
        lat, lon = h3.h3_to_geo(h3_cell)
        return lat, lon
    except:
        return None, None

def get_h3_neighbors(h3_cell: str, k_rings: int = 1) -> List[str]:
    """Get H3 neighbors within k rings"""
    try:
        neighbors = list(h3.k_ring(h3_cell, k_rings))
        return neighbors
    except:
        return []

def h3_distance(h3_cell1: str, h3_cell2: str) -> int:
    """Calculate distance between two H3 cells"""
    try:
        return h3.h3_distance(h3_cell1, h3_cell2)
    except:
        return -1

def get_h3_boundary(h3_cell: str) -> List[Tuple[float, float]]:
    """Get boundary coordinates for H3 cell"""
    try:
        boundary = h3.h3_to_geo_boundary(h3_cell)
        return [(lat, lon) for lat, lon in boundary]
    except:
        return []

def batch_lat_lon_to_h3(df: pd.DataFrame, lat_col: str, lon_col: str, 
                       resolution: int = 8, output_col: str = 'h3_cell') -> pd.DataFrame:
    """Convert latitude/longitude columns to H3 cells in batch"""
    logger.info(f"Converting {len(df)} coordinates to H3 resolution {resolution}")
    
    df[output_col] = df.apply(
        lambda row: lat_lon_to_h3(row[lat_col], row[lon_col], resolution), 
        axis=1
    )
    
    # Count successful conversions
    valid_conversions = df[output_col].notna().sum()
    logger.info(f"✅ Successfully converted {valid_conversions}/{len(df)} coordinates")
    
    return df

def get_h3_stats(h3_cells: List[str]) -> Dict:
    """Get statistics for H3 cells"""
    stats = {
        'total_cells': len(h3_cells),
        'unique_cells': len(set(h3_cells)),
        'resolution': h3.h3_get_resolution(h3_cells[0]) if h3_cells else None,
        'coverage_area_km2': len(set(h3_cells)) * h3.hex_area(h3.h3_get_resolution(h3_cells[0])) if h3_cells else 0
    }
    
    return stats