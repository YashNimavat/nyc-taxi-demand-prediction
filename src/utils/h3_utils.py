#!/usr/bin/env python3

"""
H3 Feature Computing Functions - Extends basic H3 utilities
Used by feature_engineering.py for spatial feature creation
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

def compute_h3_features(trips_df: pd.DataFrame, time_window: str = "15T") -> pd.DataFrame:
    """
    Main function to compute aggregated H3 features from trip data
    This is used by feature_engineering.py
    """
    logger.info(f"🧮 Computing H3 features with {time_window} time windows...")
    
    # Create time window column
    trips_df['time_window'] = trips_df['pickup_datetime'].dt.floor(time_window)
    
    # Aggregate by time window and H3 cell
    agg_df = trips_df.groupby(['pickup_h3', 'time_window']).agg({
        'VendorID': 'count',  # trip count
        'trip_distance': ['mean', 'std', 'min', 'max'],
        'fare_amount': ['mean', 'std', 'min', 'max'],
        'duration_minutes': ['mean', 'std'],
        'passenger_count': 'sum',
        'hour': 'first',
        'day_of_week': 'first',
        'day_of_month': 'first',
        'day_of_year': 'first',
        'month': 'first',
        'quarter': 'first',
        'is_weekend': 'first'
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        'pickup_h3', 'time_window', 'trip_count',
        'avg_dist', 'std_distance', 'min_distance', 'max_distance',
        'avg_fare', 'std_fare', 'min_fare', 'max_fare',
        'avg_duration', 'std_duration', 'total_passengers',
        'hour', 'day_of_week', 'day_of_month', 'day_of_year',
        'month', 'quarter', 'is_weekend'
    ]
    
    # Fill NaN values for standard deviations
    agg_df['std_distance'] = agg_df['std_distance'].fillna(0)
    agg_df['std_fare'] = agg_df['std_fare'].fillna(0)
    agg_df['std_duration'] = agg_df['std_duration'].fillna(0)
    
    # Sort by location and time for lag calculations
    agg_df = agg_df.sort_values(['pickup_h3', 'time_window'])
    
    # Add lag features
    agg_df['ema3'] = agg_df.groupby('pickup_h3')['trip_count'].ewm(span=3, min_periods=1).mean().reset_index(0, drop=True)
    agg_df['trip_prev'] = agg_df.groupby('pickup_h3')['trip_count'].shift(1).fillna(0)
    agg_df['demand_trend'] = agg_df.groupby('pickup_h3')['trip_count'].pct_change().fillna(0)
    agg_df['demand_trend'] = agg_df['demand_trend'].replace([np.inf, -np.inf], 0)
    
    logger.info(f"✅ Computed features for {len(agg_df)} time windows")
    return agg_df