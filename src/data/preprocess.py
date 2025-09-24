#!/usr/bin/env python3
"""
Data Preprocessing Script
Cleans and filters NYC taxi data for feature engineering
"""

import pandas as pd
import numpy as np
import logging
import yaml
import h3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaxiDataPreprocessor:
    """Preprocesses NYC taxi trip data"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.h3_resolution = self.config['data']['processing']['h3_resolution']
        self.quality_filters = self.config['data']['quality_filters']
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load all raw trip data files"""
        dfs = []
        raw_dir = Path("data/raw")
        
        # Load all parquet files
        for parquet_file in raw_dir.glob("yellow_tripdata_*.parquet"):
            logger.info(f"Loading {parquet_file.name}...")
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✅ Loaded {len(combined_df):,} total trips")
        
        return combined_df
    
    def load_zone_lookup(self) -> pd.DataFrame:
        """Load taxi zone lookup table"""
        zone_path = Path("data/raw/taxi_zone_lookup.csv")
        zones_df = pd.read_csv(zone_path)
        logger.info(f"✅ Loaded {len(zones_df)} taxi zones")
        return zones_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        initial_count = len(df)
        logger.info(f"🧹 Cleaning {initial_count:,} trips...")
        
        # Filter by trip distance
        df = df[
            (df['trip_distance'] >= self.quality_filters['trip_distance'][0]) &
            (df['trip_distance'] <= self.quality_filters['trip_distance'][1])
        ]
        
        # Filter by fare amount
        df = df[
            (df['fare_amount'] >= self.quality_filters['fare_amount'][0]) &
            (df['fare_amount'] <= self.quality_filters['fare_amount'][1])
        ]
        
        # Filter by passenger count
        df = df[
            (df['passenger_count'] >= self.quality_filters['passenger_count'][0]) &
            (df['passenger_count'] <= self.quality_filters['passenger_count'][1])
        ]
        
        # Calculate trip duration and filter
        df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        df['duration_minutes'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
        
        df = df[
            (df['duration_minutes'] >= self.quality_filters['duration_minutes'][0]) &
            (df['duration_minutes'] <= self.quality_filters['duration_minutes'][1])
        ]
        
        # Remove invalid coordinates
        spatial_config = self.config['data']['spatial']
        df = df[
            (df['pickup_latitude'] >= spatial_config['lat_min']) &
            (df['pickup_latitude'] <= spatial_config['lat_max']) &
            (df['pickup_longitude'] >= spatial_config['lon_min']) &
            (df['pickup_longitude'] <= spatial_config['lon_max'])
        ]
        
        # Remove nulls in critical columns
        critical_columns = ['pickup_latitude', 'pickup_longitude', 'fare_amount', 'trip_distance']
        df = df.dropna(subset=critical_columns)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        removal_rate = (removed_count / initial_count) * 100
        
        logger.info(f"✅ Cleaned data: {final_count:,} trips ({removal_rate:.1f}% removed)")
        
        return df
    
    def add_h3_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add H3 hexagon cell identifiers"""
        logger.info("🗺️ Adding H3 spatial indexing...")
        
        def lat_lon_to_h3(lat, lon):
            try:
                return h3.geo_to_h3(lat, lon, self.h3_resolution)
            except:
                return None
        
        df['pickup_h3'] = df.apply(
            lambda row: lat_lon_to_h3(row['pickup_latitude'], row['pickup_longitude']), 
            axis=1
        )
        
        # Remove rows where H3 conversion failed
        initial_count = len(df)
        df = df.dropna(subset=['pickup_h3'])
        final_count = len(df)
        
        logger.info(f"✅ Added H3 cells: {final_count:,} trips ({len(df['pickup_h3'].unique()):,} unique cells)")
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        logger.info("⏰ Adding temporal features...")
        
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['day_of_month'] = df['pickup_datetime'].dt.day
        df['day_of_year'] = df['pickup_datetime'].dt.dayofyear
        df['week_of_year'] = df['pickup_datetime'].dt.isocalendar().week
        df['month'] = df['pickup_datetime'].dt.month
        df['quarter'] = df['pickup_datetime'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info("✅ Added temporal features")
        return df
    
    def generate_quality_metrics(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Generate data quality metrics"""
        metrics = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'removal_rate': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100,
            'unique_h3_cells': int(cleaned_df['pickup_h3'].nunique()),
            'date_range': {
                'start': cleaned_df['pickup_datetime'].min().isoformat(),
                'end': cleaned_df['pickup_datetime'].max().isoformat()
            },
            'statistics': {
                'avg_trip_distance': float(cleaned_df['trip_distance'].mean()),
                'avg_fare_amount': float(cleaned_df['fare_amount'].mean()),
                'avg_duration_minutes': float(cleaned_df['duration_minutes'].mean())
            },
            'quality_checks': {
                'null_coordinates': int((cleaned_df[['pickup_latitude', 'pickup_longitude']].isnull()).sum().sum()),
                'invalid_fares': int((cleaned_df['fare_amount'] <= 0).sum()),
                'zero_distance_trips': int((cleaned_df['trip_distance'] <= 0).sum())
            }
        }
        
        return metrics
    
    def run(self) -> Dict:
        """Run the preprocessing pipeline"""
        logger.info("🚀 Starting data preprocessing...")
        
        # Load raw data
        raw_df = self.load_raw_data()
        zones_df = self.load_zone_lookup()
        
        # Clean data
        cleaned_df = self.clean_data(raw_df)
        
        # Add spatial features
        cleaned_df = self.add_h3_cells(cleaned_df)
        
        # Add temporal features
        cleaned_df = self.add_temporal_features(cleaned_df)
        
        # Save processed data
        output_path = Path("data/processed/cleaned_trips.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_parquet(output_path, index=False)
        
        # Generate quality metrics
        quality_metrics = self.generate_quality_metrics(raw_df, cleaned_df)
        
        # Save metrics
        metrics_path = Path("metrics/data_quality.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        logger.info(f"✅ Preprocessing complete!")
        logger.info(f"💾 Saved {len(cleaned_df):,} cleaned trips to {output_path}")
        logger.info(f"📊 Quality metrics saved to {metrics_path}")
        
        return quality_metrics

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Preprocess NYC taxi data')
    parser.add_argument('--config', default='config/data_config.yaml',
                       help='Path to data configuration file')
    args = parser.parse_args()
    
    try:
        preprocessor = TaxiDataPreprocessor(args.config)
        metrics = preprocessor.run()
        
        logger.info("🎉 Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()