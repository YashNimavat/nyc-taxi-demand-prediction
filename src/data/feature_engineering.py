#!/usr/bin/env python3
"""
Feature Engineering Script
Creates aggregated features for taxi demand prediction
"""

import pandas as pd
import numpy as np
import logging
import yaml
import json
import h3
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaxiFeatureEngineer:
    """Creates features for taxi demand prediction"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.time_window = self.config['data']['processing']['time_window']
        self.weather_config = self.config['data']['feature_engineering']['weather']
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned trip data"""
        data_path = Path("data/processed/cleaned_trips.parquet")
        df = pd.read_parquet(data_path)
        logger.info(f"✅ Loaded {len(df):,} cleaned trips")
        return df
    
    def create_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trips into time windows"""
        logger.info(f"🕒 Creating {self.time_window} time windows...")
        
        # Create time window column
        df['time_window'] = df['pickup_datetime'].dt.floor(self.time_window)
        
        # Aggregate by time window and H3 cell
        agg_df = df.groupby(['time_window', 'pickup_h3']).agg({
            'pickup_datetime': 'count',  # trip count
            'trip_distance': ['mean', 'std', 'min', 'max'],
            'fare_amount': ['mean', 'std', 'min', 'max'], 
            'duration_minutes': ['mean', 'std'],
            'passenger_count': 'mean',
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
            'time_window', 'pickup_h3', 'trip_count',
            'avg_distance', 'std_distance', 'min_distance', 'max_distance',
            'avg_fare', 'std_fare', 'min_fare', 'max_fare',
            'avg_duration', 'std_duration', 'avg_passengers',
            'hour', 'day_of_week', 'day_of_month', 'day_of_year',
            'month', 'quarter', 'is_weekend'
        ]
        
        # Fill NaN values for standard deviations
        agg_df['std_distance'] = agg_df['std_distance'].fillna(0)
        agg_df['std_fare'] = agg_df['std_fare'].fillna(0)
        agg_df['std_duration'] = agg_df['std_duration'].fillna(0)
        
        logger.info(f"✅ Created {len(agg_df):,} time window aggregations")
        return agg_df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged trip count features"""
        logger.info("📈 Adding lag features...")
        
        # Sort by location and time
        df = df.sort_values(['pickup_h3', 'time_window'])
        
        # Create lag features for each H3 cell
        lookback_windows = self.config['data']['feature_engineering']['aggregation_features']['lookback_windows']
        
        for window in lookback_windows:
            # Previous trip count
            df[f'trip_count_lag_{window}'] = df.groupby('pickup_h3')['trip_count'].shift(window)
            
            # Rolling mean
            df[f'trip_count_rolling_mean_{window}'] = (
                df.groupby('pickup_h3')['trip_count']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
        
        # Exponential moving average (EMA)
        df['ema3'] = (
            df.groupby('pickup_h3')['trip_count']
            .ewm(span=3)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Previous period comparison
        df['trip_prev'] = df.groupby('pickup_h3')['trip_count'].shift(1)
        df['demand_trend'] = (df['trip_count'] - df['trip_prev']).fillna(0)
        
        # Fill NaN values
        lag_columns = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        df[lag_columns] = df.groupby('pickup_h3')[lag_columns].fillna(method='bfill')
        df[lag_columns] = df[lag_columns].fillna(0)
        
        logger.info("✅ Added lag features")
        return df
    
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data from Open-Meteo API"""
        logger.info("🌤️ Fetching weather data...")
        
        weather_config = self.weather_config
        
        params = {
            'latitude': weather_config['location']['latitude'],
            'longitude': weather_config['location']['longitude'], 
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(weather_config['variables']),
            'timezone': 'UTC'
        }
        
        try:
            response = requests.get(weather_config['api_url'], params=params)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Create DataFrame
            weather_df = pd.DataFrame({
                'datetime': pd.to_datetime(weather_data['hourly']['time']),
                'temperature_c': weather_data['hourly']['temperature_2m'],
                'precipitation': weather_data['hourly']['precipitation'],
                'wind_speed': weather_data['hourly']['windspeed_10m'],
                'humidity': weather_data['hourly']['relativehumidity_2m']
            })
            
            # Convert to Fahrenheit
            weather_df['temperature_f'] = (weather_df['temperature_c'] * 9/5) + 32
            
            # Create weather condition flags
            weather_df['is_raining'] = (weather_df['precipitation'] > 0.1).astype(int)
            weather_df['is_cold'] = (weather_df['temperature_f'] < 40).astype(int)
            weather_df['is_hot'] = (weather_df['temperature_f'] > 80).astype(int)
            
            # Round to nearest time window
            weather_df['time_window'] = weather_df['datetime'].dt.floor(self.time_window)
            
            # Aggregate by time window
            weather_agg = weather_df.groupby('time_window').agg({
                'temperature_f': 'mean',
                'precipitation': 'sum',
                'wind_speed': 'mean',
                'humidity': 'mean',
                'is_raining': 'max',
                'is_cold': 'max', 
                'is_hot': 'max'
            }).reset_index()
            
            logger.info(f"✅ Fetched weather data for {len(weather_agg)} time windows")
            return weather_agg
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch weather data: {e}")
            logger.info("Creating dummy weather data...")
            
            # Create dummy weather data
            date_range = pd.date_range(start=start_date, end=end_date, freq=self.time_window)
            dummy_weather = pd.DataFrame({
                'time_window': date_range,
                'temperature_f': 60.0,  # Default NYC temperature
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'humidity': 50.0,
                'is_raining': 0,
                'is_cold': 0,
                'is_hot': 0
            })
            
            return dummy_weather
    
    def add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event and holiday features"""
        logger.info("🎉 Adding event features...")
        
        # Simple event impact based on hour and day
        df['rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19]))).astype(int)
        df['late_night'] = ((df['hour'].isin([22, 23, 0, 1, 2, 3]))).astype(int)
        df['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Weekend vs weekday impact
        df['weekend_night'] = (df['is_weekend'] & df['late_night']).astype(int)
        df['weekday_rush'] = ((1 - df['is_weekend']) & df['rush_hour']).astype(int)
        
        # Simple event impact score
        df['event_impact'] = (
            df['rush_hour'] * 1.5 + 
            df['weekend_night'] * 1.2 + 
            df['business_hours'] * 1.0
        )
        
        logger.info("✅ Added event features")
        return df
    
    def generate_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Generate feature statistics"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isnull().sum())
            }
        
        feature_stats = {
            'total_features': len(df.columns),
            'total_rows': len(df),
            'time_range': {
                'start': df['time_window'].min().isoformat(),
                'end': df['time_window'].max().isoformat()
            },
            'unique_locations': int(df['pickup_h3'].nunique()),
            'feature_statistics': stats
        }
        
        return feature_stats
    
    def run(self) -> Dict:
        """Run the feature engineering pipeline"""
        logger.info("🚀 Starting feature engineering...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Create time window aggregations
        features_df = self.create_time_windows(df)
        
        # Add lag features
        features_df = self.add_lag_features(features_df)
        
        # Get date range for weather data
        start_date = features_df['time_window'].min().date().isoformat()
        end_date = features_df['time_window'].max().date().isoformat()
        
        # Fetch weather data
        weather_df = self.fetch_weather_data(start_date, end_date)
        
        # Add event features
        features_df = self.add_event_features(features_df)
        
        # Save feature data
        features_path = Path("data/processed/aggregated_h3_features.parquet")
        features_df.to_parquet(features_path, index=False)
        
        # Save weather data
        weather_path = Path("data/processed/weather_data.parquet")
        weather_df.to_parquet(weather_path, index=False)
        
        # Generate feature statistics
        feature_stats = self.generate_feature_stats(features_df)
        
        # Save statistics
        stats_path = Path("metrics/feature_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        logger.info(f"✅ Feature engineering complete!")
        logger.info(f"💾 Saved {len(features_df):,} feature records to {features_path}")
        logger.info(f"🌤️ Saved {len(weather_df):,} weather records to {weather_path}")
        logger.info(f"📊 Feature stats saved to {stats_path}")
        
        return feature_stats

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Engineer features for taxi demand prediction')
    parser.add_argument('--config', default='config/data_config.yaml',
                       help='Path to data configuration file')
    args = parser.parse_args()
    
    try:
        engineer = TaxiFeatureEngineer(args.config)
        stats = engineer.run()
        
        logger.info("🎉 Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()