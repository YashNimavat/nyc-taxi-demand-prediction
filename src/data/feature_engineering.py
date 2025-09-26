#!/usr/bin/env python3
"""
FIXED Feature Engineering Script - Based on Your Actual Model Features
Creates the same features as your trained XGBoost model
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
    """Creates features matching your trained XGBoost model"""
    
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
        """Aggregate trips into time windows - matching your notebook approach"""
        logger.info(f"🕒 Creating {self.time_window} time windows...")
        
        # Create time window column  
        df['time_window'] = df['pickup_datetime'].dt.floor(self.time_window)
        
        # Aggregate by time window and H3 cell (matching your notebook exactly)
        agg_df = df.groupby(['pickup_h3', 'time_window']).agg({
            'VendorID': 'count',  # trip count (matching your notebook)
            'trip_distance': ['mean', 'std', 'min', 'max'],
            'fare_amount': ['mean', 'std', 'min', 'max'], 
            'duration_minutes': ['mean', 'std'],
            'passenger_count': 'sum',  # total passengers (matching notebook)
            'hour': 'first',
            'day_of_week': 'first',
            'day_of_month': 'first',
            'day_of_year': 'first',
            'month': 'first',
            'quarter': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        # Flatten column names to match your notebook
        agg_df.columns = [
            'pickup_h3', 'time_window', 'trip_count',  # Renamed from VendorID count
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
        
        logger.info(f"✅ Created {len(agg_df):,} time window aggregations")
        return agg_df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features matching your trained model"""
        logger.info("📈 Adding lag features (EMA3, trip_prev, demand_trend)...")
        
        # Sort by location and time for proper lag calculation
        df = df.sort_values(['pickup_h3', 'time_window'])
        
        # EMA3 - Your model's TOP feature (53.8% importance)
        df['ema3'] = df.groupby('pickup_h3')['trip_count'].ewm(span=3, min_periods=1).mean().reset_index(0, drop=True)
        
        # trip_prev - Previous trip count (11.5% importance)
        df['trip_prev'] = df.groupby('pickup_h3')['trip_count'].shift(1)
        
        # demand_trend - Percent change in demand (used in your model)
        df['demand_trend'] = df.groupby('pickup_h3')['trip_count'].pct_change().fillna(0)
        df['demand_trend'] = df['demand_trend'].replace([np.inf, -np.inf], 0)
        
        # Fill NaN values with reasonable defaults
        df['trip_prev'] = df['trip_prev'].fillna(0)
        
        logger.info("✅ Added lag features matching your trained model")
        return df
    
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data matching your training data exactly"""
        logger.info("🌤️ Fetching weather data...")
        
        weather_config = self.weather_config
        
        params = {
            'latitude': weather_config['location']['latitude'],
            'longitude': weather_config['location']['longitude'], 
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(weather_config['variables']),
            'timezone': 'America/New_York',  # NYC timezone matching your data
            'temperature_unit': 'fahrenheit',
            'windspeed_unit': 'mph',
            'precipitation_unit': 'inch'
        }
        
        try:
            response = requests.get(weather_config['api_url'], params=params)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Create DataFrame matching your notebook structure
            weather_df = pd.DataFrame({
                'datetime': pd.to_datetime(weather_data['hourly']['time']),
                'temperature_f': weather_data['hourly']['temperature_2m'],
                'precipitation_inches': weather_data['hourly']['precipitation'],
                'wind_speed_mph': weather_data['hourly']['windspeed_10m'],
                'weather_code': weather_data['hourly']['weathercode']
            })
            
            # Create weather condition flags matching your training
            weather_df['is_raining'] = (weather_df['precipitation_inches'] > 0.01).astype(int)
            weather_df['is_cold'] = (weather_df['temperature_f'] < 35).astype(int)  # Your threshold
            
            # Weather demand multiplier matching your logic
            weather_df['weather_demand_multiplier'] = 1.0
            weather_df.loc[weather_df['is_raining'] == 1, 'weather_demand_multiplier'] = 1.25  # 25% for rain
            weather_df.loc[weather_df['is_cold'] == 1, 'weather_demand_multiplier'] = 1.1     # 10% for cold
            
            # Round to nearest time window
            weather_df['time_window'] = weather_df['datetime'].dt.floor(self.time_window)
            
            # Aggregate by time window
            weather_agg = weather_df.groupby('time_window').agg({
                'temperature_f': 'mean',
                'precipitation_inches': 'sum',
                'wind_speed_mph': 'mean',
                'weather_code': 'first',
                'is_raining': 'max',
                'is_cold': 'max', 
                'weather_demand_multiplier': 'mean'
            }).reset_index()
            
            logger.info(f"✅ Fetched weather data for {len(weather_agg)} time windows")
            return weather_agg
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch weather data: {e}")
            logger.info("Creating dummy weather data...")
            
            # Create dummy weather data matching your structure
            date_range = pd.date_range(start=start_date, end=end_date, freq=self.time_window)
            dummy_weather = pd.DataFrame({
                'time_window': date_range,
                'temperature_f': 60.0,  # Default NYC temperature
                'precipitation_inches': 0.0,
                'wind_speed_mph': 5.0,
                'weather_code': 0,
                'is_raining': 0,
                'is_cold': 0,
                'weather_demand_multiplier': 1.0
            })
            
            return dummy_weather
    
    def add_temporal_and_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and event features matching your trained model"""
        logger.info("🎉 Adding temporal and event features...")
        
        # Rush hour flags matching your model
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 3)).astype(int)
        
        # Calendar features
        df['weekday'] = df['day_of_week']  # Alias for compatibility
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Holiday detection (simplified)
        df['is_holiday'] = 0  # Will be enhanced with actual holiday data
        
        # Event impact simulation (matching your notebook's weekend evening events)
        df['event_impact'] = np.where(
            (df['hour'].isin([19, 20])) & (df['weekday'].isin([4, 5, 6])),  # 7-8PM on Fri/Sat/Sun
            1, 0
        )
        
        # Interaction features matching your model structure
        df['rush_weekend_interaction'] = df['is_rush_hour'].astype(int) * df['is_weekend'].astype(int)
        df['hour_weekday_interaction'] = df['hour'] * df['weekday']
        
        logger.info("✅ Added temporal and event features")
        return df
    
    def create_h3_one_hot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create H3 one-hot features matching your trained model"""
        logger.info("🗺️ Creating H3 one-hot features...")
        
        # Get the H3 cells that appear in your trained model (based on your notebook)
        # These are the top H3 cells from your feature importance
        important_h3_cells = [
            '882a100d2dfffff',  # Times Square area (11.8% importance)
            '882a100d65fffff',  # Upper East Side (7.9% importance) 
            '882a100895fffff',  # Financial District (6.8% importance)
            '882a100e07fffff',  # Upper West Side (1.4% importance)
            '882a107043fffff',  # Lower Manhattan area
            '882a1001abfffff',  # Bronx area
            '882a103b1dfffff'   # Brooklyn area
        ]
        
        # Create one-hot encoding for these specific H3 cells
        for h3_cell in important_h3_cells:
            df[f'h3_{h3_cell}'] = (df['pickup_h3'] == h3_cell).astype(int)
        
        logger.info(f"✅ Created {len(important_h3_cells)} H3 one-hot features")
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
            'h3_cells_in_model': [col for col in df.columns if col.startswith('h3_')],
            'feature_categories': {
                'temporal': [col for col in df.columns if any(x in col for x in ['hour', 'day', 'week', 'month', 'rush', 'weekend'])],
                'historical': ['ema3', 'trip_prev', 'demand_trend'],
                'weather': [col for col in df.columns if 'weather' in col or 'temperature' in col or 'rain' in col],
                'spatial': [col for col in df.columns if col.startswith('h3_')],
                'distance_fare': ['avg_dist', 'avg_fare'],
                'interaction': [col for col in df.columns if 'interaction' in col]
            },
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
        
        # Add lag features (EMA3, trip_prev, demand_trend)
        features_df = self.add_lag_features(features_df)
        
        # Get date range for weather data
        start_date = features_df['time_window'].min().date().isoformat()
        end_date = features_df['time_window'].max().date().isoformat()
        
        # Fetch weather data
        weather_df = self.fetch_weather_data(start_date, end_date)
        
        # Merge weather data
        features_df = features_df.merge(weather_df, on='time_window', how='left')
        
        # Fill missing weather data
        weather_columns = ['temperature_f', 'precipitation_inches', 'wind_speed_mph', 
                          'weather_code', 'is_raining', 'is_cold', 'weather_demand_multiplier']
        for col in weather_columns:
            if col in features_df.columns:
                if col == 'temperature_f':
                    features_df[col] = features_df[col].fillna(60.0)
                elif col in ['is_raining', 'is_cold']:
                    features_df[col] = features_df[col].fillna(0)
                elif col == 'weather_demand_multiplier':
                    features_df[col] = features_df[col].fillna(1.0)
                else:
                    features_df[col] = features_df[col].fillna(features_df[col].mean())
        
        # Add temporal and event features
        features_df = self.add_temporal_and_event_features(features_df)
        
        # Create H3 one-hot features
        features_df = self.create_h3_one_hot_features(features_df)
        
        # Save feature data
        features_path = Path("data/processed/aggregated_h3_features.parquet")
        features_df.to_parquet(features_path, index=False)
        
        # Save weather data separately
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
        logger.info(f"🎯 Created features matching your trained XGBoost model")
        
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