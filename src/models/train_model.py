#!/usr/bin/env python3
"""
Model Training Script with MLflow Integration
Trains XGBoost model for taxi demand prediction
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import dagshub
import yaml
import json
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaxiDemandTrainer:
    """Trains taxi demand prediction models"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize DagsHub MLflow integration
        try:
            dagshub.init(repo_owner='YOUR_USERNAME', repo_name='nyc-taxi-demand-prediction', mlflow=True)
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            logger.info("✅ DagsHub MLflow integration initialized")
        except Exception as e:
            logger.warning(f"⚠️ DagsHub initialization failed: {e}")
            mlflow.set_tracking_uri("file:./mlruns")
            
    def load_data(self) -> tuple:
        """Load processed feature data"""
        logger.info("📥 Loading processed data...")
        
        # Load aggregated features
        features_path = Path("data/processed/aggregated_h3_features.parquet")
        features_df = pd.read_parquet(features_path)
        
        # Load weather data
        weather_path = Path("data/processed/weather_data.parquet")
        weather_df = pd.read_parquet(weather_path)
        
        # Merge datasets
        merged_df = features_df.merge(weather_df, on='time_window', how='left')
        
        # Fill missing weather data with defaults
        weather_columns = ['temperature_f', 'precipitation', 'wind_speed', 'humidity', 
                          'is_raining', 'is_cold', 'is_hot']
        for col in weather_columns:
            if col in merged_df.columns:
                if col == 'temperature_f':
                    merged_df[col] = merged_df[col].fillna(60.0)  # Default NYC temp
                elif col in ['is_raining', 'is_cold', 'is_hot']:
                    merged_df[col] = merged_df[col].fillna(0)
                else:
                    merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
        
        logger.info(f"✅ Loaded {len(merged_df):,} feature records")
        return merged_df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training"""
        logger.info("🔧 Preparing features...")
        
        # Get feature columns from config
        feature_columns = self.config['model']['features']
        
        # Ensure all required features exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
            # Remove missing features from config
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Add any missing features with defaults
        for col in missing_features:
            if col == 'event_impact':
                df[col] = 1.0  # Default event impact
            elif col in ['is_raining', 'is_cold']:
                df[col] = 0
            elif col == 'temperature_f':
                df[col] = 60.0
            elif col in ['ema3', 'trip_prev', 'demand_trend']:
                df[col] = df.groupby('pickup_h3')['trip_count'].transform('mean')
            else:
                df[col] = 0
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['trip_count'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        logger.info(f"✅ Prepared {len(feature_columns)} features for {len(X)} samples")
        return X, y, feature_columns
    
    def train_model(self, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info("🎯 Training XGBoost model...")
        
        # Model parameters from config
        params = self.config['model']['parameters']
        
        # Create and train model
        model = xgb.XGBRegressor(**params)
        
        # Fit model with validation set for early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        logger.info("✅ Model training completed")
        return model
    
    def evaluate_model(self, model, X_test, y_test, X_train, y_train) -> dict:
        """Evaluate model performance"""
        logger.info("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mape': mean_absolute_percentage_error(y_train, y_pred_train),
            'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"✅ Model evaluation completed - Test MAE: {metrics['test_mae']:.3f}")
        return metrics
    
    def save_model_artifacts(self, model, feature_columns: list, metrics: dict) -> str:
        """Save model artifacts"""
        logger.info("💾 Saving model artifacts...")
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata
        metadata = {
            'model_type': 'xgboost',
            'features': feature_columns,
            'feature_count': len(feature_columns),
            'training_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'parameters': self.config['model']['parameters']
        }
        
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        importance_dict = dict(zip(feature_columns, model.feature_importances_))
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Save feature importance plot data
        importance_path = plots_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        # Save metrics for DVC
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        performance_path = metrics_dir / "model_performance.json"
        with open(performance_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Model artifacts saved to {models_dir}")
        return str(model_path)
    
    def run(self) -> dict:
        """Run the complete training pipeline"""
        logger.info("🚀 Starting model training pipeline...")
        
        # Set MLflow experiment
        experiment_name = self.config['mlflow']['experiment_name']
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{self.config['mlflow']['run_name_prefix']}{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Load data
            df = self.load_data()
            
            # Prepare features
            X, y, feature_columns = self.prepare_features(df)
            
            # Train/validation/test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state']
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.config['training']['validation_size']/(1-self.config['training']['test_size']),
                random_state=self.config['training']['random_state']
            )
            
            # Log dataset information
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("features", feature_columns)
            
            # Log model parameters
            mlflow.log_params(self.config['model']['parameters'])
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log model to MLflow
            mlflow.xgboost.log_model(
                model, 
                "xgboost_model",
                registered_model_name=self.config.get('model_registry', {}).get('model_name', 'taxi-demand-xgboost')
            )
            
            # Save local artifacts
            model_path = self.save_model_artifacts(model, feature_columns, metrics)
            
            # Log artifacts
            mlflow.log_artifacts("models", "models")
            mlflow.log_artifacts("plots", "plots")
            mlflow.log_artifacts("metrics", "metrics")
            
            logger.info(f"✅ Training pipeline completed!")
            logger.info(f"📊 Test MAE: {metrics['test_mae']:.3f}")
            logger.info(f"📊 Test RMSE: {metrics['test_rmse']:.3f}")
            logger.info(f"📊 Test MAPE: {metrics['test_mape']:.3f}")
            logger.info(f"🔗 MLflow Run ID: {run.info.run_id}")
            
            return {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'model_path': model_path,
                'feature_columns': feature_columns
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train taxi demand prediction model')
    parser.add_argument('--config', default='config/model_config.yaml',
                       help='Path to model configuration file')
    args = parser.parse_args()
    
    try:
        trainer = TaxiDemandTrainer(args.config)
        results = trainer.run()
        
        logger.info("🎉 Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()