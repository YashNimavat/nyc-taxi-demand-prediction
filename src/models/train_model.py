#!/usr/bin/env python3

"""
FIXED Model Training Script - Uses consolidated features
Loads the consolidated feature file that includes all weather data
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
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
    """Trains taxi demand prediction models - FIXED for consolidated features"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # SIMPLIFIED: Only use local MLflow to avoid DagsHub API issues
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("✅ Using local MLflow tracking")

    def load_data(self) -> pd.DataFrame:
        """Load consolidated feature data - FIXED"""
        logger.info("📥 Loading consolidated feature data...")
        
        # Load consolidated features (includes weather data already merged)
        features_path = Path("data/processed/aggregated_h3_features.parquet")
        merged_df = pd.read_parquet(features_path)
        
        logger.info(f"✅ Loaded {len(merged_df):,} consolidated feature records")
        return merged_df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training - FIXED"""
        logger.info("🔧 Preparing features...")
        
        # Get exact feature columns from config
        feature_columns = self.config['model']['features']
        
        # Ensure all required features exist with defaults
        required_defaults = {
            'temperature_f': 60.0,
            'is_raining': 0,
            'is_cold': 0,
            'is_rush_hour': 0,
            'weekday': 0,
            'ema3': 0,
            'trip_prev': 0,
            'demand_trend': 0
        }
        
        for feature, default_val in required_defaults.items():
            if feature not in df.columns:
                logger.warning(f"Adding missing feature {feature} with default value {default_val}")
                df[feature] = default_val

        # Add weekday as alias for day_of_week if needed
        if 'weekday' not in df.columns and 'day_of_week' in df.columns:
            df['weekday'] = df['day_of_week']

        # Add is_rush_hour if missing
        if 'is_rush_hour' not in df.columns and 'hour' in df.columns:
            df['is_rush_hour'] = (df['hour'].isin([7, 8, 9, 17, 18, 19])).astype(int)

        # Filter to only include features that exist and are needed
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) != len(feature_columns):
            missing_features = set(feature_columns) - set(available_features)
            logger.warning(f"Missing features (will use defaults): {missing_features}")

        # Prepare features and target
        X = df[available_features].copy()
        y = df['trip_count'].copy()

        # Handle missing values
        X = X.fillna(0)

        logger.info(f"✅ Prepared {len(available_features)} features for {len(X)} samples")
        logger.info(f"📊 Features: {available_features}")
        return X, y, available_features

    def train_model(self, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info("🎯 Training XGBoost model...")
        
        params = self.config['model']['parameters']
        model = xgb.XGBRegressor(**params)
        
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
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'train_mape': float(mean_absolute_percentage_error(y_train, y_pred_train)),
            'test_mape': float(mean_absolute_percentage_error(y_test, y_pred_test)),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'test_r2': float(r2_score(y_test, y_pred_test))
        }

        logger.info(f"✅ Model evaluation completed - Test MAE: {metrics['test_mae']:.3f}")
        return metrics

    def save_model_artifacts(self, model, feature_columns: list, metrics: dict) -> str:
        """Save model artifacts"""
        logger.info("💾 Saving model artifacts...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Save model
        model_path = models_dir / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = {
            'model_type': 'xgboost',
            'features': feature_columns,
            'feature_count': len(feature_columns),
            'training_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'parameters': self.config['model']['parameters']
        }

        with open(models_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance (FIXED: Convert numpy types)
        importance_dict = {
            feature: float(importance)
            for feature, importance in zip(feature_columns, model.feature_importances_)
        }

        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        with open(plots_dir / "feature_importance.json", 'w') as f:
            json.dump(importance_dict, f, indent=2)

        # Save metrics
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        with open(metrics_dir / "model_performance.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ Model artifacts saved to {models_dir}")
        return str(model_path)

    def run(self) -> dict:
        """Run the complete training pipeline"""
        logger.info("🚀 Starting model training pipeline...")

        experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'taxi-demand-prediction')
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"xgboost-v{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Load consolidated data
            df = self.load_data()

            # Prepare features
            X, y, feature_columns = self.prepare_features(df)

            # Train/validation/test split
            test_size = self.config.get('training', {}).get('test_size', 0.2)
            val_size = self.config.get('training', {}).get('validation_size', 0.1)
            random_state = self.config.get('training', {}).get('random_state', 42)

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state
            )

            # Log parameters
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_params(self.config['model']['parameters'])

            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val)

            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)

            # Log metrics
            mlflow.log_metrics(metrics)

            # SIMPLIFIED: Just log the model without registry (avoids DagsHub API issues)
            mlflow.sklearn.log_model(model, "xgboost_model")

            # Save local artifacts
            model_path = self.save_model_artifacts(model, feature_columns, metrics)

            logger.info(f"✅ Training pipeline completed!")
            logger.info(f"📊 Test MAE: {metrics['test_mae']:.3f}")
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