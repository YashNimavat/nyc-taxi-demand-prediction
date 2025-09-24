#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained model performance with detailed analysis
"""

import pandas as pd
import numpy as np
import pickle
import json
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates taxi demand prediction model"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_model(self) -> object:
        """Load trained model"""
        model_path = Path("models/xgboost_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Loaded trained model")
        return model
    
    def load_test_data(self) -> tuple:
        """Load test data for evaluation"""
        # Load processed data
        features_path = Path("data/processed/aggregated_h3_features.parquet")
        features_df = pd.read_parquet(features_path)
        
        weather_path = Path("data/processed/weather_data.parquet")
        weather_df = pd.read_parquet(weather_path)
        
        # Merge datasets
        merged_df = features_df.merge(weather_df, on='time_window', how='left')
        
        # Fill missing values
        weather_columns = ['temperature_f', 'precipitation', 'wind_speed', 'humidity', 
                          'is_raining', 'is_cold', 'is_hot']
        for col in weather_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean() if col not in ['is_raining', 'is_cold', 'is_hot'] else 0)
        
        # Get features
        feature_columns = self.config['model']['features']
        available_features = [col for col in feature_columns if col in merged_df.columns]
        
        X = merged_df[available_features]
        y = merged_df['trip_count']
        
        return X, y, merged_df
    
    def detailed_evaluation(self, model, X, y, df) -> dict:
        """Perform detailed model evaluation"""
        logger.info("📊 Performing detailed evaluation...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Basic metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Add predictions to dataframe
        df = df.copy()
        df['predicted'] = y_pred
        df['actual'] = y
        df['residual'] = y - y_pred
        df['abs_error'] = np.abs(df['residual'])
        df['pct_error'] = (df['residual'] / np.maximum(df['actual'], 1)) * 100
        
        # Temporal analysis
        temporal_metrics = df.groupby('hour').agg({
            'abs_error': 'mean',
            'pct_error': 'mean'
        }).to_dict()
        
        # Spatial analysis
        spatial_metrics = df.groupby('pickup_h3').agg({
            'abs_error': 'mean',
            'pct_error': 'mean'
        }).describe().to_dict()
        
        # Error distribution analysis
        error_percentiles = np.percentile(df['abs_error'], [25, 50, 75, 90, 95, 99])
        
        evaluation_results = {
            'overall_metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2)
            },
            'error_analysis': {
                'mean_abs_error': float(df['abs_error'].mean()),
                'median_abs_error': float(df['abs_error'].median()),
                'error_std': float(df['abs_error'].std()),
                'error_percentiles': {
                    'p25': float(error_percentiles[0]),
                    'p50': float(error_percentiles[1]),
                    'p75': float(error_percentiles[2]),
                    'p90': float(error_percentiles[3]),
                    'p95': float(error_percentiles[4]),
                    'p99': float(error_percentiles[5])
                }
            },
            'temporal_performance': temporal_metrics,
            'spatial_performance_stats': spatial_metrics,
            'prediction_range': {
                'min_predicted': float(df['predicted'].min()),
                'max_predicted': float(df['predicted'].max()),
                'min_actual': float(df['actual'].min()),
                'max_actual': float(df['actual'].max())
            }
        }
        
        logger.info(f"✅ Detailed evaluation completed - Overall MAE: {mae:.3f}")
        return evaluation_results, df
    
    def create_evaluation_plots(self, df: pd.DataFrame, model) -> dict:
        """Create evaluation plots"""
        logger.info("📈 Creating evaluation plots...")
        
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        plot_data = {}
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['actual'], df['predicted'], alpha=0.5, s=20)
        
        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Trip Count')
        plt.ylabel('Predicted Trip Count')
        plt.title('Actual vs Predicted Trip Counts')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residuals plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(df['predicted'], df['residual'], alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Trip Count')
        plt.ylabel('Residual')
        plt.title('Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(df['residual'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error by hour
        plt.figure(figsize=(12, 6))
        hourly_error = df.groupby('hour')['abs_error'].mean()
        plt.bar(hourly_error.index, hourly_error.values, alpha=0.7)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.title('Model Error by Hour of Day')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        plt.savefig(plots_dir / 'error_by_hour.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save plot data for DVC
        plot_data['residuals'] = {
            'residuals': df['residual'].tolist()[:1000],  # Limit for JSON size
            'predictions': df['predicted'].tolist()[:1000]
        }
        
        plot_data['hourly_error'] = hourly_error.to_dict()
        
        # Save to plots directory for DVC tracking
        with open(plots_dir / 'prediction_analysis.json', 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        logger.info("✅ Evaluation plots created")
        return plot_data
    
    def run(self) -> dict:
        """Run the evaluation pipeline"""
        logger.info("🚀 Starting model evaluation...")
        
        # Load model and data
        model = self.load_model()
        X, y, df = self.load_test_data()
        
        # Perform detailed evaluation
        evaluation_results, df_with_predictions = self.detailed_evaluation(model, X, y, df)
        
        # Create plots
        plot_data = self.create_evaluation_plots(df_with_predictions, model)
        
        # Save evaluation results
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        eval_path = metrics_dir / "model_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"✅ Model evaluation completed!")
        logger.info(f"📊 MAE: {evaluation_results['overall_metrics']['mae']:.3f}")
        logger.info(f"📊 RMSE: {evaluation_results['overall_metrics']['rmse']:.3f}")
        logger.info(f"📊 MAPE: {evaluation_results['overall_metrics']['mape']:.3f}")
        logger.info(f"📊 R²: {evaluation_results['overall_metrics']['r2_score']:.3f}")
        logger.info(f"💾 Results saved to {eval_path}")
        
        return evaluation_results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate taxi demand prediction model')
    parser.add_argument('--config', default='config/model_config.yaml',
                       help='Path to model configuration file')
    args = parser.parse_args()
    
    try:
        evaluator = ModelEvaluator(args.config)
        results = evaluator.run()
        
        logger.info("🎉 Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()