#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating Prophet forecasting pipeline usage.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data import DataManager
from models import ProphetModelWrapper
from training import TrainingPipeline
from evaluation import ModelEvaluator
from utils import ReportGenerator, Visualizer, ArtifactManager


def create_sample_data():
    """Create sample time series data for demonstration."""
    print("Creating sample time series data...")
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=60, freq='MS')  # 5 years of monthly data
    
    # Generate target series with trend and seasonality
    trend = np.linspace(1000, 1200, len(dates))
    seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # Annual seasonality
    noise = np.random.normal(0, 50, len(dates))
    target_values = trend + seasonal + noise
    
    # Create target dataframe
    target_df = pd.DataFrame({
        'DATA': dates,
        'TOTAL_CASOS': target_values
    })
    
    # Generate exogenous features
    feature1 = np.random.normal(50, 10, len(dates))
    feature2 = np.random.normal(20, 5, len(dates))
    feature3 = np.random.normal(100, 15, len(dates))
    
    # Create features dataframe
    features_df = pd.DataFrame({
        'data': dates,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3
    })
    
    # Save sample data
    target_df.to_csv('sample_target_data.csv', index=False)
    features_df.to_csv('sample_features_data.csv', index=False)
    
    print(f"Created sample data:")
    print(f"  Target series: {len(target_df)} rows")
    print(f"  Features: {len(features_df)} rows")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    return target_df, features_df


def create_sample_config():
    """Create sample configuration for the pipeline."""
    config = {
        'data': {
            'target_series_path': 'sample_target_data.csv',
            'exogenous_features_path': 'sample_features_data.csv',
            'target_column': 'TOTAL_CASOS',
            'date_column': 'DATA',
            'preprocessing': {
                'remove_outliers': True,
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0,
                'fill_missing': True,
                'fill_method': 'forward_fill'
            }
        },
        'model': {
            'prophet_params': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'daily_seasonality': False,
                'weekly_seasonality': True,
                'yearly_seasonality': True
            },
            'exogenous_vars': {
                'enabled': True,
                'features': ['feature1', 'feature2', 'feature3']
            },
            'holidays': {
                'enabled': False,
                'country': 'BR',
                'custom_holidays': []
            }
        },
        'training': {
            'cv': {
                'n_splits': 3,
                'test_size': 0.2,
                'gap': 0
            },
            'hyperparameter_optimization': {
                'enabled': False,  # Disable for quick demo
                'n_trials': 10,
                'timeout': 300
            },
            'train_start': '2020-01-01',
            'train_end': '2023-12-01'
        },
        'forecasting': {
            'horizon_months': 12,
            'prediction_intervals': {
                'enabled': True,
                'intervals': [0.8, 0.95]
            },
            'uncertainty_samples': 1000
        },
        'evaluation': {
            'metrics': ['mae', 'mse', 'rmse', 'mape', 'smape', 'r2'],
            'component_analysis': {
                'enabled': True,
                'components': ['trend', 'seasonal', 'holidays', 'exogenous']
            }
        },
        'output': {
            'base_dir': 'example_outputs',
            'save_model': True,
            'save_predictions': True,
            'save_metrics': True,
            'save_plots': True,
            'plots': {
                'forecast_plot': True,
                'components_plot': True,
                'cross_validation_plot': True,
                'residuals_plot': True
            },
            'reports': {
                'enabled': True,
                'format': 'html',
                'include_metrics': True,
                'include_plots': True,
                'include_components': True
            }
        },
        'logging': {
            'level': 'INFO',
            'file': 'example_outputs/logs/example_pipeline.log',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}'
        },
        'random_seed': 42
    }
    
    return config


def run_example_pipeline():
    """Run the complete example pipeline."""
    print("="*60)
    print("PROPHET FORECASTING PIPELINE - EXAMPLE")
    print("="*60)
    
    try:
        # Step 1: Create sample data
        target_df, features_df = create_sample_data()
        
        # Step 2: Create configuration
        config = create_sample_config()
        
        # Step 3: Initialize components
        print("\nInitializing pipeline components...")
        data_manager = DataManager(config)
        training_pipeline = TrainingPipeline(config)
        evaluator = ModelEvaluator(config)
        visualizer = Visualizer(config)
        artifact_manager = ArtifactManager(config)
        report_generator = ReportGenerator(config)
        
        # Step 4: Load and prepare data
        print("\nLoading and preparing data...")
        prophet_df, exogenous_features = data_manager.load_and_prepare_data()
        print(f"Prepared data: {prophet_df.shape}")
        print(f"Exogenous features: {exogenous_features}")
        
        # Step 5: Train model
        print("\nTraining model...")
        training_results = training_pipeline.run_training_pipeline(
            prophet_df, exogenous_features
        )
        print("Model training completed!")
        
        # Step 6: Make predictions
        print("\nMaking predictions...")
        model_wrapper = training_results['model']
        forecast = model_wrapper.predict(prophet_df, horizon_months=12)
        print(f"Generated forecast: {len(forecast)} periods")
        
        # Step 7: Evaluate model
        print("\nEvaluating model...")
        evaluation_results = evaluator.evaluate_model(
            model_wrapper, forecast, prophet_df
        )
        
        # Print key metrics
        metrics = evaluation_results['metrics']
        print("\nModel Performance:")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        
        # Step 8: Generate visualizations
        print("\nGenerating visualizations...")
        plots = {}
        
        # Forecast plot
        plots['forecast'] = visualizer.create_forecast_plot(forecast, prophet_df)
        
        # Components plot
        plots['components'] = visualizer.create_components_plot(forecast)
        
        # Residuals plot
        actual_values = prophet_df['y'].values
        predicted_values = forecast['yhat'].values
        plots['residuals'] = visualizer.create_residuals_plot(actual_values, predicted_values)
        
        print(f"Generated {len(plots)} plots")
        
        # Step 9: Save artifacts
        print("\nSaving artifacts...")
        artifacts = {}
        
        # Save model
        artifacts['model'] = artifact_manager.save_model(model_wrapper)
        
        # Save predictions
        artifacts['predictions'] = artifact_manager.save_predictions(forecast)
        
        # Save metrics
        artifacts['metrics'] = artifact_manager.save_metrics(metrics)
        
        # Save configuration
        artifacts['config'] = artifact_manager.save_config(config)
        
        print(f"Saved {len(artifacts)} artifacts")
        
        # Step 10: Generate report
        print("\nGenerating report...")
        report_path = report_generator.generate_report(
            training_results, evaluation_results, forecast, prophet_df
        )
        
        # Summary
        print("\n" + "="*60)
        print("EXAMPLE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nResults saved to: example_outputs/")
        print(f"Report: {report_path}")
        print(f"Model: {artifacts['model']}")
        print(f"Predictions: {artifacts['predictions']}")
        print(f"Metrics: {artifacts['metrics']}")
        
        print(f"\nPlots generated:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path}")
        
        print(f"\nKey Metrics:")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        
        print(f"\nForecast Summary:")
        print(f"  Forecast periods: {len(forecast) - len(prophet_df)}")
        print(f"  Forecast range: {forecast['ds'].iloc[-12]} to {forecast['ds'].iloc[-1]}")
        print(f"  Mean forecast: {forecast['yhat'].iloc[-12:].mean():.2f}")
        
    except Exception as e:
        print(f"\nError running example pipeline: {str(e)}")
        raise
    
    finally:
        # Cleanup sample data files
        for file in ['sample_target_data.csv', 'sample_features_data.csv']:
            if Path(file).exists():
                Path(file).unlink()
                print(f"Cleaned up: {file}")


if __name__ == "__main__":
    run_example_pipeline()
