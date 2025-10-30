#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline script for Prophet forecasting.
Orchestrates the complete forecasting pipeline from data loading to report generation.
"""

import sys
import os
from pathlib import Path
import click
import yaml
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data import DataManager
from training import TrainingPipeline
from evaluation import ModelEvaluator
from utils import ReportGenerator, Visualizer, ArtifactManager, ConfigManager


class ProphetForecastingPipeline:
    """Main pipeline class that orchestrates the complete forecasting workflow."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = ConfigManager.load_config(config_path)
        
        # Validate configuration
        if not ConfigManager.validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.training_pipeline = TrainingPipeline(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.visualizer = Visualizer(self.config)
        self.artifact_manager = ArtifactManager(self.config)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        
        # Remove default logger
        logger.remove()
        
        # Add console logging
        logger.add(
            sys.stdout,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")
        )
        
        # Add file logging
        log_file = log_config.get('file', 'logs/prophet_pipeline.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
            rotation="10 MB",
            retention="30 days"
        )
        
        logger.info("Logging setup completed")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete forecasting pipeline."""
        logger.info("Starting Prophet forecasting pipeline...")
        
        try:
            # Step 1: Load and prepare data
            logger.info("Step 1: Loading and preparing data...")
            prophet_df, exogenous_features = self.data_manager.load_and_prepare_data()
            
            # Step 2: Train model
            logger.info("Step 2: Training model...")
            training_results = self.training_pipeline.run_training_pipeline(
                prophet_df, exogenous_features
            )
            
            # Step 3: Make predictions
            logger.info("Step 3: Making predictions...")
            model_wrapper = training_results['model']
            horizon_months = self.config['forecasting']['horizon_months']
            forecast = model_wrapper.predict(prophet_df, horizon_months)
            
            # Step 4: Evaluate model
            logger.info("Step 4: Evaluating model...")
            evaluation_results = self.evaluator.evaluate_model(
                model_wrapper, forecast, prophet_df
            )
            
            # Step 5: Generate visualizations
            logger.info("Step 5: Generating visualizations...")
            plots = self._generate_plots(forecast, prophet_df, evaluation_results)
            
            # Step 6: Save artifacts
            logger.info("Step 6: Saving artifacts...")
            artifacts = self._save_artifacts(
                model_wrapper, forecast, evaluation_results, training_results
            )
            
            # Step 7: Generate report
            logger.info("Step 7: Generating report...")
            report_path = self.report_generator.generate_report(
                training_results, evaluation_results, forecast, prophet_df
            )
            
            # Compile results
            pipeline_results = {
                'data': {
                    'prophet_df': prophet_df,
                    'exogenous_features': exogenous_features
                },
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'forecast': forecast,
                'plots': plots,
                'artifacts': artifacts,
                'report_path': report_path,
                'config': self.config
            }
            
            logger.info("Prophet forecasting pipeline completed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _generate_plots(self, forecast: pd.DataFrame, actual_data: pd.DataFrame, 
                       evaluation_results: Dict) -> Dict[str, str]:
        """Generate all visualization plots."""
        plots = {}
        
        # Forecast plot
        if self.config['output']['plots']['forecast_plot']:
            plots['forecast'] = self.visualizer.create_forecast_plot(forecast, actual_data)
        
        # Components plot
        if self.config['output']['plots']['components_plot']:
            plots['components'] = self.visualizer.create_components_plot(forecast)
        
        # Residuals plot
        if self.config['output']['plots']['residuals_plot']:
            actual_values = actual_data['y'].values
            predicted_values = forecast['yhat'].values
            plots['residuals'] = self.visualizer.create_residuals_plot(actual_values, predicted_values)
        
        return plots
    
    def _save_artifacts(self, model_wrapper, forecast: pd.DataFrame, 
                       evaluation_results: Dict, training_results: Dict) -> Dict[str, str]:
        """Save all pipeline artifacts."""
        artifacts = {}
        
        # Save model
        if self.config['output']['save_model']:
            artifacts['model'] = self.artifact_manager.save_model(model_wrapper)
        
        # Save predictions
        if self.config['output']['save_predictions']:
            artifacts['predictions'] = self.artifact_manager.save_predictions(forecast)
        
        # Save metrics
        if self.config['output']['save_metrics']:
            artifacts['metrics'] = self.artifact_manager.save_metrics(evaluation_results['metrics'])
        
        # Save configuration
        artifacts['config'] = self.artifact_manager.save_config(self.config)
        
        return artifacts


@click.command()
@click.option('--config', '-c', default='configs/default_config.yaml', 
              help='Path to configuration file')
@click.option('--output-dir', '-o', default='outputs', 
              help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
def main(config: str, output_dir: str, verbose: bool):
    """Run the Prophet forecasting pipeline."""
    
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(42)
    
    try:
        # Initialize pipeline
        pipeline = ProphetForecastingPipeline(config)
        
        # Update output directory in config if specified
        if output_dir != 'outputs':
            pipeline.config['output']['base_dir'] = output_dir
        
        # Set verbose logging if requested
        if verbose:
            pipeline.config['logging']['level'] = 'DEBUG'
            pipeline._setup_logging()
        
        # Run pipeline
        results = pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        metrics = results['evaluation_results']['metrics']
        print(f"Model Performance:")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        
        print(f"\nArtifacts Saved:")
        for artifact_type, path in results['artifacts'].items():
            print(f"  {artifact_type}: {path}")
        
        print(f"\nPlots Generated:")
        for plot_type, path in results['plots'].items():
            print(f"  {plot_type}: {path}")
        
        print(f"\nReport Generated: {results['report_path']}")
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
