#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for Prophet forecasting pipeline.
Handles model training, hyperparameter optimization, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from models import ProphetModelWrapper, ProphetHyperparameterOptimizer


class TimeSeriesSplitter:
    """Custom time series cross-validation splitter."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cv_config = config['training']['cv']
        
    def split_data(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split data into train/validation sets for time series CV."""
        logger.info("Splitting data for time series cross-validation...")
        
        n_splits = self.cv_config['n_splits']
        test_size = int(len(df) * self.cv_config['test_size'])
        gap = self.cv_config['gap']
        
        splits = []
        
        for i in range(n_splits):
            # Calculate split indices
            test_start = len(df) - test_size - (i * test_size // n_splits)
            test_end = test_start + test_size
            
            train_end = test_start - gap
            
            if train_end > 0 and test_end <= len(df):
                train_df = df.iloc[:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()
                
                splits.append((train_df, test_df))
                
                logger.info(f"Split {i+1}: Train {len(train_df)} samples, Test {len(test_df)} samples")
        
        logger.info(f"Created {len(splits)} train/test splits")
        
        return splits


class ModelTrainer:
    """Handles model training and hyperparameter optimization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_config = config['training']
        self.model_config = config['model']
        
    def train_model(self, df: pd.DataFrame, exogenous_features: List[str], 
                   optimize_hyperparams: bool = True) -> Tuple[ProphetModelWrapper, Dict[str, Any]]:
        """Train Prophet model with optional hyperparameter optimization."""
        logger.info("Starting model training...")
        
        # Prepare training data
        train_df = self._prepare_training_data(df)
        
        # Optimize hyperparameters if enabled
        best_params = {}
        if optimize_hyperparams and self.training_config['hyperparameter_optimization']['enabled']:
            logger.info("Optimizing hyperparameters...")
            
            optimizer = ProphetHyperparameterOptimizer(self.config)
            opt_config = self.training_config['hyperparameter_optimization']
            
            best_params = optimizer.optimize(
                train_df,
                exogenous_features,
                n_trials=opt_config['n_trials'],
                timeout=opt_config['timeout']
            )
            
            # Update model configuration with best parameters
            self._update_model_config(best_params)
        
        # Train final model
        logger.info("Training final model...")
        
        model_wrapper = ProphetModelWrapper(self.config)
        model_wrapper.fit(train_df, exogenous_features)
        
        logger.info("Model training completed successfully")
        
        return model_wrapper, best_params
    
    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training based on configuration."""
        train_start = self.training_config['train_start']
        train_end = self.training_config['train_end']
        
        # Filter data by training period
        train_df = df[
            (df['ds'] >= train_start) & 
            (df['ds'] <= train_end)
        ].copy()
        
        logger.info(f"Training data: {len(train_df)} samples from {train_start} to {train_end}")
        
        return train_df
    
    def _update_model_config(self, best_params: Dict[str, Any]) -> None:
        """Update model configuration with optimized parameters."""
        prophet_params = self.model_config['prophet_params']
        
        for param, value in best_params.items():
            prophet_params[param] = value
            logger.info(f"Updated {param} to {value}")


class CrossValidator:
    """Handles cross-validation for time series models."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def cross_validate(self, df: pd.DataFrame, exogenous_features: List[str], 
                      model_wrapper: ProphetModelWrapper) -> Dict[str, Any]:
        """Perform cross-validation on the trained model."""
        logger.info("Starting cross-validation...")
        
        # Create time series splitter
        splitter = TimeSeriesSplitter(self.config)
        splits = splitter.split_data(df)
        
        cv_results = {
            'splits': [],
            'metrics': {},
            'predictions': []
        }
        
        # Perform cross-validation
        for i, (train_df, test_df) in enumerate(splits):
            logger.info(f"Cross-validation fold {i+1}/{len(splits)}...")
            
            # Train model on training set
            fold_model = ProphetModelWrapper(self.config)
            fold_model.fit(train_df, exogenous_features)
            
            # Make predictions on test set
            forecast = fold_model.predict(test_df, horizon_months=len(test_df))
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(test_df['y'], forecast['yhat'])
            
            # Store results
            cv_results['splits'].append({
                'fold': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'metrics': fold_metrics
            })
            
            cv_results['predictions'].append({
                'fold': i + 1,
                'actual': test_df['y'].values,
                'predicted': forecast['yhat'].values,
                'dates': test_df['ds'].values
            })
        
        # Calculate average metrics
        cv_results['metrics'] = self._calculate_average_metrics(cv_results['splits'])
        
        logger.info("Cross-validation completed successfully")
        
        return cv_results
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {metric: np.nan for metric in ['mae', 'mse', 'rmse', 'mape', 'smape', 'r2']}
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(actual_clean - predicted_clean))
        
        # Mean Squared Error
        metrics['mse'] = np.mean((actual_clean - predicted_clean) ** 2)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # Symmetric Mean Absolute Percentage Error
        metrics['smape'] = np.mean(
            2 * np.abs(actual_clean - predicted_clean) / 
            (np.abs(actual_clean) + np.abs(predicted_clean))
        ) * 100
        
        # R-squared
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return metrics
    
    def _calculate_average_metrics(self, splits: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics across all folds."""
        metrics_list = [split['metrics'] for split in splits]
        
        average_metrics = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list if not np.isnan(m[metric])]
            if values:
                average_metrics[metric] = np.mean(values)
                average_metrics[f'{metric}_std'] = np.std(values)
            else:
                average_metrics[metric] = np.nan
                average_metrics[f'{metric}_std'] = np.nan
        
        return average_metrics


class TrainingPipeline:
    """Main training pipeline that orchestrates all training operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trainer = ModelTrainer(config)
        self.cross_validator = CrossValidator(config)
        
    def run_training_pipeline(self, df: pd.DataFrame, exogenous_features: List[str]) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting training pipeline...")
        
        # Train model
        model_wrapper, best_params = self.trainer.train_model(
            df, 
            exogenous_features, 
            optimize_hyperparams=True
        )
        
        # Perform cross-validation
        cv_results = self.cross_validator.cross_validate(
            df, 
            exogenous_features, 
            model_wrapper
        )
        
        # Get feature importance
        feature_importance = model_wrapper.get_feature_importance()
        
        # Compile results
        training_results = {
            'model': model_wrapper,
            'best_params': best_params,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'training_config': self.config['training']
        }
        
        logger.info("Training pipeline completed successfully")
        
        return training_results
