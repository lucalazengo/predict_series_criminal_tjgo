#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models module for Prophet forecasting pipeline.
Contains Prophet model wrapper with exogenous variables support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from prophet import Prophet
import holidays
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ProphetModelWrapper:
    """Wrapper for Facebook Prophet with enhanced functionality."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']
        self.model = None
        self.exogenous_features = []
        self.holidays_df = None
        
    def _create_holidays(self) -> Optional[pd.DataFrame]:
        """Create holidays dataframe for Prophet."""
        holidays_config = self.model_config.get('holidays', {})
        
        if not holidays_config.get('enabled', False):
            return None
            
        logger.info("Creating holidays dataframe...")
        
        country = holidays_config.get('country', 'BR')
        custom_holidays = holidays_config.get('custom_holidays', [])
        
        # Get country holidays
        country_holidays = holidays.country_holidays(country)
        
        # Create holidays dataframe
        holidays_list = []
        
        # Add country holidays
        for date, name in country_holidays.items():
            holidays_list.append({
                'holiday': name,
                'ds': pd.to_datetime(date),
                'lower_window': 0,
                'upper_window': 0
            })
        
        # Add custom holidays
        for holiday in custom_holidays:
            holidays_list.append({
                'holiday': holiday['name'],
                'ds': pd.to_datetime(holiday['date']),
                'lower_window': holiday.get('lower_window', 0),
                'upper_window': holiday.get('upper_window', 0)
            })
        
        if holidays_list:
            holidays_df = pd.DataFrame(holidays_list)
            logger.info(f"Created {len(holidays_df)} holidays")
            return holidays_df
        
        return None
    
    def _setup_model(self, exogenous_features: List[str]) -> Prophet:
        """Setup Prophet model with configuration."""
        logger.info("Setting up Prophet model...")
        
        prophet_params = self.model_config['prophet_params']
        
        # Initialize Prophet with parameters
        model = Prophet(
            growth=prophet_params.get('growth', 'linear'),
            changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0),
            seasonality_mode=prophet_params.get('seasonality_mode', 'additive'),
            daily_seasonality=prophet_params.get('daily_seasonality', False),
            weekly_seasonality=prophet_params.get('weekly_seasonality', True),
            yearly_seasonality=prophet_params.get('yearly_seasonality', True),
            holidays=self.holidays_df
        )
        
        # Add exogenous regressors
        if self.model_config['exogenous_vars']['enabled']:
            for feature in exogenous_features:
                model.add_regressor(feature)
                logger.info(f"Added exogenous regressor: {feature}")
        
        self.exogenous_features = exogenous_features
        logger.info(f"Prophet model setup completed with {len(exogenous_features)} exogenous features")
        
        return model
    
    def fit(self, df: pd.DataFrame, exogenous_features: List[str]) -> 'ProphetModelWrapper':
        """Fit the Prophet model."""
        logger.info("Fitting Prophet model...")
        
        # Create holidays if enabled
        self.holidays_df = self._create_holidays()
        
        # Setup model
        self.model = self._setup_model(exogenous_features)
        
        # Fit model
        self.model.fit(df)
        
        logger.info("Prophet model fitted successfully")
        
        return self
    
    def predict(self, df: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
        """Make predictions with Prophet."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making predictions for {horizon_months} months ahead...")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon_months, freq='MS')
        
        # Add exogenous features to future dataframe
        if self.exogenous_features:
            # For future periods, we need to provide exogenous values
            # This is a limitation - in practice, you'd need to forecast these too
            # For now, we'll use the last known values
            last_values = df[self.exogenous_features].iloc[-1]
            for feature in self.exogenous_features:
                future[feature] = last_values[feature]
        
        # Make prediction
        forecast = self.model.predict(future)
        
        logger.info("Predictions completed successfully")
        
        return forecast
    
    def cross_validation(self, df: pd.DataFrame, initial: str = '365 days', 
                        period: str = '180 days', horizon: str = '30 days') -> pd.DataFrame:
        """Perform cross-validation."""
        if self.model is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        logger.info("Performing cross-validation...")
        
        from prophet.diagnostics import cross_validation
        
        cv_results = cross_validation(
            self.model, 
            initial=initial, 
            period=period, 
            horizon=horizon
        )
        
        logger.info(f"Cross-validation completed: {len(cv_results)} folds")
        
        return cv_results
    
    def get_model_components(self, forecast: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract model components from forecast."""
        logger.info("Extracting model components...")
        
        components = {}
        
        # Trend component
        components['trend'] = forecast[['ds', 'trend']].copy()
        
        # Seasonal components
        seasonal_cols = [col for col in forecast.columns if col.startswith('yearly') or col.startswith('weekly')]
        if seasonal_cols:
            components['seasonal'] = forecast[['ds'] + seasonal_cols].copy()
        
        # Holiday component
        if 'holidays' in forecast.columns:
            components['holidays'] = forecast[['ds', 'holidays']].copy()
        
        # Exogenous components
        if self.exogenous_features:
            exogenous_cols = [col for col in forecast.columns if col in self.exogenous_features]
            if exogenous_cols:
                components['exogenous'] = forecast[['ds'] + exogenous_cols].copy()
        
        logger.info(f"Extracted {len(components)} model components")
        
        return components
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Prophet model."""
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        logger.info("Calculating feature importance...")
        
        importance = {}
        
        # Get regressor coefficients
        if hasattr(self.model, 'params') and 'beta' in self.model.params:
            beta = self.model.params['beta']
            
            # Map coefficients to feature names
            if self.exogenous_features:
                for i, feature in enumerate(self.exogenous_features):
                    if i < len(beta):
                        importance[feature] = float(np.mean(beta[i])) if hasattr(beta[i], '__len__') else float(beta[i])
        
        logger.info(f"Calculated importance for {len(importance)} features")
        
        return importance
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model."""
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        logger.info(f"Saving model to {filepath}...")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'exogenous_features': self.exogenous_features,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ProphetModelWrapper':
        """Load a saved model."""
        logger.info(f"Loading model from {filepath}...")
        
        import joblib
        
        model_data = joblib.load(filepath)
        
        # Create wrapper instance
        wrapper = cls(model_data['config'])
        wrapper.model = model_data['model']
        wrapper.exogenous_features = model_data['exogenous_features']
        
        logger.info("Model loaded successfully")
        
        return wrapper


class ProphetHyperparameterOptimizer:
    """Hyperparameter optimization for Prophet using Optuna."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']
        
    def optimize(self, df: pd.DataFrame, exogenous_features: List[str], 
                n_trials: int = 50, timeout: int = 3600) -> Dict[str, Any]:
        """Optimize Prophet hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization ({n_trials} trials)...")
        
        import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            changepoint_prior_scale = trial.suggest_float(
                'changepoint_prior_scale', 0.001, 0.5, log=True
            )
            seasonality_prior_scale = trial.suggest_float(
                'seasonality_prior_scale', 0.01, 10.0, log=True
            )
            holidays_prior_scale = trial.suggest_float(
                'holidays_prior_scale', 0.01, 10.0, log=True
            )
            
            # Create model with suggested parameters
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                holidays=self._create_holidays()
            )
            
            # Add exogenous regressors
            for feature in exogenous_features:
                model.add_regressor(feature)
            
            # Fit model
            model.fit(df)
            
            # Cross-validation for evaluation
            from prophet.diagnostics import cross_validation, performance_metrics
            
            cv_results = cross_validation(
                model, 
                initial='365 days', 
                period='180 days', 
                horizon='30 days'
            )
            
            metrics = performance_metrics(cv_results)
            
            # Return RMSE as optimization metric
            return metrics['rmse'].mean()
        
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        return best_params
    
    def _create_holidays(self) -> Optional[pd.DataFrame]:
        """Create holidays dataframe (same as ProphetModelWrapper)."""
        holidays_config = self.model_config.get('holidays', {})
        
        if not holidays_config.get('enabled', False):
            return None
            
        country = holidays_config.get('country', 'BR')
        custom_holidays = holidays_config.get('custom_holidays', [])
        
        country_holidays = holidays.country_holidays(country)
        
        holidays_list = []
        
        for date, name in country_holidays.items():
            holidays_list.append({
                'holiday': name,
                'ds': pd.to_datetime(date),
                'lower_window': 0,
                'upper_window': 0
            })
        
        for holiday in custom_holidays:
            holidays_list.append({
                'holiday': holiday['name'],
                'ds': pd.to_datetime(holiday['date']),
                'lower_window': holiday.get('lower_window', 0),
                'upper_window': holiday.get('upper_window', 0)
            })
        
        if holidays_list:
            return pd.DataFrame(holidays_list)
        
        return None
