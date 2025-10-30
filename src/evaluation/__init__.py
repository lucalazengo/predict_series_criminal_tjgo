#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module for Prophet forecasting pipeline.
Handles model evaluation, metrics calculation, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculates various evaluation metrics for time series forecasting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.evaluation_config = config['evaluation']
        
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray, 
                        metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate specified evaluation metrics."""
        if metric_names is None:
            metric_names = self.evaluation_config['metrics']
        
        logger.info(f"Calculating metrics: {metric_names}")
        
        # Ensure arrays have the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            logger.warning("No valid data points for metric calculation")
            return {metric: np.nan for metric in metric_names}
        
        metrics = {}
        
        for metric in metric_names:
            if metric == 'mae':
                metrics['mae'] = self._calculate_mae(actual_clean, predicted_clean)
            elif metric == 'mse':
                metrics['mse'] = self._calculate_mse(actual_clean, predicted_clean)
            elif metric == 'rmse':
                metrics['rmse'] = self._calculate_rmse(actual_clean, predicted_clean)
            elif metric == 'mape':
                metrics['mape'] = self._calculate_mape(actual_clean, predicted_clean)
            elif metric == 'smape':
                metrics['smape'] = self._calculate_smape(actual_clean, predicted_clean)
            elif metric == 'r2':
                metrics['r2'] = self._calculate_r2(actual_clean, predicted_clean)
            else:
                logger.warning(f"Unknown metric: {metric}")
        
        logger.info(f"Calculated {len(metrics)} metrics")
        
        return metrics
    
    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(actual - predicted))
    
    def _calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((actual - predicted) ** 2)
    
    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self._calculate_mse(actual, predicted))
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = actual != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = np.abs(actual) + np.abs(predicted)
        mask = denominator != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class ComponentAnalyzer:
    """Analyzes Prophet model components and their contributions."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_components(self, forecast: pd.DataFrame, 
                          components: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze model components and their contributions."""
        logger.info("Analyzing model components...")
        
        analysis = {
            'component_contributions': {},
            'component_statistics': {},
            'trend_analysis': {},
            'seasonality_analysis': {}
        }
        
        # Analyze trend component
        if 'trend' in components:
            analysis['trend_analysis'] = self._analyze_trend(components['trend'])
        
        # Analyze seasonal components
        if 'seasonal' in components:
            analysis['seasonality_analysis'] = self._analyze_seasonality(components['seasonal'])
        
        # Calculate component contributions
        analysis['component_contributions'] = self._calculate_contributions(forecast, components)
        
        # Calculate component statistics
        analysis['component_statistics'] = self._calculate_component_stats(components)
        
        logger.info("Component analysis completed")
        
        return analysis
    
    def _analyze_trend(self, trend_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend component."""
        trend_values = trend_df['trend'].values
        
        analysis = {
            'trend_direction': 'increasing' if trend_values[-1] > trend_values[0] else 'decreasing',
            'trend_change_rate': (trend_values[-1] - trend_values[0]) / len(trend_values),
            'trend_volatility': np.std(np.diff(trend_values)),
            'trend_range': [np.min(trend_values), np.max(trend_values)]
        }
        
        return analysis
    
    def _analyze_seasonality(self, seasonal_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal components."""
        analysis = {}
        
        seasonal_cols = [col for col in seasonal_df.columns if col != 'ds']
        
        for col in seasonal_cols:
            values = seasonal_df[col].values
            analysis[col] = {
                'amplitude': np.max(values) - np.min(values),
                'mean_amplitude': np.mean(np.abs(values)),
                'peak_value': np.max(values),
                'trough_value': np.min(values)
            }
        
        return analysis
    
    def _calculate_contributions(self, forecast: pd.DataFrame, 
                                components: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate relative contributions of each component."""
        contributions = {}
        
        total_variance = np.var(forecast['yhat'])
        
        for component_name, component_df in components.items():
            if component_name == 'trend':
                component_variance = np.var(component_df['trend'])
            elif component_name == 'seasonal':
                # Sum all seasonal components
                seasonal_cols = [col for col in component_df.columns if col != 'ds']
                component_variance = np.var(component_df[seasonal_cols].sum(axis=1))
            elif component_name == 'holidays':
                component_variance = np.var(component_df['holidays'])
            elif component_name == 'exogenous':
                exogenous_cols = [col for col in component_df.columns if col != 'ds']
                component_variance = np.var(component_df[exogenous_cols].sum(axis=1))
            else:
                continue
            
            contributions[component_name] = component_variance / total_variance if total_variance > 0 else 0
        
        return contributions
    
    def _calculate_component_stats(self, components: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for each component."""
        stats = {}
        
        for component_name, component_df in components.items():
            component_stats = {}
            
            if component_name == 'trend':
                values = component_df['trend'].values
            elif component_name == 'seasonal':
                seasonal_cols = [col for col in component_df.columns if col != 'ds']
                values = component_df[seasonal_cols].sum(axis=1).values
            elif component_name == 'holidays':
                values = component_df['holidays'].values
            elif component_name == 'exogenous':
                exogenous_cols = [col for col in component_df.columns if col != 'ds']
                values = component_df[exogenous_cols].sum(axis=1).values
            else:
                continue
            
            component_stats = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
            
            stats[component_name] = component_stats
        
        return stats


class PerformanceAnalyzer:
    """Analyzes model performance across different time periods and conditions."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_performance(self, actual: pd.Series, predicted: pd.Series, 
                           dates: pd.Series) -> Dict[str, Any]:
        """Analyze model performance across different dimensions."""
        logger.info("Analyzing model performance...")
        
        analysis = {
            'overall_performance': {},
            'temporal_performance': {},
            'error_analysis': {},
            'residual_analysis': {}
        }
        
        # Overall performance
        metrics_calc = MetricsCalculator(self.config)
        analysis['overall_performance'] = metrics_calc.calculate_metrics(
            actual.values, predicted.values
        )
        
        # Temporal performance analysis
        analysis['temporal_performance'] = self._analyze_temporal_performance(
            actual, predicted, dates
        )
        
        # Error analysis
        analysis['error_analysis'] = self._analyze_errors(actual, predicted)
        
        # Residual analysis
        analysis['residual_analysis'] = self._analyze_residuals(actual, predicted)
        
        logger.info("Performance analysis completed")
        
        return analysis
    
    def _analyze_temporal_performance(self, actual: pd.Series, predicted: pd.Series, 
                                    dates: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different time periods."""
        df = pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'dates': dates
        })
        
        df['year'] = df['dates'].dt.year
        df['month'] = df['dates'].dt.month
        df['quarter'] = df['dates'].dt.quarter
        
        temporal_analysis = {}
        
        # Performance by year
        yearly_metrics = []
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            if len(year_data) > 0:
                metrics_calc = MetricsCalculator(self.config)
                year_metrics = metrics_calc.calculate_metrics(
                    year_data['actual'].values, 
                    year_data['predicted'].values
                )
                year_metrics['year'] = year
                yearly_metrics.append(year_metrics)
        
        temporal_analysis['yearly'] = yearly_metrics
        
        # Performance by month
        monthly_metrics = []
        for month in df['month'].unique():
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                metrics_calc = MetricsCalculator(self.config)
                month_metrics = metrics_calc.calculate_metrics(
                    month_data['actual'].values, 
                    month_data['predicted'].values
                )
                month_metrics['month'] = month
                monthly_metrics.append(month_metrics)
        
        temporal_analysis['monthly'] = monthly_metrics
        
        return temporal_analysis
    
    def _analyze_errors(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = actual - predicted
        
        error_analysis = {
            'error_distribution': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'skewness': self._calculate_skewness(errors),
                'kurtosis': self._calculate_kurtosis(errors)
            },
            'error_percentiles': {
                'p5': np.percentile(errors, 5),
                'p25': np.percentile(errors, 25),
                'p50': np.percentile(errors, 50),
                'p75': np.percentile(errors, 75),
                'p95': np.percentile(errors, 95)
            },
            'large_errors': {
                'count': np.sum(np.abs(errors) > 2 * np.std(errors)),
                'percentage': np.sum(np.abs(errors) > 2 * np.std(errors)) / len(errors) * 100
            }
        }
        
        return error_analysis
    
    def _analyze_residuals(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, Any]:
        """Analyze residuals for model diagnostics."""
        residuals = actual - predicted
        
        residual_analysis = {
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals)
            },
            'normality_test': self._test_normality(residuals),
            'autocorrelation': self._test_autocorrelation(residuals),
            'heteroscedasticity': self._test_heteroscedasticity(residuals, predicted)
        }
        
        return residual_analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test normality of residuals."""
        from scipy import stats
        
        try:
            statistic, p_value = stats.shapiro(residuals)
            return {
                'shapiro_statistic': statistic,
                'shapiro_p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {'shapiro_statistic': np.nan, 'shapiro_p_value': np.nan, 'is_normal': False}
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test autocorrelation in residuals."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            result = acorr_ljungbox(residuals, lags=10, return_df=True)
            return {
                'ljung_box_statistic': result['lb_stat'].iloc[-1],
                'ljung_box_p_value': result['lb_pvalue'].iloc[-1],
                'has_autocorrelation': result['lb_pvalue'].iloc[-1] < 0.05
            }
        except:
            return {'ljung_box_statistic': np.nan, 'ljung_box_p_value': np.nan, 'has_autocorrelation': False}
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Test heteroscedasticity in residuals."""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            # Create a simple regression of residuals on predicted values
            X = np.column_stack([np.ones(len(predicted)), predicted])
            lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)
            
            return {
                'breusch_pagan_statistic': lm,
                'breusch_pagan_p_value': lm_pvalue,
                'has_heteroscedasticity': lm_pvalue < 0.05
            }
        except:
            return {'breusch_pagan_statistic': np.nan, 'breusch_pagan_p_value': np.nan, 'has_heteroscedasticity': False}


class FeatureImportanceAnalyzer:
    """Analyzes importance of exogenous features in Prophet model."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_importance(self, model_wrapper, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive feature importance analysis."""
        logger.info("Starting detailed feature importance analysis...")
        
        analysis = {
            'coefficients': {},
            'relative_importance': {},
            'contribution_analysis': {},
            'correlation_analysis': {}
        }
        
        # Get basic importance from model
        basic_importance = model_wrapper.get_feature_importance()
        analysis['coefficients'] = basic_importance
        
        # Calculate relative importance (normalized)
        if basic_importance:
            total_abs = sum(abs(v) for v in basic_importance.values())
            if total_abs > 0:
                for feature, value in basic_importance.items():
                    analysis['relative_importance'][feature] = abs(value) / total_abs * 100
        
        # Analyze contribution to predictions
        if hasattr(model_wrapper, 'exogenous_features') and model_wrapper.exogenous_features:
            analysis['contribution_analysis'] = self._analyze_contribution(
                model_wrapper, df, model_wrapper.exogenous_features
            )
        
        # Correlation with target
        if hasattr(model_wrapper, 'exogenous_features') and model_wrapper.exogenous_features:
            correlations = {}
            target_col = self.config['data']['target_column']
            if target_col in df.columns:
                for feature in model_wrapper.exogenous_features:
                    if feature in df.columns:
                        corr = df[feature].corr(df[target_col])
                        correlations[feature] = corr
            
            analysis['correlation_analysis'] = correlations
        
        logger.info("Feature importance analysis completed")
        return analysis
    
    def _analyze_contribution(self, model_wrapper, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Analyze contribution of each feature to predictions."""
        contributions = {}
        
        if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
            try:
                # Get predictions
                forecast = model_wrapper.predict(df, horizon_months=0)
                
                # Estimate contribution by analyzing coefficients and feature values
                for feature in features:
                    if feature in df.columns:
                        # Contribution is roughly coefficient * feature_value
                        coeff = model_wrapper.get_feature_importance().get(feature, 0)
                        avg_contribution = abs(coeff * df[feature].mean())
                        contributions[feature] = {
                            'average_contribution': avg_contribution,
                            'coefficient': coeff,
                            'feature_mean': df[feature].mean(),
                            'feature_std': df[feature].std()
                        }
            except Exception as e:
                logger.warning(f"Could not calculate detailed contributions: {e}")
        
        return contributions


class ModelEvaluator:
    """Main evaluation class that orchestrates all evaluation operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.metrics_calculator = MetricsCalculator(config)
        self.component_analyzer = ComponentAnalyzer(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.feature_analyzer = FeatureImportanceAnalyzer(config)
        
    def evaluate_model(self, model_wrapper, forecast: pd.DataFrame, 
                       actual_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Starting comprehensive model evaluation...")
        
        evaluation_results = {
            'metrics': {},
            'component_analysis': {},
            'performance_analysis': {},
            'model_diagnostics': {},
            'feature_importance_analysis': {}
        }
        
        # Calculate metrics
        actual_values = actual_data['y'].values
        predicted_values = forecast['yhat'].values
        
        evaluation_results['metrics'] = self.metrics_calculator.calculate_metrics(
            actual_values, predicted_values
        )
        
        # Component analysis
        if self.evaluation_config['component_analysis']['enabled']:
            components = model_wrapper.get_model_components(forecast)
            evaluation_results['component_analysis'] = self.component_analyzer.analyze_components(
                forecast, components
            )
        
        # Performance analysis
        evaluation_results['performance_analysis'] = self.performance_analyzer.analyze_performance(
            pd.Series(actual_values),
            pd.Series(predicted_values),
            pd.Series(actual_data['ds'])
        )
        
        # Detailed feature importance analysis
        try:
            # Merge forecast with actual data for analysis
            analysis_df = actual_data.copy()
            for col in forecast.columns:
                if col not in analysis_df.columns:
                    analysis_df[col] = forecast[col].values[:len(analysis_df)]
            
            evaluation_results['feature_importance_analysis'] = self.feature_analyzer.analyze_importance(
                model_wrapper, analysis_df
            )
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            evaluation_results['feature_importance_analysis'] = {}
        
        # Model diagnostics
        evaluation_results['model_diagnostics'] = self._run_model_diagnostics(
            model_wrapper, forecast, actual_data
        )
        
        logger.info("Model evaluation completed successfully")
        
        return evaluation_results
    
    def _run_model_diagnostics(self, model_wrapper, forecast: pd.DataFrame, 
                              actual_data: pd.DataFrame) -> Dict[str, Any]:
        """Run model diagnostics."""
        diagnostics = {
            'feature_importance': model_wrapper.get_feature_importance(),
            'model_parameters': self._extract_model_parameters(model_wrapper),
            'forecast_uncertainty': self._analyze_forecast_uncertainty(forecast)
        }
        
        return diagnostics
    
    def _extract_model_parameters(self, model_wrapper) -> Dict[str, Any]:
        """Extract model parameters for analysis."""
        if model_wrapper.model is None:
            return {}
        
        params = {}
        
        # Extract Prophet parameters
        if hasattr(model_wrapper.model, 'params'):
            params['prophet_params'] = model_wrapper.model.params
        
        # Extract configuration
        params['config'] = model_wrapper.config
        
        return params
    
    def _analyze_forecast_uncertainty(self, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Analyze forecast uncertainty."""
        uncertainty_analysis = {}
        
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            uncertainty_analysis['prediction_intervals'] = {
                'mean_width': np.mean(forecast['yhat_upper'] - forecast['yhat_lower']),
                'width_std': np.std(forecast['yhat_upper'] - forecast['yhat_lower']),
                'coverage_80': None,  # Would need actual values to calculate
                'coverage_95': None   # Would need actual values to calculate
            }
        
        return uncertainty_analysis
