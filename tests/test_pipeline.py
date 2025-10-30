#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Prophet forecasting pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data import DataLoader, DataPreprocessor, DataValidator, DataManager
from models import ProphetModelWrapper, ProphetHyperparameterOptimizer
from training import TimeSeriesSplitter, ModelTrainer, CrossValidator, TrainingPipeline
from evaluation import MetricsCalculator, ComponentAnalyzer, PerformanceAnalyzer, ModelEvaluator
from utils import ReportGenerator, Visualizer, ArtifactManager, ConfigManager


class TestDataModule:
    """Test cases for data module."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'data': {
                'target_series_path': 'test_data.csv',
                'exogenous_features_path': 'test_features.csv',
                'target_column': 'TOTAL_CASOS',
                'date_column': 'DATA',
                'preprocessing': {
                    'remove_outliers': True,
                    'outlier_method': 'iqr',
                    'outlier_threshold': 3.0,
                    'fill_missing': True,
                    'fill_method': 'forward_fill'
                }
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample time series data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='MS')
        data = pd.DataFrame({
            'DATA': dates,
            'TOTAL_CASOS': np.random.normal(1000, 100, 100),
            'feature1': np.random.normal(50, 10, 100),
            'feature2': np.random.normal(20, 5, 100)
        })
        return data
    
    def test_data_preprocessor_outlier_removal(self, sample_config, sample_data):
        """Test outlier removal functionality."""
        preprocessor = DataPreprocessor(sample_config)
        
        # Add some outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.loc[0, 'TOTAL_CASOS'] = 10000  # Extreme outlier
        
        cleaned_data = preprocessor.remove_outliers(data_with_outliers, 'TOTAL_CASOS')
        
        assert len(cleaned_data) < len(data_with_outliers)
        assert cleaned_data['TOTAL_CASOS'].max() < 10000
    
    def test_data_preprocessor_fill_missing(self, sample_config, sample_data):
        """Test missing value filling functionality."""
        preprocessor = DataPreprocessor(sample_config)
        
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[10:15, 'TOTAL_CASOS'] = np.nan
        
        filled_data = preprocessor.fill_missing_values(data_with_missing)
        
        assert filled_data['TOTAL_CASOS'].isnull().sum() == 0
    
    def test_data_preprocessor_lag_features(self, sample_config, sample_data):
        """Test lag feature creation."""
        preprocessor = DataPreprocessor(sample_config)
        
        features = ['feature1', 'feature2']
        lagged_data = preprocessor.create_lag_features(sample_data, features, max_lags=2)
        
        expected_new_cols = ['feature1_lag_1', 'feature1_lag_2', 'feature2_lag_1', 'feature2_lag_2']
        for col in expected_new_cols:
            assert col in lagged_data.columns
        
        assert len(lagged_data) == len(sample_data) - 2  # 2 rows lost due to lagging
    
    def test_data_validator_target_series(self, sample_config, sample_data):
        """Test target series validation."""
        validator = DataValidator(sample_config)
        
        # Valid data should pass
        assert validator.validate_target_series(sample_data, 'TOTAL_CASOS', 'DATA') == True
        
        # Invalid data should fail
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'TOTAL_CASOS'] = np.nan
        assert validator.validate_target_series(invalid_data, 'TOTAL_CASOS', 'DATA') == False


class TestModelsModule:
    """Test cases for models module."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
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
                    'features': ['feature1', 'feature2']
                },
                'holidays': {
                    'enabled': False,
                    'country': 'BR',
                    'custom_holidays': []
                }
            }
        }
    
    @pytest.fixture
    def sample_prophet_data(self):
        """Sample Prophet-formatted data."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
        data = pd.DataFrame({
            'ds': dates,
            'y': np.random.normal(1000, 100, 50),
            'feature1': np.random.normal(50, 10, 50),
            'feature2': np.random.normal(20, 5, 50)
        })
        return data
    
    def test_prophet_model_wrapper_initialization(self, sample_config):
        """Test Prophet model wrapper initialization."""
        wrapper = ProphetModelWrapper(sample_config)
        
        assert wrapper.config == sample_config
        assert wrapper.model is None
        assert wrapper.exogenous_features == []
    
    def test_prophet_model_wrapper_fit(self, sample_config, sample_prophet_data):
        """Test Prophet model fitting."""
        wrapper = ProphetModelWrapper(sample_config)
        exogenous_features = ['feature1', 'feature2']
        
        wrapper.fit(sample_prophet_data, exogenous_features)
        
        assert wrapper.model is not None
        assert wrapper.exogenous_features == exogenous_features
    
    def test_prophet_model_wrapper_predict(self, sample_config, sample_prophet_data):
        """Test Prophet model prediction."""
        wrapper = ProphetModelWrapper(sample_config)
        exogenous_features = ['feature1', 'feature2']
        
        wrapper.fit(sample_prophet_data, exogenous_features)
        forecast = wrapper.predict(sample_prophet_data, horizon_months=12)
        
        assert 'yhat' in forecast.columns
        assert 'ds' in forecast.columns
        assert len(forecast) == len(sample_prophet_data) + 12


class TestEvaluationModule:
    """Test cases for evaluation module."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'evaluation': {
                'metrics': ['mae', 'mse', 'rmse', 'mape', 'smape', 'r2'],
                'component_analysis': {
                    'enabled': True,
                    'components': ['trend', 'seasonal', 'holidays', 'exogenous']
                }
            }
        }
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data for testing."""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([105, 108, 125, 128, 145])
        return actual, predicted
    
    def test_metrics_calculator(self, sample_config, sample_predictions):
        """Test metrics calculation."""
        calculator = MetricsCalculator(sample_config)
        actual, predicted = sample_predictions
        
        metrics = calculator.calculate_metrics(actual, predicted)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'smape' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_performance_analyzer(self, sample_config, sample_predictions):
        """Test performance analysis."""
        analyzer = PerformanceAnalyzer(sample_config)
        actual, predicted = sample_predictions
        
        dates = pd.date_range(start='2020-01-01', periods=len(actual), freq='MS')
        
        analysis = analyzer.analyze_performance(
            pd.Series(actual),
            pd.Series(predicted),
            pd.Series(dates)
        )
        
        assert 'overall_performance' in analysis
        assert 'temporal_performance' in analysis
        assert 'error_analysis' in analysis
        assert 'residual_analysis' in analysis


class TestUtilsModule:
    """Test cases for utils module."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'output': {
                'base_dir': 'test_outputs',
                'plots': {
                    'forecast_plot': True,
                    'components_plot': True,
                    'residuals_plot': True
                }
            }
        }
    
    def test_config_manager_load_config(self):
        """Test configuration loading."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
data:
  target_column: "test"
model:
  prophet_params:
    growth: "linear"
training:
  cv:
    n_splits: 5
evaluation:
  metrics: ["mae", "rmse"]
output:
  base_dir: "outputs"
"""
            f.write(config_content)
            config_path = f.name
        
        try:
            config = ConfigManager.load_config(config_path)
            assert 'data' in config
            assert 'model' in config
            assert 'training' in config
            assert 'evaluation' in config
            assert 'output' in config
        finally:
            os.unlink(config_path)
    
    def test_config_manager_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            'data': {},
            'model': {},
            'training': {},
            'evaluation': {},
            'output': {}
        }
        
        assert ConfigManager.validate_config(valid_config) == True
        
        invalid_config = {
            'data': {},
            'model': {}
            # Missing required sections
        }
        
        assert ConfigManager.validate_config(invalid_config) == False


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for integration testing."""
        return {
            'data': {
                'target_series_path': 'test_data.csv',
                'exogenous_features_path': 'test_features.csv',
                'target_column': 'TOTAL_CASOS',
                'date_column': 'DATA',
                'preprocessing': {
                    'remove_outliers': False,  # Disable for testing
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
                    'weekly_seasonality': False,  # Disable for testing
                    'yearly_seasonality': False   # Disable for testing
                },
                'exogenous_vars': {
                    'enabled': True,
                    'features': ['feature1', 'feature2']
                },
                'holidays': {
                    'enabled': False
                }
            },
            'training': {
                'cv': {
                    'n_splits': 3,
                    'test_size': 0.2,
                    'gap': 0
                },
                'hyperparameter_optimization': {
                    'enabled': False  # Disable for testing
                },
                'train_start': '2020-01-01',
                'train_end': '2022-12-01'
            },
            'forecasting': {
                'horizon_months': 6
            },
            'evaluation': {
                'metrics': ['mae', 'rmse', 'r2'],
                'component_analysis': {
                    'enabled': True,
                    'components': ['trend']
                }
            },
            'output': {
                'base_dir': 'test_outputs',
                'save_model': True,
                'save_predictions': True,
                'save_metrics': True,
                'save_plots': False,  # Disable for testing
                'plots': {
                    'forecast_plot': False,
                    'components_plot': False,
                    'residuals_plot': False
                }
            },
            'logging': {
                'level': 'WARNING'  # Reduce logging for testing
            }
        }
    
    def test_end_to_end_pipeline(self, sample_config):
        """Test complete end-to-end pipeline execution."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        
        target_data = pd.DataFrame({
            'DATA': dates,
            'TOTAL_CASOS': np.random.normal(1000, 100, 36)
        })
        
        features_data = pd.DataFrame({
            'data': dates,
            'feature1': np.random.normal(50, 10, 36),
            'feature2': np.random.normal(20, 5, 36)
        })
        
        # Save sample data
        target_data.to_csv('test_data.csv', index=False)
        features_data.to_csv('test_features.csv', index=False)
        
        try:
            # Initialize components
            data_manager = DataManager(sample_config)
            training_pipeline = TrainingPipeline(sample_config)
            evaluator = ModelEvaluator(sample_config)
            
            # Run pipeline steps
            prophet_df, exogenous_features = data_manager.load_and_prepare_data()
            training_results = training_pipeline.run_training_pipeline(prophet_df, exogenous_features)
            
            model_wrapper = training_results['model']
            forecast = model_wrapper.predict(prophet_df, horizon_months=6)
            evaluation_results = evaluator.evaluate_model(model_wrapper, forecast, prophet_df)
            
            # Verify results
            assert len(prophet_df) > 0
            assert len(exogenous_features) > 0
            assert model_wrapper.model is not None
            assert len(forecast) == len(prophet_df) + 6
            assert 'metrics' in evaluation_results
            
        finally:
            # Cleanup
            if os.path.exists('test_data.csv'):
                os.unlink('test_data.csv')
            if os.path.exists('test_features.csv'):
                os.unlink('test_features.csv')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
