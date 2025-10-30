#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data module for Prophet forecasting pipeline.
Handles data loading, preprocessing, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and basic validation of time series data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        
    def load_target_series(self) -> pd.DataFrame:
        """Load the target time series data."""
        logger.info("Loading target series data...")
        
        file_path = self.data_config['target_series_path']
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        date_col = self.data_config['date_column']
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)
        
        logger.info(f"Loaded target series: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        return df
    
    def load_exogenous_features(self) -> pd.DataFrame:
        """Load exogenous features data."""
        logger.info("Loading exogenous features data...")
        
        file_path = self.data_config['exogenous_features_path']
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['data'] = pd.to_datetime(df['data'])
        
        # Sort by date
        df = df.sort_values('data').reset_index(drop=True)
        
        logger.info(f"Loaded exogenous features: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Date range: {df['data'].min()} to {df['data'].max()}")
        
        return df
    
    def merge_datasets(self, target_df: pd.DataFrame, exogenous_df: pd.DataFrame) -> pd.DataFrame:
        """Merge target series with exogenous features."""
        logger.info("Merging datasets...")
        
        date_col = self.data_config['date_column']
        
        # Merge on date columns
        merged_df = pd.merge(
            target_df, 
            exogenous_df, 
            left_on=date_col, 
            right_on='data', 
            how='inner'
        )
        
        # Drop duplicate date column
        merged_df = merged_df.drop(columns=['data'])
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        return merged_df


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessing_config = config['data']['preprocessing']
        
    def remove_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Remove outliers from the target series."""
        if not self.preprocessing_config['remove_outliers']:
            return df
            
        logger.info("Removing outliers...")
        
        method = self.preprocessing_config['outlier_method']
        threshold = self.preprocessing_config['outlier_threshold']
        
        original_size = len(df)
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
            df = df[z_scores <= threshold]
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df[[target_col]])
            df = df[outliers == 1]
        
        removed_count = original_size - len(df)
        logger.info(f"Removed {removed_count} outliers ({removed_count/original_size*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the dataset."""
        if not self.preprocessing_config['fill_missing']:
            return df
            
        logger.info("Filling missing values...")
        
        method = self.preprocessing_config['fill_method']
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate()
        elif method == 'mean':
            df = df.fillna(df.mean())
        
        missing_count = df.isnull().sum().sum()
        logger.info(f"Missing values after filling: {missing_count}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, features: List[str], max_lags: int = 3) -> pd.DataFrame:
        """Create lag features for exogenous variables."""
        logger.info(f"Creating lag features (max_lags={max_lags})...")
        
        df_lagged = df.copy()
        
        for lag in range(1, max_lags + 1):
            for feature in features:
                if feature in df.columns:
                    df_lagged[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Remove rows with NaN values from lagging
        df_lagged = df_lagged.dropna()
        
        logger.info(f"Created {len(features) * max_lags} lag features")
        logger.info(f"Data shape after lag creation: {df_lagged.shape}")
        
        return df_lagged
    
    def prepare_prophet_data(self, df: pd.DataFrame, target_col: str, date_col: str, 
                           exogenous_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data in Prophet format."""
        logger.info("Preparing data for Prophet...")
        
        # Select relevant columns
        prophet_df = df[[date_col, target_col]].copy()
        
        # Rename columns for Prophet
        prophet_df.columns = ['ds', 'y']
        
        # Add exogenous features
        available_features = []
        for feature in exogenous_features:
            if feature in df.columns:
                prophet_df[feature] = df[feature]
                available_features.append(feature)
            else:
                logger.warning(f"Feature '{feature}' not found in dataset")
        
        logger.info(f"Prophet data shape: {prophet_df.shape}")
        logger.info(f"Available exogenous features: {len(available_features)}")
        
        return prophet_df, available_features


class DataValidator:
    """Validates data quality and schema."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def validate_target_series(self, df: pd.DataFrame, target_col: str, date_col: str) -> bool:
        """Validate target series data."""
        logger.info("Validating target series...")
        
        try:
            # Check if date column exists and has valid dates
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found")
                return False
            
            if df[date_col].isnull().any():
                logger.error(f"Date column '{date_col}' has null values")
                return False
            
            # Check if target column exists and has valid numeric values
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found")
                return False
            
            if df[target_col].isnull().any():
                logger.error(f"Target column '{target_col}' has null values")
                return False
            
            logger.info("Target series validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Target series validation failed: {e}")
            return False
    
    def validate_exogenous_features(self, df: pd.DataFrame, features: List[str]) -> bool:
        """Validate exogenous features data."""
        logger.info("Validating exogenous features...")
        
        # Check for missing values
        missing_features = []
        for feature in features:
            if feature not in df.columns:
                missing_features.append(feature)
            elif df[feature].isnull().any():
                logger.warning(f"Feature '{feature}' has missing values")
        
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        logger.info("Exogenous features validation passed")
        return True


class DataManager:
    """Main data management class that orchestrates all data operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.validator = DataValidator(config)
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Complete data loading and preparation pipeline."""
        logger.info("Starting data loading and preparation...")
        
        # Load data
        target_df = self.loader.load_target_series()
        exogenous_df = self.loader.load_exogenous_features()
        
        # Merge datasets
        merged_df = self.loader.merge_datasets(target_df, exogenous_df)
        
        # Get configuration
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        exogenous_features = self.config['model']['exogenous_vars']['features']
        
        # Validate data
        if not self.validator.validate_target_series(merged_df, target_col, date_col):
            raise ValueError("Target series validation failed")
        
        if not self.validator.validate_exogenous_features(merged_df, exogenous_features):
            raise ValueError("Exogenous features validation failed")
        
        # Preprocess data
        merged_df = self.preprocessor.remove_outliers(merged_df, target_col)
        merged_df = self.preprocessor.fill_missing_values(merged_df)
        
        # Create lag features if needed
        if self.config['model']['exogenous_vars']['enabled']:
            merged_df = self.preprocessor.create_lag_features(
                merged_df, 
                exogenous_features, 
                max_lags=3
            )
        
        # Prepare Prophet format
        prophet_df, available_features = self.preprocessor.prepare_prophet_data(
            merged_df, target_col, date_col, exogenous_features
        )
        
        logger.info("Data loading and preparation completed successfully")
        
        return prophet_df, available_features
