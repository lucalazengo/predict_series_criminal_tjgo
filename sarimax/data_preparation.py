#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISP-DM Fase 3: Data Preparation
Preparação de dados específica para modelagem SARIMAX.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class SARIMAXDataPreparer:
    """
    Prepara dados para modelagem SARIMAX.
    Implementa transformações específicas para séries temporais.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o preparador de dados.
        
        Parameters:
        -----------
        config : Dict
            Configuração do pipeline
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.preprocessing_config = self.data_config.get('preprocessing', {})
        self.model_config = config.get('model', {})
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Carrega e une os datasets de série alvo e variáveis exógenas.
        
        Returns:
        --------
        pd.DataFrame
            Dataset merged com todas as colunas
        """
        logger.info("Loading and merging datasets...")
        
        # Carrega série alvo
        target_path = self.data_config['target_series_path']
        target_df = pd.read_csv(target_path)
        target_df[self.data_config['date_column']] = pd.to_datetime(
            target_df[self.data_config['date_column']]
        )
        target_df = target_df.sort_values(self.data_config['date_column']).reset_index(drop=True)
        
        # Carrega variáveis exógenas
        exogenous_path = self.data_config['exogenous_features_path']
        exogenous_df = pd.read_csv(exogenous_path)
        exogenous_df['data'] = pd.to_datetime(exogenous_df['data'])
        exogenous_df = exogenous_df.sort_values('data').reset_index(drop=True)
        
        # Merge
        merged_df = pd.merge(
            target_df,
            exogenous_df,
            left_on=self.data_config['date_column'],
            right_on='data',
            how='inner'
        ).drop(columns=['data']).reset_index(drop=True)
        
        logger.info(f"Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        logger.info(f"Date range: {merged_df[self.data_config['date_column']].min()} to {merged_df[self.data_config['date_column']].max()}")
        
        return merged_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa os dados: remove outliers e preenche valores faltantes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset para limpar
            
        Returns:
        --------
        pd.DataFrame
            Dataset limpo
        """
        logger.info("Cleaning data...")
        
        target_col = self.data_config['target_column']
        original_shape = df.shape
        
        # Remove outliers (opcional para SARIMAX)
        if self.preprocessing_config.get('remove_outliers', False):
            df = self._remove_outliers(df, target_col)
            logger.info(f"Removed {original_shape[0] - df.shape[0]} outliers")
        
        # Preenche valores faltantes
        if self.preprocessing_config.get('fill_missing', True):
            df = self._fill_missing_values(df)
            missing_after = df.isnull().sum().sum()
            if missing_after > 0:
                logger.warning(f"{missing_after} missing values remaining after fill")
            else:
                logger.info("All missing values filled")
        
        return df.reset_index(drop=True)
    
    def _remove_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Remove outliers usando método configurado."""
        method = self.preprocessing_config.get('outlier_method', 'iqr')
        threshold = self.preprocessing_config.get('outlier_threshold', 3.0)
        
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
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche valores faltantes usando método configurado."""
        method = self.preprocessing_config.get('fill_method', 'forward_fill')
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            df = df.interpolate(method='time', limit_direction='both')
        elif method == 'mean':
            df = df.fillna(df.mean())
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Cria features de lag para variáveis exógenas.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        features : List[str]
            Lista de features para criar lags
            
        Returns:
        --------
        pd.DataFrame
            Dataset com features de lag
        """
        if not self.model_config.get('exogenous_vars', {}).get('create_lags', False):
            return df
        
        max_lags = self.model_config.get('exogenous_vars', {}).get('max_lags', 3)
        
        logger.info(f"Creating lag features (max_lags={max_lags})...")
        
        df_lagged = df.copy()
        
        for lag in range(1, max_lags + 1):
            for feature in features:
                if feature in df.columns:
                    df_lagged[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Remove linhas com NaN dos lags
        df_lagged = df_lagged.dropna().reset_index(drop=True)
        
        logger.info(f"Created lag features. New shape: {df_lagged.shape}")
        
        return df_lagged
    
    def prepare_sarimax_data(self, df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.DataFrame], pd.DatetimeIndex]:
        """
        Prepara dados no formato necessário para SARIMAX.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset completo
            
        Returns:
        --------
        Tuple[pd.Series, Optional[pd.DataFrame], pd.DatetimeIndex]
            - Série temporal alvo (y)
            - Variáveis exógenas (exog) ou None
            - Índice temporal (dates)
        """
        logger.info("Preparing data for SARIMAX...")
        
        target_col = self.data_config['target_column']
        date_col = self.data_config['date_column']
        
        # Extrai série alvo
        y = df[target_col].copy()
        
        # Extrai índice temporal
        dates = pd.to_datetime(df[date_col])
        
        # Extrai variáveis exógenas
        exog = None
        if self.model_config.get('exogenous_vars', {}).get('enabled', False):
            exog_features = self._get_exogenous_features(df)
            if exog_features:
                exog = df[exog_features].copy()
                logger.info(f"Exogenous variables: {len(exog_features)} features")
                logger.info(f"Features: {exog_features}")
            else:
                logger.warning("No exogenous features available")
        else:
            logger.info("Exogenous variables disabled")
        
        # Define índice temporal na série
        y.index = dates
        
        if exog is not None:
            exog.index = dates
        
        logger.info(f"SARIMAX data prepared: {len(y)} observations")
        logger.info(f"Target series range: {y.min():.2f} to {y.max():.2f}")
        
        return y, exog, dates
    
    def _get_exogenous_features(self, df: pd.DataFrame) -> List[str]:
        """
        Obtém lista de features exógenas disponíveis.
        Inclui features originais e lags se configurado.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        List[str]
            Lista de features exógenas
        """
        base_features = self.model_config.get('exogenous_vars', {}).get('features', [])
        
        available_features = []
        
        for feature in base_features:
            # Feature original
            if feature in df.columns:
                available_features.append(feature)
            
            # Features de lag se criadas
            if self.model_config.get('exogenous_vars', {}).get('create_lags', False):
                max_lags = self.model_config.get('exogenous_vars', {}).get('max_lags', 3)
                for lag in range(1, max_lags + 1):
                    lag_feature = f'{feature}_lag_{lag}'
                    if lag_feature in df.columns:
                        available_features.append(lag_feature)
        
        return available_features
    
    def prepare_training_data(self, y: pd.Series, exog: Optional[pd.DataFrame],
                             train_start: str, train_end: str) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Prepara dados de treinamento filtrando por período.
        
        Parameters:
        -----------
        y : pd.Series
            Série temporal completa
        exog : Optional[pd.DataFrame]
            Variáveis exógenas completas
        train_start : str
            Data inicial de treinamento
        train_end : str
            Data final de treinamento
            
        Returns:
        --------
        Tuple[pd.Series, Optional[pd.DataFrame]]
            Dados de treinamento
        """
        logger.info(f"Preparing training data: {train_start} to {train_end}")
        
        # Filtra por período
        mask = (y.index >= train_start) & (y.index <= train_end)
        y_train = y[mask].copy()
        
        exog_train = None
        if exog is not None:
            exog_train = exog[mask].copy()
        
        logger.info(f"Training data: {len(y_train)} observations")
        
        return y_train, exog_train
    
    def run_preparation_pipeline(self) -> Tuple[pd.Series, Optional[pd.DataFrame], pd.DatetimeIndex]:
        """
        Executa pipeline completo de preparação de dados.
        
        Returns:
        --------
        Tuple[pd.Series, Optional[pd.DataFrame], pd.DatetimeIndex]
            Série alvo, variáveis exógenas, e índice temporal
        """
        logger.info("Starting SARIMAX data preparation pipeline...")
        
        # 1. Carrega e une dados
        merged_df = self.load_and_merge_data()
        
        # 2. Limpa dados
        cleaned_df = self.clean_data(merged_df)
        
        # 3. Cria features de lag se configurado
        exogenous_config = self.model_config.get('exogenous_vars', {})
        if exogenous_config.get('enabled', False) and exogenous_config.get('create_lags', False):
            base_features = exogenous_config.get('features', [])
            cleaned_df = self.create_lag_features(cleaned_df, base_features)
        
        # 4. Prepara formato SARIMAX
        y, exog, dates = self.prepare_sarimax_data(cleaned_df)
        
        logger.info("Data preparation pipeline completed successfully")
        
        return y, exog, dates

