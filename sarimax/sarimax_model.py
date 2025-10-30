#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISP-DM Fase 4: Modeling
Implementação do modelo SARIMAX usando pmdarima (auto_arima).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from pmdarima import auto_arima
    from pmdarima.arima import ARIMA
    from pmdarima.model_selection import train_test_split
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.error("pmdarima not installed. Install with: pip install pmdarima")


class SARIMAXModel:
    """
    Wrapper para modelo SARIMAX usando pmdarima.
    Implementa busca automática de hiperparâmetros e modelagem.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o modelo SARIMAX.
        
        Parameters:
        -----------
        config : Dict
            Configuração do pipeline
        """
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required. Install with: pip install pmdarima")
        
        self.config = config
        self.model_config = config.get('model', {}).get('auto_arima', {})
        self.model = None
        self.best_params = {}
        self.model_summary = None
        self.has_exogenous = False  # Rastreia se variáveis exógenas foram usadas
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> 'SARIMAXModel':
        """
        Treina o modelo SARIMAX usando auto_arima.
        
        Parameters:
        -----------
        y : pd.Series
            Série temporal alvo (deve ter índice DatetimeIndex)
        exog : Optional[pd.DataFrame]
            Variáveis exógenas (opcional)
            
        Returns:
        --------
        SARIMAXModel
            Self para method chaining
        """
        logger.info("Fitting SARIMAX model using auto_arima...")
        
        if exog is not None and len(exog) != len(y):
            raise ValueError("Length of y and exog must match")
        
        # Configurações do auto_arima baseadas na configuração
        arima_kwargs = self._build_arima_kwargs(exog is not None)
        
        logger.info("Starting auto_arima search...")
        logger.info(f"Configuration: {arima_kwargs}")
        
        # Armazena informação sobre variáveis exógenas
        self.has_exogenous = exog is not None
        
        # Treina modelo
        self.model = auto_arima(
            y,
            exogenous=exog,
            **arima_kwargs
        )
        
        # Extrai parâmetros ótimos
        self.best_params = {
            'order': self.model.order,  # (p, d, q)
            'seasonal_order': self.model.seasonal_order,  # (P, D, Q, s)
            'aic': self.model.aic(),
            'aicc': self.model.aicc(),
            'bic': self.model.bic(),
        }
        
        # Obtém resumo do modelo
        self.model_summary = str(self.model.summary())
        
        logger.info("Model fitted successfully")
        logger.info(f"Best order (p,d,q): {self.best_params['order']}")
        logger.info(f"Best seasonal order (P,D,Q,s): {self.best_params['seasonal_order']}")
        logger.info(f"AIC: {self.best_params['aic']:.2f}")
        logger.info(f"AICc: {self.best_params['aicc']:.2f}")
        logger.info(f"BIC: {self.best_params['bic']:.2f}")
        
        return self
    
    def _build_arima_kwargs(self, has_exog: bool) -> Dict:
        """
        Constrói dicionário de argumentos para auto_arima baseado na configuração.
        
        Parameters:
        -----------
        has_exog : bool
            Se há variáveis exógenas
            
        Returns:
        --------
        Dict
            Argumentos para auto_arima
        """
        kwargs = {
            # Ordens máximas
            'max_p': self.model_config.get('max_p', 5),
            'max_d': self.model_config.get('max_d', 2),
            'max_q': self.model_config.get('max_q', 5),
            'max_P': self.model_config.get('max_P', 2),
            'max_D': self.model_config.get('max_D', 1),
            'max_Q': self.model_config.get('max_Q', 2),
            
            # Sazonalidade
            'seasonal': self.model_config.get('seasonal', True),
            'm': self.model_config.get('m', 12) or self.model_config.get('seasonal_periods', 12),
            
            # Critério de informação
            'information_criterion': self.model_config.get('information_criterion', 'aicc'),
            
            # Busca
            'stepwise': self.model_config.get('stepwise', True),
            'trace': self.model_config.get('trace', True),
            'suppress_warnings': self.model_config.get('suppress_warnings', True),
            'error_action': self.model_config.get('error_action', 'ignore'),
            
            # Teste de estacionariedade
            'test': self.model_config.get('test', 'adf'),
            'test_kwargs': self.model_config.get('test_kwargs'),
            
            # Outras configurações
            'with_intercept': self.model_config.get('with_intercept', 'auto'),
            'method': self.model_config.get('method', 'lbfgs'),
            'max_order': self.model_config.get('max_order', 10),
            'n_jobs': self.model_config.get('n_jobs', 1),
        }
        
        # Remove valores None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return kwargs
    
    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None,
               return_conf_int: bool = False, alpha: float = 0.05) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Gera previsões para n períodos futuros.
        
        Parameters:
        -----------
        n_periods : int
            Número de períodos a prever
        exog : Optional[pd.DataFrame]
            Variáveis exógenas futuras (se necessário)
        return_conf_int : bool
            Se deve retornar intervalos de confiança
        alpha : float
            Nível de significância para intervalos de confiança
            
        Returns:
        --------
        Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]
            Previsões e opcionalmente intervalos de confiança
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating predictions for {n_periods} periods...")
        
        # Valida variáveis exógenas se necessário
        if self.has_exogenous and exog is None:
            raise ValueError("Exogenous variables required for prediction (model was trained with exogenous variables)")
        
        if exog is not None and len(exog) != n_periods:
            raise ValueError(f"Length of exog ({len(exog)}) must match n_periods ({n_periods})")
        
        # Gera previsões
        if return_conf_int:
            forecast, conf_int = self.model.predict(
                n_periods=n_periods,
                exogenous=exog,
                return_conf_int=True,
                alpha=alpha
            )
            
            # Converte para DataFrame com colunas apropriadas
            conf_int_df = pd.DataFrame(
                conf_int,
                columns=['lower', 'upper'],
                index=forecast.index
            )
            
            logger.info("Predictions with confidence intervals generated")
            return forecast, conf_int_df
        else:
            forecast = self.model.predict(
                n_periods=n_periods,
                exogenous=exog,
                return_conf_int=False
            )
            
            logger.info("Predictions generated")
            return forecast
    
    def forecast(self, y_future: Optional[pd.Series] = None, exog_future: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões para o período futuro definido na configuração.
        
        Parameters:
        -----------
        y_future : Optional[pd.Series]
            Valores reais futuros (para comparação, opcional)
        exog_future : Optional[pd.DataFrame]
            Variáveis exógenas futuras
            
        Returns:
        --------
        pd.DataFrame
            DataFrame com previsões, intervalos de confiança e valores reais (se fornecidos)
        """
        horizon = self.config.get('forecasting', {}).get('horizon_months', 12)
        alpha = self.config.get('forecasting', {}).get('prediction_intervals', {}).get('alpha', 0.05)
        
        # Gera previsões com intervalos
        forecast, conf_int = self.predict(
            n_periods=horizon,
            exog=exog_future,
            return_conf_int=True,
            alpha=alpha
        )
        
        # Cria DataFrame com resultados
        forecast_df = pd.DataFrame({
            'date': forecast.index,
            'forecast': forecast.values,
            'lower': conf_int['lower'].values,
            'upper': conf_int['upper'].values
        })
        
        # Adiciona valores reais se fornecidos
        if y_future is not None:
            forecast_df['actual'] = y_future.values
            forecast_df['error'] = forecast_df['actual'] - forecast_df['forecast']
            forecast_df['error_pct'] = (forecast_df['error'] / forecast_df['actual']) * 100
        
        return forecast_df
    
    def get_residuals(self) -> pd.Series:
        """
        Obtém resíduos do modelo treinado.
        
        Returns:
        --------
        pd.Series
            Resíduos do modelo
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.model.resid()
    
    def get_summary(self) -> str:
        """
        Obtém resumo textual do modelo.
        
        Returns:
        --------
        str
            Resumo do modelo
        """
        if self.model_summary is None:
            raise ValueError("Model must be fitted before getting summary")
        
        return self.model_summary
    
    def save_model(self, filepath: str) -> str:
        """
        Salva modelo treinado.
        
        Parameters:
        -----------
        filepath : str
            Caminho para salvar modelo
            
        Returns:
        --------
        str
            Caminho do arquivo salvo
        """
        import joblib
        
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'config': self.config,
            'has_exogenous': self.has_exogenous
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SARIMAXModel':
        """
        Carrega modelo salvo.
        
        Parameters:
        -----------
        filepath : str
            Caminho do modelo salvo
            
        Returns:
        --------
        SARIMAXModel
            Instância do modelo carregado
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.best_params = model_data['best_params']
        instance.has_exogenous = model_data.get('has_exogenous', False)
        instance.model_summary = str(instance.model.summary())
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance

