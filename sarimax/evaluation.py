#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISP-DM Fase 5: Evaluation
Avaliação completa do modelo SARIMAX com diagnósticos e métricas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels/scipy not available. Some diagnostics will be skipped")


class SARIMAXEvaluator:
    """
    Avaliador completo para modelos SARIMAX.
    Calcula métricas, realiza diagnósticos e validações.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o avaliador.
        
        Parameters:
        -----------
        config : Dict
            Configuração do pipeline
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de avaliação.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Valores reais
        y_pred : np.ndarray
            Valores previstos
            
        Returns:
        --------
        Dict[str, float]
            Dicionário com métricas calculadas
        """
        logger.info("Calculating evaluation metrics...")
        
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid data for metrics calculation")
            return {}
        
        metrics = {}
        
        # MAE - Mean Absolute Error
        if 'mae' in self.eval_config.get('metrics', []):
            metrics['mae'] = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        # MSE - Mean Squared Error
        if 'mse' in self.eval_config.get('metrics', []):
            metrics['mse'] = np.mean((y_true_clean - y_pred_clean) ** 2)
        
        # RMSE - Root Mean Squared Error
        if 'rmse' in self.eval_config.get('metrics', []):
            metrics['rmse'] = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
        
        # MAPE - Mean Absolute Percentage Error
        if 'mape' in self.eval_config.get('metrics', []):
            mask = y_true_clean != 0
            if mask.sum() > 0:
                metrics['mape'] = np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100
            else:
                metrics['mape'] = np.nan
        
        # SMAPE - Symmetric Mean Absolute Percentage Error
        if 'smape' in self.eval_config.get('metrics', []):
            denominator = (np.abs(y_true_clean) + np.abs(y_pred_clean)) / 2
            mask = denominator != 0
            if mask.sum() > 0:
                metrics['smape'] = np.mean(np.abs(y_true_clean[mask] - y_pred_clean[mask]) / denominator[mask]) * 100
            else:
                metrics['smape'] = np.nan
        
        # R² - Coefficient of Determination
        if 'r2' in self.eval_config.get('metrics', []):
            ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
            ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            if ss_tot > 0:
                metrics['r2'] = 1 - (ss_res / ss_tot)
            else:
                metrics['r2'] = np.nan
        
        logger.info("Metrics calculated:")
        for key, value in metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")
        
        return metrics
    
    def analyze_residuals(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Analisa resíduos do modelo.
        
        Parameters:
        -----------
        residuals : pd.Series
            Resíduos do modelo
            
        Returns:
        --------
        Dict[str, Any]
            Resultados da análise de resíduos
        """
        logger.info("Analyzing residuals...")
        
        residuals_clean = residuals.dropna()
        results = {}
        
        # Estatísticas básicas
        results['mean'] = float(residuals_clean.mean())
        results['std'] = float(residuals_clean.std())
        results['min'] = float(residuals_clean.min())
        results['max'] = float(residuals_clean.max())
        
        logger.info(f"Residuals stats: mean={results['mean']:.4f}, std={results['std']:.4f}")
        
        # Testes se disponíveis
        if STATSMODELS_AVAILABLE and self.eval_config.get('residual_analysis', {}).get('enabled', False):
            tests = self.eval_config.get('residual_analysis', {}).get('tests', [])
            
            # Teste de Ljung-Box (autocorrelação dos resíduos)
            if 'ljung_box' in tests:
                try:
                    lb_result = acorr_ljungbox(residuals_clean, lags=min(10, len(residuals_clean)//4), return_df=True)
                    results['ljung_box'] = {
                        'statistic': float(lb_result['lb_stat'].iloc[-1]),
                        'pvalue': float(lb_result['lb_pvalue'].iloc[-1]),
                        'is_white_noise': float(lb_result['lb_pvalue'].iloc[-1]) > 0.05
                    }
                    logger.info(f"Ljung-Box test: p-value={results['ljung_box']['pvalue']:.4f} "
                              f"({'White noise' if results['ljung_box']['is_white_noise'] else 'Not white noise'})")
                except Exception as e:
                    logger.warning(f"Ljung-Box test failed: {e}")
                    results['ljung_box'] = None
            
            # Teste de normalidade
            if 'normality' in tests:
                try:
                    # Jarque-Bera (mais robusto para amostras maiores)
                    if len(residuals_clean) > 50:
                        jb_stat, jb_pvalue = jarque_bera(residuals_clean)
                        results['normality'] = {
                            'test': 'jarque_bera',
                            'statistic': float(jb_stat),
                            'pvalue': float(jb_pvalue),
                            'is_normal': float(jb_pvalue) > 0.05
                        }
                    else:
                        # Shapiro-Wilk (melhor para amostras pequenas)
                        sw_stat, sw_pvalue = shapiro(residuals_clean[:5000])  # Limite para Shapiro
                        results['normality'] = {
                            'test': 'shapiro_wilk',
                            'statistic': float(sw_stat),
                            'pvalue': float(sw_pvalue),
                            'is_normal': float(sw_pvalue) > 0.05
                        }
                    
                    logger.info(f"Normality test ({results['normality']['test']}): "
                              f"p-value={results['normality']['pvalue']:.4f} "
                              f"({'Normal' if results['normality']['is_normal'] else 'Not normal'})")
                except Exception as e:
                    logger.warning(f"Normality test failed: {e}")
                    results['normality'] = None
            
            # Teste de heterocedasticidade (simplificado)
            if 'heteroskedasticity' in tests:
                try:
                    # Divide resíduos em grupos e compara variâncias
                    n_groups = 3
                    group_size = len(residuals_clean) // n_groups
                    variances = []
                    for i in range(n_groups):
                        group = residuals_clean.iloc[i*group_size:(i+1)*group_size]
                        variances.append(group.var())
                    
                    # Teste F simples entre primeiro e último grupo
                    var_ratio = variances[-1] / variances[0] if variances[0] > 0 else np.nan
                    results['heteroskedasticity'] = {
                        'variances': [float(v) for v in variances],
                        'variance_ratio': float(var_ratio),
                        'is_homoskedastic': 0.5 < var_ratio < 2.0  # Critério simples
                    }
                    logger.info(f"Heteroskedasticity check: variance ratio={var_ratio:.4f} "
                              f"({'Homoskedastic' if results['heteroskedasticity']['is_homoskedastic'] else 'Heteroskedastic'})")
                except Exception as e:
                    logger.warning(f"Heteroskedasticity test failed: {e}")
                    results['heteroskedasticity'] = None
        
        return results
    
    def time_series_cross_validation(self, model, y: pd.Series, exog: Optional[pd.DataFrame] = None,
                                     n_splits: int = 5) -> Dict[str, Any]:
        """
        Realiza validação cruzada temporal.
        
        Parameters:
        -----------
        model : SARIMAXModel
            Modelo já treinado (usado apenas para obter configuração)
        y : pd.Series
            Série temporal completa
        exog : Optional[pd.DataFrame]
            Variáveis exógenas completas
        n_splits : int
            Número de splits
            
        Returns:
        --------
        Dict[str, Any]
            Resultados da validação cruzada
        """
        logger.info(f"Performing time series cross-validation with {n_splits} splits...")
        
        # Divide dados em splits temporais
        n = len(y)
        test_size = max(1, n // (n_splits + 1))
        
        cv_metrics = []
        
        for i in range(n_splits):
            # Define índices de treino e teste
            train_end = n - test_size * (n_splits - i)
            test_start = train_end
            test_end = min(test_start + test_size, n)
            
            if train_end <= 0 or test_start >= n:
                continue
            
            # Divide dados
            y_train = y.iloc[:train_end]
            y_test = y.iloc[test_start:test_end]
            
            exog_train = None
            exog_test = None
            if exog is not None:
                exog_train = exog.iloc[:train_end]
                exog_test = exog.iloc[test_start:test_end]
            
            try:
                # Treina modelo no split
                # Import local para evitar dependência circular
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent))
                from sarimax.sarimax_model import SARIMAXModel
                split_model = SARIMAXModel(model.config)
                split_model.fit(y_train, exog_train)
                
                # Previsões
                n_test = len(y_test)
                y_pred = split_model.predict(n_periods=n_test, exog=exog_test)
                
                # Calcula métricas
                metrics = self.calculate_metrics(y_test.values, y_pred.values)
                metrics['split'] = i + 1
                cv_metrics.append(metrics)
                
                logger.info(f"Split {i+1}: RMSE={metrics.get('rmse', np.nan):.2f}, "
                          f"MAPE={metrics.get('mape', np.nan):.2f}%")
                
            except Exception as e:
                logger.warning(f"Split {i+1} failed: {e}")
                continue
        
        # Agrega resultados
        if len(cv_metrics) == 0:
            logger.warning("No successful CV splits")
            return {}
        
        results = {
            'splits': cv_metrics,
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Calcula médias e desvios
        metric_names = set()
        for split_metrics in cv_metrics:
            metric_names.update(split_metrics.keys())
        
        for metric_name in metric_names:
            if metric_name == 'split':
                continue
            values = [m.get(metric_name) for m in cv_metrics if metric_name in m and not np.isnan(m.get(metric_name, np.nan))]
            if values:
                results['mean_metrics'][metric_name] = float(np.mean(values))
                results['std_metrics'][metric_name] = float(np.std(values))
        
        logger.info("Cross-validation completed:")
        for metric_name, mean_val in results['mean_metrics'].items():
            std_val = results['std_metrics'].get(metric_name, 0)
            logger.info(f"  {metric_name.upper()}: {mean_val:.4f} (±{std_val:.4f})")
        
        return results
    
    def evaluate_model(self, model, y_train: pd.Series, y_test: Optional[pd.Series] = None,
                      exog_train: Optional[pd.DataFrame] = None,
                      exog_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Avaliação completa do modelo.
        
        Parameters:
        -----------
        model : SARIMAXModel
            Modelo treinado
        y_train : pd.Series
            Série temporal de treino
        y_test : Optional[pd.Series]
            Série temporal de teste (opcional)
        exog_train : Optional[pd.DataFrame]
            Variáveis exógenas de treino
        exog_test : Optional[pd.DataFrame]
            Variáveis exógenas de teste
            
        Returns:
        --------
        Dict[str, Any]
            Resultados completos da avaliação
        """
        logger.info("Running comprehensive model evaluation...")
        
        results = {
            'model_params': model.best_params,
            'residual_analysis': None,
            'metrics': {},
            'cv_results': None
        }
        
        # Análise de resíduos
        try:
            residuals = model.get_residuals()
            results['residual_analysis'] = self.analyze_residuals(residuals)
        except Exception as e:
            logger.warning(f"Residual analysis failed: {e}")
        
        # Métricas em conjunto de teste se disponível
        if y_test is not None:
            try:
                n_test = len(y_test)
                y_pred = model.predict(n_periods=n_test, exog=exog_test)
                results['metrics'] = self.calculate_metrics(y_test.values, y_pred.values)
            except Exception as e:
                logger.warning(f"Test set evaluation failed: {e}")
        
        # Validação cruzada se configurada
        if self.config.get('training', {}).get('cv', {}).get('n_splits', 0) > 0:
            try:
                # Precisamos da série completa para CV
                # Aqui assumimos que temos apenas treino, então pulamos CV
                pass
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        logger.info("Model evaluation completed")
        
        return results

