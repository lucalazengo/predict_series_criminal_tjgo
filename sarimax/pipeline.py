#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISP-DM Fase 6: Deployment
Pipeline completo para previsão com SARIMAX.
Orquestra todas as fases do CRISP-DM.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import yaml
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from sarimax.data_exploration import SARIMAXDataExplorer
    from sarimax.data_preparation import SARIMAXDataPreparer
    from sarimax.sarimax_model import SARIMAXModel
    from sarimax.evaluation import SARIMAXEvaluator
except ImportError:
    # Fallback para imports relativos
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_exploration import SARIMAXDataExplorer
    from data_preparation import SARIMAXDataPreparer
    from sarimax_model import SARIMAXModel
    from evaluation import SARIMAXEvaluator


class SARIMAXPipeline:
    """
    Pipeline completo seguindo metodologia CRISP-DM.
    """
    
    def __init__(self, config_path: str):
        """
        Inicializa o pipeline.
        
        Parameters:
        -----------
        config_path : str
            Caminho para arquivo de configuração YAML
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Inicializa componentes
        self.data_preparer = SARIMAXDataPreparer(self.config)
        self.model = None
        self.evaluator = SARIMAXEvaluator(self.config)
        
        # Resultados
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração YAML."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """Configura logging."""
        log_config = self.config.get('logging', {})
        
        # Remove logger padrão
        logger.remove()
        
        # Console
        logger.add(
            sys.stdout,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        )
        
        # Arquivo
        log_file = log_config.get('file', 'logs/sarimax_pipeline.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
            rotation="10 MB",
            retention="30 days"
        )
        
        logger.info("Logging configured")
    
    def run_data_exploration(self) -> Dict:
        """
        CRISP-DM Fase 2: Data Understanding
        Executa análise exploratória completa.
        """
        logger.info("="*70)
        logger.info("CRISP-DM FASE 2: DATA UNDERSTANDING")
        logger.info("="*70)
        
        explorer = SARIMAXDataExplorer(
            target_path=self.config['data']['target_series_path'],
            exogenous_path=self.config['data']['exogenous_features_path']
        )
        
        exploration_results = explorer.run_full_analysis()
        
        self.results['data_exploration'] = exploration_results
        
        return exploration_results
    
    def run_data_preparation(self) -> tuple:
        """
        CRISP-DM Fase 3: Data Preparation
        Prepara dados para modelagem.
        """
        logger.info("="*70)
        logger.info("CRISP-DM FASE 3: DATA PREPARATION")
        logger.info("="*70)
        
        y, exog, dates = self.data_preparer.run_preparation_pipeline()
        
        self.results['data'] = {
            'y': y,
            'exog': exog,
            'dates': dates,
            'n_observations': len(y)
        }
        
        return y, exog, dates
    
    def train_model(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> SARIMAXModel:
        """
        CRISP-DM Fase 4: Modeling
        Treina modelo SARIMAX.
        """
        logger.info("="*70)
        logger.info("CRISP-DM FASE 4: MODELING")
        logger.info("="*70)
        
        # Prepara dados de treinamento
        training_config = self.config.get('training', {})
        train_start = training_config.get('train_start')
        train_end = training_config.get('train_end')
        
        y_train, exog_train = self.data_preparer.prepare_training_data(
            y, exog, train_start, train_end
        )
        
        # Treina modelo
        self.model = SARIMAXModel(self.config)
        self.model.fit(y_train, exog_train)
        
        self.results['model'] = {
            'best_params': self.model.best_params,
            'summary': self.model.get_summary()
        }
        
        return self.model
    
    def evaluate_model(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> Dict:
        """
        CRISP-DM Fase 5: Evaluation
        Avalia modelo treinado.
        """
        logger.info("="*70)
        logger.info("CRISP-DM FASE 5: EVALUATION")
        logger.info("="*70)
        
        # Prepara dados de validação
        training_config = self.config.get('training', {})
        val_start = training_config.get('val_start')
        val_end = training_config.get('val_end')
        
        y_val = None
        exog_val = None
        
        if val_start and val_end:
            mask = (y.index >= val_start) & (y.index <= val_end)
            y_val = y[mask].copy() if mask.sum() > 0 else None
            if exog is not None:
                exog_val = exog[mask].copy() if mask.sum() > 0 else None
        
        # Avaliação completa
        evaluation_results = self.evaluator.evaluate_model(
            self.model,
            y_train=self.results['data']['y'][:len(self.results['data']['y'])],
            y_test=y_val,
            exog_train=None,  # Já foi usado no treino
            exog_test=exog_val
        )
        
        self.results['evaluation'] = evaluation_results
        
        return evaluation_results
    
    def generate_forecast(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões futuras.
        """
        logger.info("="*70)
        logger.info("GENERATING FORECASTS")
        logger.info("="*70)
        
        horizon = self.config.get('forecasting', {}).get('horizon_months', 12)
        
        # Prepara variáveis exógenas futuras se necessário
        exog_future = None
        if exog is not None:
            # Assumimos que temos dados futuros de exógenas
            # Em produção, isso viria de previsões ou dados externos
            last_date = y.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
            
            # Para exemplo, replicamos último valor (em produção, fazer previsão)
            exog_future = pd.DataFrame(
                index=future_dates,
                columns=exog.columns,
                data=exog.iloc[-1:].values.repeat(horizon, axis=0)
            )
        
        forecast_df = self.model.forecast(exog_future=exog_future)
        
        self.results['forecast'] = forecast_df
        
        logger.info(f"Forecast generated for {horizon} periods")
        
        return forecast_df
    
    def save_artifacts(self) -> Dict[str, str]:
        """
        Salva todos os artefatos do pipeline.
        """
        logger.info("="*70)
        logger.info("SAVING ARTIFACTS")
        logger.info("="*70)
        
        output_config = self.config.get('output', {})
        base_dir = Path(output_config.get('base_dir', 'outputs/sarimax'))
        base_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        artifacts = {}
        
        # Salva modelo
        if output_config.get('save_model', True):
            model_path = base_dir / f"models/sarimax_model_{timestamp}.joblib"
            self.model.save_model(str(model_path))
            artifacts['model'] = str(model_path)
        
        # Salva previsões
        if output_config.get('save_predictions', True) and 'forecast' in self.results:
            pred_path = base_dir / f"predictions/forecast_{timestamp}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            self.results['forecast'].to_csv(pred_path, index=False)
            artifacts['predictions'] = str(pred_path)
        
        # Salva métricas
        if output_config.get('save_metrics', True) and 'evaluation' in self.results:
            import json
            metrics_path = base_dir / f"reports/metrics_{timestamp}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(self.results['evaluation'], f, indent=2, default=str)
            artifacts['metrics'] = str(metrics_path)
        
        # Salva configuração
        config_path = base_dir / f"reports/config_{timestamp}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        artifacts['config'] = str(config_path)
        
        logger.info("Artifacts saved:")
        for artifact_type, path in artifacts.items():
            logger.info(f"  {artifact_type}: {path}")
        
        return artifacts
    
    def run_full_pipeline(self, skip_exploration: bool = False) -> Dict[str, Any]:
        """
        Executa pipeline completo seguindo CRISP-DM.
        
        Parameters:
        -----------
        skip_exploration : bool
            Se True, pula análise exploratória (útil em reexecuções)
            
        Returns:
        --------
        Dict[str, Any]
            Resultados completos do pipeline
        """
        logger.info("="*70)
        logger.info("SARIMAX FORECASTING PIPELINE")
        logger.info("CRISP-DM Methodology")
        logger.info("="*70)
        
        try:
            # Fase 2: Data Understanding
            if not skip_exploration:
                self.run_data_exploration()
            
            # Fase 3: Data Preparation
            y, exog, dates = self.run_data_preparation()
            
            # Fase 4: Modeling
            self.train_model(y, exog)
            
            # Fase 5: Evaluation
            self.evaluate_model(y, exog)
            
            # Gera previsões
            forecast_df = self.generate_forecast(y, exog)
            
            # Salva artefatos
            artifacts = self.save_artifacts()
            self.results['artifacts'] = artifacts
            
            logger.info("="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sarimax_config.yaml"
    
    pipeline = SARIMAXPipeline(config_path)
    results = pipeline.run_full_pipeline()

