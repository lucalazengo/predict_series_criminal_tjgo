#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script executável direto para o pipeline de previsão de casos criminais.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules directly
from data import DataManager
from models import ProphetModelWrapper
from training import TrainingPipeline
from evaluation import ModelEvaluator
from utils import ReportGenerator, Visualizer, ArtifactManager, ConfigManager


def run_criminal_cases_pipeline():
    """Executa o pipeline completo para casos criminais."""
    
    print("="*70)
    print("PROPHET FORECASTING PIPELINE - CASOS CRIMINAIS TJGO")
    print("="*70)
    
    try:
        # Carregar configuração
        config_path = 'configs/criminal_cases_config.yaml'
        config = ConfigManager.load_config(config_path)
        
        # Usar configuração completa com otimização robusta
        # n_trials já está definido como 50 na configuração
        if 'training' not in config:
            config['training'] = {}
        if 'hyperparameter_optimization' not in config['training']:
            config['training']['hyperparameter_optimization'] = {}
        if 'n_trials' not in config['training']['hyperparameter_optimization']:
            config['training']['hyperparameter_optimization']['n_trials'] = 50
        
        # Aumentar splits de validação cruzada
        if 'cv' not in config['training']:
            config['training']['cv'] = {}
        config['training']['cv']['n_splits'] = 5  # Aumentado de 3 para 5
        
        print("Configuração carregada com sucesso!")
        
        # Inicializar componentes
        print("\nInicializando componentes do pipeline...")
        data_manager = DataManager(config)
        training_pipeline = TrainingPipeline(config)
        evaluator = ModelEvaluator(config)
        visualizer = Visualizer(config)
        artifact_manager = ArtifactManager(config)
        report_generator = ReportGenerator(config)
        
        # Passo 1: Carregar e preparar dados
        print("\nPasso 1: Carregando e preparando dados...")
        prophet_df, exogenous_features = data_manager.load_and_prepare_data()
        print(f"Dados preparados: {prophet_df.shape}")
        print(f"Variáveis exógenas: {exogenous_features}")
        
        # Passo 2: Treinar modelo
        print("\nPasso 2: Treinando modelo...")
        training_results = training_pipeline.run_training_pipeline(
            prophet_df, exogenous_features
        )
        print("Modelo treinado com sucesso!")
        
        # Passo 3: Fazer previsões
        print("\nPasso 3: Fazendo previsões...")
        model_wrapper = training_results['model']
        forecast = model_wrapper.predict(prophet_df, horizon_months=6)
        print(f"Previsões geradas: {len(forecast)} períodos")
        
        # Passo 4: Avaliar modelo
        print("\nPasso 4: Avaliando modelo...")
        evaluation_results = evaluator.evaluate_model(
            model_wrapper, forecast, prophet_df
        )
        
        # Mostrar métricas principais
        metrics = evaluation_results['metrics']
        print("\nMétricas de Performance:")
        print(f"  MAE (Mean Absolute Error): {metrics.get('mae', 'N/A'):.4f}")
        print(f"  RMSE (Root Mean Squared Error): {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  R² (Coefficient of Determination): {metrics.get('r2', 'N/A'):.4f}")
        print(f"  SMAPE (Symmetric MAPE): {metrics.get('smape', 'N/A'):.2f}%")
        
        # Passo 5: Gerar visualizações
        print("\nPasso 5: Gerando visualizações...")
        plots = {}
        
        # Gráfico de previsão
        plots['forecast'] = visualizer.create_forecast_plot(forecast, prophet_df)
        
        # Gráfico de componentes
        plots['components'] = visualizer.create_components_plot(forecast)
        
        # Gráfico de resíduos
        actual_values = prophet_df['y'].values
        predicted_values = forecast['yhat'].values
        plots['residuals'] = visualizer.create_residuals_plot(actual_values, predicted_values)
        
        print(f"Visualizações geradas: {len(plots)} gráficos")
        
        # Passo 6: Salvar artefatos
        print("\nPasso 6: Salvando artefatos...")
        artifacts = {}
        
        # Salvar modelo
        artifacts['model'] = artifact_manager.save_model(model_wrapper)
        
        # Salvar previsões
        artifacts['predictions'] = artifact_manager.save_predictions(forecast)
        
        # Salvar métricas
        artifacts['metrics'] = artifact_manager.save_metrics(metrics)
        
        # Salvar análise de features
        if 'feature_importance_analysis' in evaluation_results:
            feature_analysis_path = artifact_manager.output_dir / f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(feature_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results['feature_importance_analysis'], f, indent=2, default=str)
            artifacts['feature_analysis'] = str(feature_analysis_path)
            print(f"Análise de features salva: {feature_analysis_path}")
        
        # Salvar configuração
        artifacts['config'] = artifact_manager.save_config(config)
        
        print(f"Artefatos salvos: {len(artifacts)} arquivos")
        
        # Passo 7: Gerar relatório
        print("\nPasso 7: Gerando relatório...")
        report_path = report_generator.generate_report(
            training_results, evaluation_results, forecast, prophet_df
        )
        
        # Resumo final
        print("\n" + "="*70)
        print("RESULTADOS DA PREVISÃO DE CASOS CRIMINAIS")
        print("="*70)
        
        print(f"\nPerformance do Modelo:")
        print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  SMAPE: {metrics.get('smape', 'N/A'):.2f}%")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        
        # Análise detalhada de importância das variáveis
        if 'feature_importance_analysis' in evaluation_results:
            feature_analysis = evaluation_results['feature_importance_analysis']
            print(f"\n{'='*70}")
            print("ANÁLISE DETALHADA DE IMPORTÂNCIA DAS VARIÁVEIS EXÓGENAS")
            print(f"{'='*70}")
            
            if 'relative_importance' in feature_analysis and feature_analysis['relative_importance']:
                print("\nImportância Relativa (%):")
                for feature, importance in sorted(feature_analysis['relative_importance'].items(), 
                                                key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {importance:.2f}%")
            
            if 'correlation_analysis' in feature_analysis and feature_analysis['correlation_analysis']:
                print("\nCorrelação com Variável Alvo:")
                for feature, corr in sorted(feature_analysis['correlation_analysis'].items(), 
                                          key=lambda x: abs(x[1]), reverse=True):
                    print(f"  {feature}: {corr:.4f}")
        
        # Importância básica (fallback)
        elif 'feature_importance' in training_results:
            print(f"\nImportância das Variáveis Exógenas:")
            feature_importance = training_results['feature_importance']
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: abs(x[1]), reverse=True):
                print(f"  {feature}: {importance:.4f}")
        
        # Resumo da previsão
        forecast_periods = len(forecast) - len(prophet_df)
        print(f"\nResumo da Previsão:")
        print(f"  Períodos previstos: {forecast_periods}")
        print(f"  Período da previsão: {forecast['ds'].iloc[-forecast_periods].iloc[0]} a {forecast['ds'].iloc[-1]}")
        print(f"  Média prevista: {forecast['yhat'].iloc[-forecast_periods:].mean():.2f}")
        
        print(f"\nArtefatos Salvos:")
        for artifact_type, path in artifacts.items():
            print(f"  {artifact_type}: {path}")
        
        print(f"\nVisualizações Geradas:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path}")
        
        print(f"\nRelatório: {report_path}")
        
        # Gerar relatório detalhado automaticamente
        print("\n" + "="*70)
        print("GERANDO RELATÓRIO TÉCNICO DETALHADO...")
        print("="*70)
        
        try:
            import subprocess
            result = subprocess.run(['python3', 'generate_detailed_report.py'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"⚠️ Aviso ao gerar relatório detalhado: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Não foi possível gerar relatório detalhado automaticamente: {e}")
            print("Execute manualmente: python3 generate_detailed_report.py")
        
        print("\n" + "="*70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'forecast': forecast,
            'plots': plots,
            'artifacts': artifacts,
            'report_path': report_path
        }
        
    except Exception as e:
        print(f"\nErro na execução do pipeline: {str(e)}")
        print(f"\nVerifique:")
        print(f"  1. Se os arquivos de dados existem")
        print(f"  2. Se todas as dependências estão instaladas")
        print(f"  3. Se a configuração está correta")
        raise


if __name__ == "__main__":
    results = run_criminal_cases_pipeline()
