#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de execução específico para o projeto de previsão de casos criminais.
Utiliza os dados reais e configuração otimizada.
"""

import sys
from pathlib import Path
import click
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the pipeline class from main.py
from main import ProphetForecastingPipeline


@click.command()
@click.option('--config', '-c', default='configs/criminal_cases_config.yaml', 
              help='Path to configuration file')
@click.option('--output-dir', '-o', default='outputs', 
              help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.option('--quick', '-q', is_flag=True, 
              help='Quick run with minimal optimization')
def main(config: str, output_dir: str, verbose: bool, quick: bool):
    """Run the criminal cases forecasting pipeline."""
    
    print("="*70)
    print("PROPHET FORECASTING PIPELINE - CASOS CRIMINAIS TJGO")
    print("="*70)
    
    try:
        # Initialize pipeline
        pipeline = ProphetForecastingPipeline(config)
        
        # Update output directory in config if specified
        if output_dir != 'outputs':
            pipeline.config['output']['base_dir'] = output_dir
        
        # Set verbose logging if requested
        if verbose:
            pipeline.config['logging']['level'] = 'DEBUG'
            pipeline._setup_logging()
        
        # Quick run modifications
        if quick:
            logger.info("Running in quick mode - reducing optimization trials")
            pipeline.config['training']['hyperparameter_optimization']['n_trials'] = 10
            pipeline.config['training']['hyperparameter_optimization']['timeout'] = 600
            pipeline.config['training']['cv']['n_splits'] = 3
            pipeline.config['forecasting']['horizon_months'] = 6
        
        # Run pipeline
        results = pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTADOS DA PREVISÃO DE CASOS CRIMINAIS")
        print("="*70)
        
        metrics = results['evaluation_results']['metrics']
        print(f"\nPerformance do Modelo:")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        
        # Feature importance
        if 'feature_importance' in results['training_results']:
            print(f"\nImportância das Variáveis Exógenas:")
            feature_importance = results['training_results']['feature_importance']
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"  {feature}: {importance:.4f}")
        
        # Forecast summary
        forecast = results['forecast']
        prophet_df = results['data']['prophet_df']
        forecast_periods = len(forecast) - len(prophet_df)
        
        print(f"\nResumo da Previsão:")
        print(f"  Períodos previstos: {forecast_periods}")
        print(f"  Período da previsão: {forecast['ds'].iloc[-forecast_periods].iloc[0]} a {forecast['ds'].iloc[-1]}")
        print(f"  Média prevista: {forecast['yhat'].iloc[-forecast_periods:].mean():.2f}")
        print(f"  Tendência: {'Crescente' if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-forecast_periods] else 'Decrescente'}")
        
        print(f"\nArtefatos Salvos:")
        for artifact_type, path in results['artifacts'].items():
            print(f"  {artifact_type}: {path}")
        
        print(f"\nVisualizações Geradas:")
        for plot_type, path in results['plots'].items():
            print(f"  {plot_type}: {path}")
        
        print(f"\nRelatório: {results['report_path']}")
        
        # Component analysis
        if 'component_analysis' in results['evaluation_results']:
            component_analysis = results['evaluation_results']['component_analysis']
            if 'component_contributions' in component_analysis:
                print(f"\nContribuição dos Componentes:")
                for component, contribution in component_analysis['component_contributions'].items():
                    print(f"  {component}: {contribution:.3f}")
        
        print("\n" + "="*70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
        print(f"\nPróximos Passos:")
        print(f"  1. Abra o relatório HTML para análise detalhada")
        print(f"  2. Verifique as visualizações geradas")
        print(f"  3. Analise as métricas de performance")
        print(f"  4. Considere ajustes nos parâmetros se necessário")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nErro: {str(e)}")
        print(f"\nVerifique:")
        print(f"  1. Se os arquivos de dados existem nos caminhos especificados")
        print(f"  2. Se todas as dependências estão instaladas")
        print(f"  3. Se a configuração está correta")
        print(f"  4. Os logs em logs/prophet_pipeline.log para mais detalhes")
        sys.exit(1)


if __name__ == "__main__":
    main()
