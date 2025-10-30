#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para execução do pipeline SARIMAX.
"""

import sys
from pathlib import Path
import click
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sarimax.pipeline import SARIMAXPipeline


@click.command()
@click.option('--config', '-c', default='configs/sarimax_config.yaml',
              help='Path to configuration file')
@click.option('--skip-exploration', '-s', is_flag=True,
              help='Skip data exploration phase (useful for re-runs)')
@click.option('--exploration-only', '-e', is_flag=True,
              help='Run only data exploration phase')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def main(config: str, skip_exploration: bool, exploration_only: bool, verbose: bool):
    """Run the SARIMAX forecasting pipeline."""
    
    print("="*70)
    print("SARIMAX FORECASTING PIPELINE - CASOS CRIMINAIS TJGO")
    print("Metodologia CRISP-DM")
    print("="*70)
    
    try:
        # Inicializa pipeline
        pipeline = SARIMAXPipeline(config)
        
        # Configura verbose se solicitado
        if verbose:
            logger.remove()
            logger.add(sys.stdout, level="DEBUG")
        
        # Executa pipeline
        if exploration_only:
            print("\nExecutando apenas análise exploratória...")
            pipeline.run_data_exploration()
            print("\nAnálise exploratória concluída!")
        else:
            results = pipeline.run_full_pipeline(skip_exploration=skip_exploration)
            
            # Exibe resumo
            print("\n" + "="*70)
            print("RESULTADOS DA PREVISÃO - SARIMAX")
            print("="*70)
            
            # Métricas do modelo
            if 'model' in results:
                print(f"\nModelo Selecionado:")
                best_params = results['model']['best_params']
                print(f"  Ordem (p,d,q): {best_params['order']}")
                print(f"  Ordem Sazonal (P,D,Q,s): {best_params['seasonal_order']}")
                print(f"  AIC: {best_params.get('aic', 'N/A'):.2f}" if isinstance(best_params.get('aic'), (int, float)) else f"  AIC: {best_params.get('aic', 'N/A')}")
                print(f"  BIC: {best_params.get('bic', 'N/A'):.2f}" if isinstance(best_params.get('bic'), (int, float)) else f"  BIC: {best_params.get('bic', 'N/A')}")
            
            # Métricas de avaliação
            if 'evaluation' in results and 'metrics' in results['evaluation']:
                metrics = results['evaluation']['metrics']
                print(f"\nMétricas de Avaliação:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):  # Not NaN
                        print(f"  {key.upper()}: {value:.4f}")
            
            # Previsão
            if 'forecast' in results:
                forecast = results['forecast']
                print(f"\nPrevisão:")
                print(f"  Períodos previstos: {len(forecast)}")
                if 'date' in forecast.columns:
                    print(f"  Período: {forecast['date'].iloc[0]} a {forecast['date'].iloc[-1]}")
                if 'forecast' in forecast.columns:
                    print(f"  Média prevista: {forecast['forecast'].mean():.2f}")
            
            # Artefatos
            if 'artifacts' in results:
                print(f"\nArtefatos Salvos:")
                for artifact_type, path in results['artifacts'].items():
                    print(f"  {artifact_type}: {path}")
            
            print("\n" + "="*70)
            print("PIPELINE CONCLUÍDO COM SUCESSO!")
            print("="*70)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nErro: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

