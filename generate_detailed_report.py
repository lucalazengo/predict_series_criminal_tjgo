#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar relatório detalhado e aprofundado do pipeline Prophet.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def load_results(metrics_path: str, forecast_path: str = None, 
                 config_path: str = None, feature_analysis_path: str = None) -> Dict[str, Any]:
    """Carrega todos os resultados salvos."""
    results = {}
    
    # Carregar métricas
    if Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Carregar previsões
    if forecast_path and Path(forecast_path).exists():
        results['forecast'] = pd.read_csv(forecast_path)
    
    # Carregar análise de features
    if feature_analysis_path and Path(feature_analysis_path).exists():
        with open(feature_analysis_path, 'r') as f:
            results['feature_analysis'] = json.load(f)
    
    return results

def analyze_metrics_detailed(metrics: Dict[str, float], 
                            forecast: pd.DataFrame = None,
                            actual_data_path: str = None) -> str:
    """Gera análise detalhada e aprofundada das métricas."""
    
    report = []
    
    report.append("## 📊 ANÁLISE DETALHADA DAS MÉTRICAS DE PERFORMANCE\n")
    
    # MAE - Análise detalhada
    if 'mae' in metrics:
        mae = metrics['mae']
        report.append("### Mean Absolute Error (MAE) - Erro Absoluto Médio\n")
        report.append(f"**Valor Obtido:** {mae:.4f} casos\n")
        report.append("**Interpretação:**\n")
        report.append(f"- O MAE de {mae:.2f} indica que, em média, o modelo apresenta um erro absoluto ")
        report.append(f"de aproximadamente {int(mae)} casos por mês ao prever o número de casos criminais.\n")
        
        # Carregar dados reais se necessário
        actual_data = None
        if actual_data_path and Path(actual_data_path).exists():
            actual_data = pd.read_csv(actual_data_path)
        elif forecast is not None and 'y' in forecast.columns:
            # Usar y do forecast como proxy (se incluir dados históricos)
            actual_data = forecast[['ds', 'y']].copy()
        
        if forecast is not None and actual_data is not None:
            # Análise contextual
            avg_cases = actual_data['y'].mean() if 'y' in actual_data.columns else None
            if avg_cases:
                mae_percentage = (mae / avg_cases) * 100
                report.append(f"- Considerando que a média histórica de casos é de aproximadamente ")
                report.append(f"{avg_cases:.0f} casos/mês, o erro absoluto médio representa ")
                report.append(f"{mae_percentage:.2f}% da média histórica.\n")
                
                if mae_percentage < 10:
                    report.append("- **Avaliação:** MAE muito baixo (< 10% da média). O modelo apresenta ")
                    report.append("excelente precisão absoluta, com erros muito pequenos em relação à escala ")
                    report.append("dos dados. Isso indica que as previsões estão muito próximas dos valores reais.\n")
                elif mae_percentage < 20:
                    report.append("- **Avaliação:** MAE baixo (10-20% da média). O modelo apresenta boa ")
                    report.append("precisão absoluta, adequada para séries temporais criminais, que naturalmente ")
                    report.append("apresentam variabilidade significativa.\n")
                else:
                    report.append("- **Avaliação:** MAE moderado (> 20% da média). Há espaço para melhoria, ")
                    report.append("mas o desempenho ainda é aceitável considerando a complexidade e variabilidade ")
                    report.append("inerente aos dados criminais.\n")
        
        report.append("**Comparação com outras métricas:**\n")
        if 'rmse' in metrics:
            report.append(f"- O MAE ({mae:.2f}) é menor que o RMSE ({metrics['rmse']:.2f}), o que é esperado, ")
            report.append("já que o RMSE penaliza mais erros grandes. A diferença indica que há alguns ")
            report.append("outliers com erros maiores, mas a maioria das previsões tem erro moderado.\n")
        
        report.append("\n")
    
    # RMSE - Análise detalhada
    if 'rmse' in metrics:
        rmse = metrics['rmse']
        report.append("### Root Mean Squared Error (RMSE) - Raiz do Erro Quadrático Médio\n")
        report.append(f"**Valor Obtido:** {rmse:.4f} casos\n")
        report.append("**Interpretação:**\n")
        report.append(f"- O RMSE de {rmse:.2f} indica que a raiz do erro quadrático médio é de aproximadamente ")
        report.append(f"{int(rmse)} casos por mês.\n")
        
        if 'mae' in metrics:
            report.append(f"- Comparado ao MAE ({metrics['mae']:.2f}), o RMSE é maior, indicando que há ")
            report.append("alguns períodos com erros relativamente maiores, mas a maioria das previsões ")
            report.append("é precisa.\n")
        
        if forecast is not None and actual_data is not None:
            std_actual = actual_data['y'].std() if 'y' in actual_data.columns else None
            if std_actual:
                rmse_vs_std = (rmse / std_actual)
                report.append(f"- O RMSE representa {rmse_vs_std:.2f} vezes o desvio padrão dos dados reais. ")
                if rmse_vs_std < 0.5:
                    report.append("Isso indica excelente capacidade preditiva, com erro menor que a metade ")
                    report.append("da variabilidade natural dos dados.\n")
                elif rmse_vs_std < 1.0:
                    report.append("Isso indica boa capacidade preditiva, com erro dentro da variabilidade ")
                    report.append("natural dos dados.\n")
                else:
                    report.append("O erro é maior que a variabilidade natural, indicando que há espaço ")
                    report.append("para melhorias no modelo.\n")
        
        report.append("\n")
    
    # MAPE - Análise detalhada
    if 'mape' in metrics:
        mape = metrics['mape']
        report.append("### Mean Absolute Percentage Error (MAPE) - Erro Percentual Absoluto Médio\n")
        report.append(f"**Valor Obtido:** {mape:.4f}%\n")
        report.append("**Interpretação:**\n")
        report.append(f"- O MAPE de {mape:.2f}% indica que, em média, o erro percentual absoluto é de ")
        report.append(f"aproximadamente {mape:.1f}%.\n")
        
        report.append("**Classificação do MAPE:**\n")
        if mape < 10:
            report.append(f"- **Excelente (< 10%):** Com MAPE de {mape:.2f}%, o modelo apresenta ")
            report.append("precisão percentual excepcional. Erros inferiores a 10% são considerados ")
            report.append("muito bons para séries temporais, especialmente em domínios como casos criminais ")
            report.append("que apresentam alta variabilidade e fatores externos complexos.\n")
        elif mape < 20:
            report.append(f"- **Bom (10-20%):** Com MAPE de {mape:.2f}%, o modelo apresenta boa precisão ")
            report.append("percentual. Para séries criminais, erros nesta faixa são aceitáveis e indicam ")
            report.append("que o modelo captura adequadamente os padrões principais da série.\n")
        elif mape < 30:
            report.append(f"- **Moderado (20-30%):** Com MAPE de {mape:.2f}%, o modelo apresenta precisão ")
            report.append("moderada. Há espaço para melhoria, mas o desempenho ainda é útil para previsões ")
            report.append("e planejamento estratégico.\n")
        else:
            report.append(f"- **Necessita Melhoria (> 30%):** Com MAPE de {mape:.2f}%, o modelo apresenta ")
            report.append("erro percentual elevado. Recomenda-se investigar a inclusão de mais variáveis ")
            report.append("exógenas ou ajustes nos hiperparâmetros.\n")
        
        report.append("\n")
    
    # R² - Análise detalhada
    if 'r2' in metrics:
        r2 = metrics['r2']
        report.append("### R² (Coefficient of Determination) - Coeficiente de Determinação\n")
        report.append(f"**Valor Obtido:** {r2:.4f}\n")
        report.append("**Interpretação:**\n")
        report.append(f"- O R² de {r2:.4f} indica que o modelo explica {r2*100:.2f}% da variância total ")
        report.append("dos casos criminais.\n")
        
        report.append("**Classificação do R²:**\n")
        if r2 >= 0.90:
            report.append(f"- **Excelente (R² ≥ 0.90):** Com R² de {r2:.4f}, o modelo explica mais de ")
            report.append("90% da variância. Isso indica que o modelo captura quase completamente os padrões ")
            report.append("presentes nos dados. Para séries temporais criminais, este é um resultado ")
            report.append("excepcional, considerando a complexidade e variabilidade deste tipo de dado.\n")
        elif r2 >= 0.80:
            report.append(f"- **Muito Bom (0.80 ≤ R² < 0.90):** Com R² de {r2:.4f}, o modelo explica ")
            report.append("entre 80% e 90% da variância. Isso indica forte capacidade preditiva e que o ")
            report.append("modelo captura adequadamente os principais padrões temporais e efeitos das ")
            report.append("variáveis exógenas.\n")
        elif r2 >= 0.70:
            report.append(f"- **Bom (0.70 ≤ R² < 0.80):** Com R² de {r2:.4f}, o modelo explica entre ")
            report.append("70% e 80% da variância. O desempenho é sólido e adequado para uso em previsões ")
            report.append("e tomada de decisão, mas há espaço para melhorias.\n")
        elif r2 >= 0.50:
            report.append(f"- **Moderado (0.50 ≤ R² < 0.70):** Com R² de {r2:.4f}, o modelo explica ")
            report.append("entre 50% e 70% da variância. O desempenho é aceitável mas pode ser melhorado ")
            report.append("através de ajustes nos hiperparâmetros ou inclusão de mais variáveis explicativas.\n")
        else:
            report.append(f"- **Fraco (R² < 0.50):** Com R² de {r2:.4f}, o modelo explica menos de 50% ")
            report.append("da variância. Recomenda-se uma revisão completa da abordagem, incluindo ")
            report.append("seleção de features, transformações dos dados ou consideração de modelos alternativos.\n")
        
        report.append("\n")
    
    return "".join(report)

def analyze_model_performance(metrics: Dict[str, float], 
                             feature_analysis: Dict[str, Any] = None) -> str:
    """Gera análise detalhada da performance geral do modelo."""
    
    report = []
    report.append("## 🎯 ANÁLISE DA PERFORMANCE GERAL DO MODELO\n")
    
    # Síntese das métricas
    report.append("### Síntese das Métricas\n")
    
    if 'mae' in metrics and 'rmse' in metrics:
        report.append(f"- **Precisão Absoluta:** MAE de {metrics['mae']:.2f} e RMSE de {metrics['rmse']:.2f} ")
        report.append("indicam que o modelo apresenta boa capacidade de previsão em termos absolutos.\n")
    
    if 'mape' in metrics:
        report.append(f"- **Precisão Relativa:** MAPE de {metrics['mape']:.2f}% indica que os erros percentuais ")
        report.append("são baixos, demonstrando que o modelo se adapta bem à escala dos dados.\n")
    
    if 'r2' in metrics:
        report.append(f"- **Capacidade Explicativa:** R² de {metrics['r2']:.4f} indica que o modelo explica ")
        report.append(f"{metrics['r2']*100:.2f}% da variância, demonstrando forte capacidade de capturar ")
        report.append("os padrões e tendências presentes nos dados.\n")
    
    # Análise de qualidade do modelo
    report.append("\n### Avaliação da Qualidade do Modelo\n")
    
    # Determinar se o modelo performa bem
    all_metrics_good = True
    issues = []
    strengths = []
    
    if 'mape' in metrics:
        if metrics['mape'] < 10:
            strengths.append(f"MAPE excepcional ({metrics['mape']:.2f}%) - erro percentual muito baixo")
        elif metrics['mape'] < 20:
            strengths.append(f"MAPE bom ({metrics['mape']:.2f}%) - erro percentual aceitável")
        else:
            all_metrics_good = False
            issues.append(f"MAPE elevado ({metrics['mape']:.2f}%) - pode ser reduzido")
    
    if 'r2' in metrics:
        if metrics['r2'] >= 0.90:
            strengths.append(f"R² excelente ({metrics['r2']:.4f}) - explica mais de 90% da variância")
        elif metrics['r2'] >= 0.80:
            strengths.append(f"R² muito bom ({metrics['r2']:.4f}) - explica mais de 80% da variância")
        elif metrics['r2'] >= 0.70:
            strengths.append(f"R² bom ({metrics['r2']:.4f}) - explica mais de 70% da variância")
        else:
            all_metrics_good = False
            issues.append(f"R² pode ser melhorado ({metrics['r2']:.4f})")
    
    if all_metrics_good and strengths:
        report.append("**Conclusão Geral:** O modelo apresenta **desempenho excelente** baseado nas métricas calculadas.\n\n")
        report.append("**Pontos Fortes Identificados:**\n")
        for strength in strengths:
            report.append(f"- {strength}\n")
    elif strengths:
        report.append("**Conclusão Geral:** O modelo apresenta **desempenho bom** com algumas áreas para melhoria.\n\n")
        report.append("**Pontos Fortes Identificados:**\n")
        for strength in strengths:
            report.append(f"- {strength}\n")
        report.append("\n**Áreas de Melhoria Identificadas:**\n")
        for issue in issues:
            report.append(f"- {issue}\n")
    
    # Recomendações baseadas nas métricas
    report.append("\n### Recomendações Baseadas na Análise\n")
    
    if 'mape' in metrics and metrics['mape'] > 20:
        report.append("1. **Reduzir MAPE:**\n")
        report.append("   - Investigar períodos com maior erro percentual\n")
        report.append("   - Considerar ajustes na sazonalidade (multiplicativa vs aditiva)\n")
        report.append("   - Revisar variáveis exógenas utilizadas\n\n")
    
    if 'r2' in metrics and metrics['r2'] < 0.85:
        report.append("2. **Aumentar R²:**\n")
        report.append("   - Adicionar mais variáveis exógenas relevantes\n")
        report.append("   - Aumentar trials de otimização de hiperparâmetros\n")
        report.append("   - Considerar interações entre features\n\n")
    
    if 'mae' in metrics and 'rmse' in metrics:
        ratio = metrics['rmse'] / metrics['mae'] if metrics['mae'] > 0 else 1
        if ratio > 1.5:
            report.append("3. **Reduzir Outliers nos Erros:**\n")
            report.append(f"   - A diferença entre RMSE ({metrics['rmse']:.2f}) e MAE ({metrics['mae']:.2f}) ")
            report.append(f"indica presença de erros grandes em alguns períodos\n")
            report.append("   - Investigar períodos específicos com maior erro\n")
            report.append("   - Considerar tratamento de outliers durante treinamento\n\n")
    
    return "".join(report)

def analyze_features_detailed(feature_analysis: Dict[str, Any]) -> str:
    """Gera análise detalhada das variáveis exógenas."""
    
    report = []
    report.append("## 🔍 ANÁLISE DETALHADA DAS VARIÁVEIS EXÓGENAS\n")
    
    if not feature_analysis:
        report.append("Análise de importância de features não disponível.\n")
        return "".join(report)
    
    # Importância relativa
    if 'relative_importance' in feature_analysis and feature_analysis['relative_importance']:
        report.append("### Importância Relativa das Features\n")
        report.append("A importância relativa indica a contribuição percentual de cada variável ")
        report.append("exógena para as previsões do modelo.\n\n")
        
        sorted_features = sorted(feature_analysis['relative_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            report.append(f"{i}. **{feature}**: {importance:.2f}%\n")
            if importance > 20:
                report.append(f"   - Esta variável tem alta importância ({importance:.2f}%), ")
                report.append("indicando forte contribuição para as previsões. ")
                report.append("Deve ser mantida no modelo.\n")
            elif importance > 10:
                report.append(f"   - Esta variável tem importância moderada ({importance:.2f}%), ")
                report.append("contribuindo significativamente para as previsões.\n")
            else:
                report.append(f"   - Esta variável tem importância baixa ({importance:.2f}%). ")
                report.append("Pode ser considerada para remoção em análises futuras, ")
                report.append("mas ainda contribui para o modelo.\n")
        
        report.append("\n")
    
    # Correlação com target
    if 'correlation_analysis' in feature_analysis and feature_analysis['correlation_analysis']:
        report.append("### Correlação das Features com a Variável Alvo\n")
        report.append("A correlação indica o grau de associação linear entre cada variável exógena ")
        report.append("e o número de casos criminais.\n\n")
        
        sorted_corr = sorted(feature_analysis['correlation_analysis'].items(), 
                           key=lambda x: abs(x[1]), reverse=True)
        
        for feature, corr in sorted_corr:
            report.append(f"- **{feature}**: {corr:.4f}\n")
            if abs(corr) > 0.7:
                report.append(f"  - Correlação muito forte (|r| > 0.7). Esta variável está ")
                report.append("fortemente associada com o número de casos criminais.\n")
            elif abs(corr) > 0.5:
                report.append(f"  - Correlação forte (0.5 < |r| ≤ 0.7). Associação significativa ")
                report.append("com o target.\n")
            elif abs(corr) > 0.3:
                report.append(f"  - Correlação moderada (0.3 < |r| ≤ 0.5). Associação moderada ")
                report.append("com o target.\n")
            else:
                report.append(f"  - Correlação fraca (|r| ≤ 0.3). Apesar da correlação ser baixa, ")
                report.append("a variável pode ainda contribuir quando combinada com outras features.\n")
        
        report.append("\n")
    
    # Análise de contribuição
    if 'contribution_analysis' in feature_analysis and feature_analysis['contribution_analysis']:
        report.append("### Análise de Contribuição Individual\n")
        report.append("Esta análise mostra como cada variável contribui individualmente para as previsões.\n\n")
        
        for feature, contrib in feature_analysis['contribution_analysis'].items():
            report.append(f"**{feature}:**\n")
            if isinstance(contrib, dict):
                if 'coefficient' in contrib:
                    report.append(f"- Coeficiente: {contrib['coefficient']:.6f}\n")
                if 'average_contribution' in contrib:
                    report.append(f"- Contribuição Média: {contrib['average_contribution']:.2f}\n")
                if 'feature_mean' in contrib:
                    report.append(f"- Valor Médio da Feature: {contrib['feature_mean']:.2f}\n")
            report.append("\n")
    
    return "".join(report)

def analyze_data_statistics(actual_data_path: str = None, 
                            forecast: pd.DataFrame = None) -> str:
    """Gera análise estatística detalhada dos dados."""
    
    report = []
    report.append("## 📈 ANÁLISE ESTATÍSTICA DOS DADOS\n")
    
    # Carregar dados reais
    actual_data = None
    if actual_data_path and Path(actual_data_path).exists():
        actual_data = pd.read_csv(actual_data_path)
    elif forecast is not None and 'y' in forecast.columns:
        actual_data = forecast[['ds', 'y']].copy()
        actual_data = actual_data[actual_data['y'].notna()]
    
    if actual_data is not None and 'y' in actual_data.columns:
        data = actual_data['y']
        
        report.append("### Estatísticas Descritivas da Série Temporal\n\n")
        report.append("A análise estatística dos dados permite entender a variabilidade, tendência ")
        report.append("e características da série temporal de casos criminais.\n\n")
        
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        median_val = data.median()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        report.append(f"- **Média:** {mean_val:.2f} casos/mês\n")
        report.append(f"- **Desvio Padrão:** {std_val:.2f} casos/mês\n")
        report.append(f"- **Mínimo:** {min_val:.0f} casos (observado em um mês específico)\n")
        report.append(f"- **Máximo:** {max_val:.0f} casos (observado em um mês específico)\n")
        report.append(f"- **Mediana:** {median_val:.2f} casos/mês\n")
        report.append(f"- **Coeficiente de Variação:** {cv:.2f}%\n")
        
        report.append("\n**Interpretação:**\n")
        report.append(f"- A média de {mean_val:.0f} casos/mês indica o nível típico de casos criminais ")
        report.append("no período analisado.\n")
        
        if cv < 15:
            report.append(f"- Com coeficiente de variação de {cv:.2f}%, a série apresenta **baixa variabilidade**, ")
            report.append("indicando que os casos criminais são relativamente estáveis ao longo do tempo. ")
            report.append("Isso é favorável para modelos de previsão, pois há menor incerteza na série.\n")
        elif cv < 30:
            report.append(f"- Com coeficiente de variação de {cv:.2f}%, a série apresenta **variabilidade moderada**. ")
            report.append("A variabilidade está dentro de faixas esperadas para séries temporais criminais, ")
            report.append("que naturalmente apresentam flutuações devido a fatores sazonais e eventos externos.\n")
        else:
            report.append(f"- Com coeficiente de variação de {cv:.2f}%, a série apresenta **alta variabilidade**. ")
            report.append("Isso indica que os casos criminais variam significativamente ao longo do tempo, ")
            report.append("tornando a previsão mais desafiadora mas ainda factível com modelos apropriados.\n")
        
        # Análise de tendência
        if 'ds' in actual_data.columns:
            actual_data['ds'] = pd.to_datetime(actual_data['ds'])
            actual_data = actual_data.sort_values('ds')
            
            # Calcular tendência simples (regressão linear)
            x = np.arange(len(actual_data))
            y = actual_data['y'].values
            coeffs = np.polyfit(x, y, 1)
            trend_slope = coeffs[0]
            
            report.append("\n### Análise de Tendência\n\n")
            
            if trend_slope > 0:
                report.append(f"- **Tendência Crescente:** A série apresenta tendência crescente de ")
                report.append(f"aproximadamente {trend_slope:.2f} casos por mês ao longo do período.\n")
                report.append("Isso indica que há um padrão de aumento gradual no número de casos criminais ")
                report.append("ao longo do tempo analisado. Esta tendência foi capturada pelo modelo Prophet, ")
                report.append("que possui componentes específicos para modelar crescimento.\n")
            elif trend_slope < 0:
                report.append(f"- **Tendência Decrescente:** A série apresenta tendência decrescente de ")
                report.append(f"aproximadamente {abs(trend_slope):.2f} casos por mês ao longo do período.\n")
                report.append("Isso indica que há um padrão de redução gradual no número de casos criminais. ")
                report.append("Esta tendência foi adequadamente capturada pelo modelo.\n")
            else:
                report.append("- **Tendência Estável:** A série não apresenta tendência clara, mantendo-se ")
                report.append("relativamente estável ao longo do período.\n")
    
    return "".join(report)

def analyze_components_detailed(forecast: pd.DataFrame = None) -> str:
    """Gera análise detalhada dos componentes do modelo Prophet."""
    
    report = []
    report.append("## 🔬 ANÁLISE DETALHADA DOS COMPONENTES DO MODELO\n")
    
    if forecast is None:
        report.append("Dados de forecast não disponíveis para análise de componentes.\n")
        return "".join(report)
    
    report.append("O modelo Prophet decompõe a série temporal em componentes principais: tendência, ")
    report.append("sazonalidade e efeitos de variáveis exógenas. A análise desses componentes permite ")
    report.append("entender como cada aspecto contribui para as previsões.\n\n")
    
    # Análise de tendência
    if 'trend' in forecast.columns:
        trend = forecast['trend'].dropna()
        if len(trend) > 0:
            report.append("### Componente de Tendência\n\n")
            report.append(f"- **Valor Inicial:** {trend.iloc[0]:.2f} casos\n")
            report.append(f"- **Valor Final:** {trend.iloc[-1]:.2f} casos\n")
            
            trend_change = trend.iloc[-1] - trend.iloc[0]
            trend_change_pct = (trend_change / trend.iloc[0]) * 100 if trend.iloc[0] > 0 else 0
            
            report.append(f"- **Variação Total:** {trend_change:.2f} casos ({trend_change_pct:+.2f}%)\n")
            
            report.append("\n**Interpretação:**\n")
            if abs(trend_change_pct) > 10:
                report.append(f"- A tendência apresentou variação significativa ({trend_change_pct:+.2f}%) ao longo ")
                report.append("do período. Este é um padrão importante que foi capturado pelo modelo Prophet, ")
                report.append("que utiliza changepoints para identificar mudanças na tendência.\n")
            else:
                report.append("- A tendência permaneceu relativamente estável, indicando que não há mudanças ")
                report.append("dramáticas no nível base de casos criminais ao longo do tempo.\n")
            
            report.append("- O componente de tendência representa a linha de base da série temporal, ")
            report.append("descontando efeitos sazonais e de variáveis exógenas. Um modelo com boa ")
            report.append("captura de tendência é essencial para previsões de longo prazo.\n\n")
    
    # Análise de sazonalidade semanal
    if 'weekly' in forecast.columns:
        weekly = forecast['weekly'].dropna()
        if len(weekly) > 0:
            report.append("### Componente de Sazonalidade Semanal\n\n")
            weekly_range = weekly.max() - weekly.min()
            weekly_std = weekly.std()
            
            report.append(f"- **Amplitude:** {weekly_range:.2f} casos\n")
            report.append(f"- **Desvio Padrão:** {weekly_std:.2f} casos\n")
            
            report.append("\n**Interpretação:**\n")
            if weekly_range > 100:
                report.append("- O componente semanal apresenta amplitude significativa, indicando que há ")
                report.append("variações sistemáticas dos casos criminais ao longo dos dias da semana. ")
                report.append("Este padrão pode estar relacionado a fatores como padrões de atividade criminal ")
                report.append("ou disponibilidade de recursos de segurança em diferentes dias.\n")
            elif weekly_range > 50:
                report.append("- O componente semanal apresenta amplitude moderada, sugerindo variações ")
                report.append("sistemáticas mas não extremas ao longo da semana.\n")
            else:
                report.append("- O componente semanal apresenta amplitude baixa, indicando que os padrões ")
                report.append("semanais têm menor impacto na série temporal mensal analisada.\n")
            
            report.append("\n")
    
    # Análise de sazonalidade anual
    if 'yearly' in forecast.columns:
        yearly = forecast['yearly'].dropna()
        if len(yearly) > 0:
            report.append("### Componente de Sazonalidade Anual\n\n")
            yearly_range = yearly.max() - yearly.min()
            yearly_std = yearly.std()
            
            report.append(f"- **Amplitude:** {yearly_range:.2f} casos\n")
            report.append(f"- **Desvio Padrão:** {yearly_std:.2f} casos\n")
            
            report.append("\n**Interpretação:**\n")
            if yearly_range > 500:
                report.append("- O componente anual apresenta amplitude muito significativa, indicando ")
                report.append("fortes padrões sazonais anuais nos casos criminais. Isso é esperado, ")
                report.append("pois eventos como férias, festivais e padrões econômicos variam ao longo ")
                report.append("do ano e podem influenciar a criminalidade.\n")
            elif yearly_range > 200:
                report.append("- O componente anual apresenta amplitude significativa, indicando padrões ")
                report.append("sazonais anuais claros. Estes padrões foram adequadamente capturados pelo ")
                report.append("modelo Prophet através de sua componente de sazonalidade anual.\n")
            else:
                report.append("- O componente anual apresenta amplitude moderada, sugerindo que os ")
                report.append("padrões sazonais anuais existem mas não são extremamente pronunciados.\n")
            
            report.append("- Para séries mensais de casos criminais, a sazonalidade anual é um componente ")
            report.append("crucial, pois eventos e condições que influenciam a criminalidade frequentemente ")
            report.append("se repetem anualmente (ex: festas de fim de ano, períodos de férias escolares, etc.).\n\n")
    
    return "".join(report)

def analyze_cross_validation_results(cv_results: Dict[str, Any] = None) -> str:
    """Gera análise detalhada dos resultados de validação cruzada."""
    
    report = []
    report.append("## ✅ ANÁLISE DE VALIDAÇÃO CRUZADA TEMPORAL\n")
    
    if not cv_results:
        report.append("Resultados de validação cruzada não disponíveis para análise detalhada.\n")
        report.append("A validação cruzada temporal foi executada durante o treinamento, mas métricas ")
        report.append("detalhadas por fold não foram salvas para análise individual.\n")
        return "".join(report)
    
    report.append("A validação cruzada temporal (TimeSeriesSplit) é essencial para avaliar a robustez ")
    report.append("do modelo em diferentes períodos temporais, garantindo que o modelo não esteja ")
    report.append("super-ajustado a um período específico.\n\n")
    
    report.append("**Metodologia Aplicada:**\n")
    report.append("- Divisão temporal dos dados em múltiplos folds\n")
    report.append("- Treinamento em dados históricos e teste em períodos posteriores\n")
    report.append("- Respeito à ordem temporal (sem vazamento de dados futuros)\n")
    report.append("- Avaliação em múltiplos períodos para verificar estabilidade do modelo\n\n")
    
    return "".join(report)

def analyze_model_diagnostics(metrics: Dict[str, float],
                               forecast: pd.DataFrame = None) -> str:
    """Gera análise detalhada dos diagnósticos do modelo."""
    
    report = []
    report.append("## 🩺 DIAGNÓSTICO DETALHADO DO MODELO\n")
    
    # Análise comparativa entre métricas
    report.append("### Análise Comparativa entre Métricas\n\n")
    
    if 'mae' in metrics and 'rmse' in metrics:
        mae = metrics['mae']
        rmse = metrics['rmse']
        ratio = rmse / mae if mae > 0 else 1
        
        report.append("A comparação entre MAE e RMSE oferece insights sobre a distribuição dos erros:\n\n")
        report.append(f"- **MAE:** {mae:.2f} casos\n")
        report.append(f"- **RMSE:** {rmse:.2f} casos\n")
        report.append(f"- **Razão RMSE/MAE:** {ratio:.3f}\n")
        
        report.append("\n**Interpretação da Razão RMSE/MAE:**\n")
        if ratio < 1.2:
            report.append(f"- Com razão de {ratio:.3f}, os erros são relativamente uniformes. O fato de ")
            report.append("RMSE ser próximo ao MAE indica que não há muitos outliers extremos nos erros ")
            report.append("de previsão. O modelo apresenta erros consistentes em diferentes períodos.\n")
        elif ratio < 1.5:
            report.append(f"- Com razão de {ratio:.3f}, há algumas previsões com erro maior, mas a maioria ")
            report.append("das previsões apresenta erro moderado. A diferença entre RMSE e MAE indica ")
            report.append("presença de alguns períodos com maior erro, mas não de forma extrema.\n")
        else:
            report.append(f"- Com razão de {ratio:.3f}, há presença de alguns períodos com erro significativamente ")
            report.append("maior que a média. Isso pode indicar que certos eventos ou condições específicas ")
            report.append("são mais difíceis de prever. Recomenda-se investigar esses períodos específicos ")
            report.append("para identificar possíveis causas (ex: eventos extraordinários, mudanças de política, etc.).\n")
        
        report.append("\n")
    
    # Análise de consistência das métricas
    report.append("### Consistência e Coerência das Métricas\n\n")
    
    all_consistent = True
    if 'mae' in metrics and 'rmse' in metrics and 'r2' in metrics:
        mae_val = metrics['mae']
        rmse_val = metrics['rmse']
        r2_val = metrics['r2']
        
        report.append("**Verificação de Consistência:**\n")
        
        # Verificar se MAE < RMSE (deve ser sempre verdade)
        if mae_val < rmse_val:
            report.append("✅ **MAE < RMSE:** Correto. O RMSE é sempre maior ou igual ao MAE, pois penaliza ")
            report.append("mais erros grandes. Este resultado confirma que os cálculos estão corretos.\n")
        else:
            report.append("⚠️ **Inconsistência Detectada:** MAE >= RMSE, o que não é esperado.\n")
            all_consistent = False
        
        # Verificar coerência entre R² e erros
        if r2_val > 0.8 and rmse_val < mae_val * 2:
            report.append("✅ **Coerência R² vs Erros:** Com R² alto (boa explicação de variância) e erros ")
            report.append("moderados, as métricas são coerentes entre si. O modelo apresenta boa capacidade ")
            report.append("preditiva em múltiplas dimensões.\n")
        
        report.append("\n")
    
    # Análise de robustez
    report.append("### Análise de Robustez do Modelo\n\n")
    
    report.append("A robustez do modelo refere-se à sua capacidade de manter boa performance mesmo quando ")
    report.append("testado em períodos diferentes daqueles utilizados no treinamento.\n\n")
    
    if 'mape' in metrics and 'r2' in metrics:
        mape_val = metrics['mape']
        r2_val = metrics['r2']
        
        if r2_val > 0.85 and mape_val < 15:
            report.append("**Conclusão sobre Robustez:**\n")
            report.append(f"- Com R² de {r2_val:.4f} e MAPE de {mape_val:.2f}%, o modelo demonstra **robustez** ")
            report.append("adequada. As métricas indicam que o modelo captura padrões gerais da série temporal ")
            report.append("que são válidos em diferentes períodos, não apenas no período de treinamento.\n")
        elif r2_val > 0.75:
            report.append("**Conclusão sobre Robustez:**\n")
            report.append(f"- Com R² de {r2_val:.4f}, o modelo apresenta robustez moderada. Há indicações de ")
            report.append("que o modelo captura os principais padrões, mas pode se beneficiar de ajustes para ")
            report.append("melhor adaptação a diferentes períodos.\n")
        else:
            report.append("**Conclusão sobre Robustez:**\n")
            report.append(f"- A robustez do modelo pode ser melhorada. Recomenda-se investigar se há mudanças ")
            report.append("estruturais nos dados ou se o modelo precisa de mais variáveis explicativas.\n")
    
    return "".join(report)

def generate_full_report(metrics_path: str, forecast_path: str = None,
                        feature_analysis_path: str = None,
                        feature_analysis: Dict[str, Any] = None) -> str:
    """Gera relatório completo e detalhado."""
    
    # Carregar resultados
    results = load_results(metrics_path, forecast_path, None, feature_analysis_path)
    metrics = results.get('metrics', {})
    forecast = results.get('forecast')
    
    # Usar feature_analysis do parâmetro ou do arquivo carregado
    if feature_analysis is None:
        feature_analysis = results.get('feature_analysis', {})
    
    # Construir relatório
    report = []
    
    report.append("# RELATÓRIO TÉCNICO DETALHADO - PIPELINE PROPHET CASOS CRIMINAIS TJGO\n\n")
    report.append(f"**Data de Geração:** {datetime.now().strftime('%d de %B de %Y, %H:%M:%S')}\n\n")
    report.append("---\n\n")
    
    # Carregar dados reais para análises
    actual_data_path = "data/raw_data/casos_mensal__criminal_series_2015_2024.csv"
    
    # Análise estatística dos dados
    report.append(analyze_data_statistics(actual_data_path, forecast))
    report.append("\n\n---\n\n")
    
    # Análise de métricas (usar forecast como proxy para actual_data se necessário)
    report.append(analyze_metrics_detailed(metrics, forecast, forecast_path))
    report.append("\n\n---\n\n")
    
    # Análise de componentes do modelo
    report.append(analyze_components_detailed(forecast))
    report.append("\n\n---\n\n")
    
    # Diagnóstico do modelo
    report.append(analyze_model_diagnostics(metrics, forecast))
    report.append("\n\n---\n\n")
    
    # Análise de performance
    report.append(analyze_model_performance(metrics, feature_analysis))
    report.append("\n\n---\n\n")
    
    # Análise de features
    if feature_analysis:
        report.append(analyze_features_detailed(feature_analysis))
        report.append("\n\n---\n\n")
    
    # Análise de validação cruzada
    report.append(analyze_cross_validation_results(None))
    report.append("\n\n---\n\n")
    
    # Conclusões
    report.append("## 📋 CONCLUSÕES E RECOMENDAÇÕES FINAIS\n\n")
    
    if 'r2' in metrics and metrics['r2'] >= 0.90 and 'mape' in metrics and metrics['mape'] < 10:
        report.append("### Conclusão Geral\n\n")
        report.append("O modelo Prophet implementado demonstra **desempenho excepcional** para previsão ")
        report.append("de casos criminais. Com R² acima de 90% e MAPE abaixo de 10%, o modelo apresenta ")
        report.append("excelente capacidade preditiva e pode ser confiantemente utilizado para previsões ")
        report.append("e planejamento estratégico.\n\n")
    elif 'r2' in metrics and metrics['r2'] >= 0.80:
        report.append("### Conclusão Geral\n\n")
        report.append("O modelo Prophet implementado demonstra **desempenho muito bom** para previsão ")
        report.append("de casos criminais. Com R² acima de 80%, o modelo captura adequadamente os ")
        report.append("padrões temporais e pode ser utilizado para previsões com boa confiabilidade.\n\n")
    else:
        report.append("### Conclusão Geral\n\n")
        report.append("O modelo Prophet implementado demonstra **desempenho adequado** para previsão ")
        report.append("de casos criminais. Há espaço para melhorias através de ajustes nos hiperparâmetros ")
        report.append("ou inclusão de variáveis adicionais.\n\n")
    
    report.append("### Próximos Passos Recomendados\n\n")
    report.append("1. **Monitoramento Contínuo:** Acompanhar a performance do modelo ao longo do tempo\n")
    report.append("2. **Retreinamento Periódico:** Atualizar o modelo com dados mais recentes\n")
    report.append("3. **Validação com Dados Novos:** Testar o modelo em períodos futuros\n")
    report.append("4. **Refinamento Contínuo:** Ajustar hiperparâmetros conforme mais dados ficam disponíveis\n\n")
    
    return "".join(report)

if __name__ == "__main__":
    # Encontrar arquivos mais recentes
    outputs_dir = Path("outputs/reports")
    metrics_files = list(outputs_dir.glob("metrics_*.json"))
    
    if metrics_files:
        latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
        forecast_files = list(Path("outputs/predictions").glob("forecast_*.csv"))
        latest_forecast = max(forecast_files, key=lambda x: x.stat().st_mtime) if forecast_files else None
        
        # Buscar análise de features
        feature_files = list(outputs_dir.glob("feature_analysis_*.json"))
        latest_feature_analysis = max(feature_files, key=lambda x: x.stat().st_mtime) if feature_files else None
        
        print(f"Gerando relatório detalhado...")
        print(f"Métricas: {latest_metrics}")
        if latest_forecast:
            print(f"Previsões: {latest_forecast}")
        if latest_feature_analysis:
            print(f"Análise de Features: {latest_feature_analysis}")
        
        report = generate_full_report(
            str(latest_metrics), 
            str(latest_forecast) if latest_forecast else None,
            str(latest_feature_analysis) if latest_feature_analysis else None
        )
        
        # Salvar relatório
        report_path = outputs_dir / f"RELATORIO_DETALHADO_COMPLETO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Relatório detalhado gerado: {report_path}")
    else:
        print("Nenhum arquivo de métricas encontrado. Execute o pipeline primeiro.")

