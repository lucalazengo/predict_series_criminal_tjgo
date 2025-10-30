#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISP-DM Fase 2: Data Understanding
Análise exploratória profunda dos dados para fundamentar decisões de modelagem SARIMAX
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SARIMAXDataExplorer:
    """
    Classe para análise exploratória completa dos dados seguindo CRISP-DM.
    Fundamenta todas as decisões de modelagem com evidências estatísticas.
    """
    
    def __init__(self, target_path: str, exogenous_path: str):
        """
        Inicializa o explorador de dados.
        
        Parameters:
        -----------
        target_path : str
            Caminho para arquivo CSV da série temporal alvo
        exogenous_path : str
            Caminho para arquivo CSV das variáveis exógenas
        """
        self.target_path = target_path
        self.exogenous_path = exogenous_path
        self.target_df = None
        self.exogenous_df = None
        self.merged_df = None
        self.exploration_results = {}
        
    def load_data(self):
        """Carrega os dados brutos."""
        print("="*70)
        print("CARREGANDO DADOS")
        print("="*70)
        
        # Carrega série alvo
        self.target_df = pd.read_csv(self.target_path)
        self.target_df['DATA'] = pd.to_datetime(self.target_df['DATA'])
        self.target_df = self.target_df.sort_values('DATA').reset_index(drop=True)
        
        # Carrega variáveis exógenas
        self.exogenous_df = pd.read_csv(self.exogenous_path)
        self.exogenous_df['data'] = pd.to_datetime(self.exogenous_df['data'])
        self.exogenous_df = self.exogenous_df.sort_values('data').reset_index(drop=True)
        
        # Merge dos datasets
        self.merged_df = pd.merge(
            self.target_df,
            self.exogenous_df,
            left_on='DATA',
            right_on='data',
            how='inner'
        ).drop(columns=['data']).reset_index(drop=True)
        
        print(f"\nSérie Temporal Alvo:")
        print(f"  - Período: {self.target_df['DATA'].min()} a {self.target_df['DATA'].max()}")
        print(f"  - Total de observações: {len(self.target_df)} meses")
        print(f"  - Frequência: Mensal")
        
        print(f"\nVariáveis Exógenas:")
        print(f"  - Total de features: {len(self.exogenous_df.columns) - 1}")
        print(f"  - Período: {self.exogenous_df['data'].min()} a {self.exogenous_df['data'].max()}")
        
        print(f"\nDataset Merged:")
        print(f"  - Shape: {self.merged_df.shape}")
        print(f"  - Colunas: {list(self.merged_df.columns)}")
        
        return self
    
    def basic_statistics(self):
        """Análise estatística básica da série alvo."""
        print("\n" + "="*70)
        print("ESTATÍSTICAS BÁSICAS - TOTAL_CASOS")
        print("="*70)
        
        target_col = 'TOTAL_CASOS'
        series = self.merged_df[target_col]
        
        stats = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            '25%': series.quantile(0.25),
            '50%': series.quantile(0.50),
            'median': series.median(),
            '75%': series.quantile(0.75),
            'max': series.max(),
            'range': series.max() - series.min(),
            'cv': (series.std() / series.mean()) * 100,  # Coeficiente de variação
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        self.exploration_results['basic_stats'] = stats
        
        print(f"\nEstatísticas Descritivas:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:12s}: {value:10.2f}")
            else:
                print(f"  {key:12s}: {value:10}")
        
        # Interpretação
        print(f"\nInterpretações:")
        print(f"  - Média: {stats['mean']:.0f} casos/mês")
        print(f"  - Variabilidade (CV): {stats['cv']:.1f}% ({'Alta' if stats['cv'] > 20 else 'Moderada' if stats['cv'] > 10 else 'Baixa'} variabilidade)")
        print(f"  - Assimetria: {stats['skewness']:.2f} ({'Assimétrica direita' if stats['skewness'] > 0.5 else 'Assimétrica esquerda' if stats['skewness'] < -0.5 else 'Simétrica'})")
        print(f"  - Curtose: {stats['kurtosis']:.2f} ({'Leptocúrtica (caudas pesadas)' if stats['kurtosis'] > 0 else 'Platicúrtica (caudas leves)'})")
        
        return stats
    
    def missing_values_analysis(self):
        """Analisa valores faltantes."""
        print("\n" + "="*70)
        print("ANÁLISE DE VALORES FALTANTES")
        print("="*70)
        
        missing = self.merged_df.isnull().sum()
        missing_pct = (missing / len(self.merged_df)) * 100
        
        missing_df = pd.DataFrame({
            'Coluna': missing.index,
            'Valores_Faltantes': missing.values,
            'Percentual': missing_pct.values
        }).sort_values('Valores_Faltantes', ascending=False)
        
        missing_df = missing_df[missing_df['Valores_Faltantes'] > 0]
        
        if len(missing_df) > 0:
            print("\nColunas com valores faltantes:")
            print(missing_df.to_string(index=False))
            self.exploration_results['missing_values'] = missing_df.to_dict('records')
        else:
            print("\n✓ Nenhum valor faltante encontrado!")
            self.exploration_results['missing_values'] = []
        
        return missing_df
    
    def stationarity_tests(self):
        """
        Testa estacionariedade da série usando ADF e KPSS.
        Fundamenta decisão sobre necessidade de diferenciação.
        """
        print("\n" + "="*70)
        print("TESTES DE ESTACIONARIEDADE")
        print("="*70)
        
        target_col = 'TOTAL_CASOS'
        series = self.merged_df.set_index('DATA')[target_col]
        
        # Teste ADF (Augmented Dickey-Fuller)
        print("\n1. Teste de Dickey-Fuller Aumentado (ADF):")
        adf_result = adfuller(series, autolag='AIC')
        adf_stat, adf_pvalue = adf_result[0], adf_result[1]
        
        print(f"   Estatística ADF: {adf_stat:.4f}")
        print(f"   p-value: {adf_pvalue:.4f}")
        
        if adf_pvalue <= 0.05:
            print("   ✓ Série é ESTACIONÁRIA (rejeita H0)")
            is_stationary_adf = True
        else:
            print("   ✗ Série NÃO é estacionária (não rejeita H0)")
            print("   → Necessária diferenciação para SARIMAX")
            is_stationary_adf = False
        
        # Teste KPSS
        print("\n2. Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
        try:
            kpss_result = kpss(series, regression='ct', nlags='auto')
            kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
            
            print(f"   Estatística KPSS: {kpss_stat:.4f}")
            print(f"   p-value: {kpss_pvalue:.4f}")
            
            if kpss_pvalue >= 0.05:
                print("   ✓ Série é ESTACIONÁRIA (não rejeita H0)")
                is_stationary_kpss = True
            else:
                print("   ✗ Série NÃO é estacionária (rejeita H0)")
                print("   → Necessária diferenciação para SARIMAX")
                is_stationary_kpss = False
        except Exception as e:
            print(f"   Erro no teste KPSS: {e}")
            is_stationary_kpss = None
            kpss_stat, kpss_pvalue = None, None
        
        results = {
            'adf': {
                'statistic': adf_stat,
                'pvalue': adf_pvalue,
                'is_stationary': is_stationary_adf
            },
            'kpss': {
                'statistic': kpss_stat,
                'pvalue': kpss_pvalue,
                'is_stationary': is_stationary_kpss if kpss_stat else None
            }
        }
        
        # Decisão de diferenciação
        if not is_stationary_adf or (is_stationary_kpss is False):
            print("\n→ DECISÃO: Série requer diferenciação")
            print("  d=1 será testado no auto_arima")
        else:
            print("\n→ DECISÃO: Série pode ser estacionária")
            print("  d=0 será testado primeiro no auto_arima")
        
        self.exploration_results['stationarity'] = results
        
        return results
    
    def seasonal_decomposition(self):
        """
        Decomposição sazonal da série.
        Identifica padrões sazonais e tendência.
        """
        print("\n" + "="*70)
        print("DECOMPOSIÇÃO SAZONAL")
        print("="*70)
        
        target_col = 'TOTAL_CASOS'
        series = self.merged_df.set_index('DATA')[target_col]
        
        # Decomposição aditiva
        print("\nDecomposição Aditiva (model='additive'):")
        decomposition_add = seasonal_decompose(series, model='additive', period=12)
        
        trend = decomposition_add.trend.dropna()
        seasonal = decomposition_add.seasonal.dropna()
        residual = decomposition_add.resid.dropna()
        
        print(f"  Tendência - Média: {trend.mean():.2f}, Std: {trend.std():.2f}")
        print(f"  Sazonalidade - Amplitude: {seasonal.max() - seasonal.min():.2f}")
        print(f"  Resíduos - Média: {residual.mean():.2f}, Std: {residual.std():.2f}")
        
        # Verifica se sazonalidade é significativa
        seasonal_strength = np.var(seasonal) / (np.var(seasonal) + np.var(residual))
        print(f"\n  Força da Sazonalidade: {seasonal_strength:.3f}")
        if seasonal_strength > 0.64:
            print("  → Sazonalidade FORTE (seasonal_strength > 0.64)")
            print("  → Recomenda-se SARIMA com componente sazonal (S > 0)")
        elif seasonal_strength > 0.36:
            print("  → Sazonalidade MODERADA (0.36 < seasonal_strength < 0.64)")
            print("  → Testar SARIMA com componente sazonal")
        else:
            print("  → Sazonalidade FRACA (seasonal_strength < 0.36)")
            print("  → Pode-se considerar ARIMA simples (S=0)")
        
        # Decomposição multiplicativa
        print("\nDecomposição Multiplicativa (model='multiplicative'):")
        try:
            decomposition_mult = seasonal_decompose(series, model='multiplicative', period=12)
            
            trend_mult = decomposition_mult.trend.dropna()
            seasonal_mult = decomposition_mult.seasonal.dropna()
            residual_mult = decomposition_mult.resid.dropna()
            
            seasonal_strength_mult = np.var(seasonal_mult) / (np.var(seasonal_mult) + np.var(residual_mult))
            print(f"  Força da Sazonalidade: {seasonal_strength_mult:.3f}")
            
            # Compara modelos
            print("\nComparação de Modelos:")
            print(f"  Aditiva - Resíduos Std: {residual.std():.2f}")
            print(f"  Multiplicativa - Resíduos Std: {residual_mult.std():.2f}")
            
            if residual_mult.std() < residual.std():
                print("  → Modelo Multiplicativo parece mais adequado")
                model_type = 'multiplicative'
            else:
                print("  → Modelo Aditivo parece mais adequado")
                model_type = 'additive'
        except Exception as e:
            print(f"  Erro na decomposição multiplicativa: {e}")
            model_type = 'additive'
        
        results = {
            'seasonal_strength_add': seasonal_strength,
            'seasonal_strength_mult': seasonal_strength_mult if 'seasonal_strength_mult' in locals() else None,
            'recommended_model': model_type,
            'trend_mean': trend.mean(),
            'trend_std': trend.std(),
            'seasonal_amplitude': seasonal.max() - seasonal.min()
        }
        
        print(f"\n→ DECISÃO: Considerar modelo {model_type.upper()} no SARIMAX")
        
        self.exploration_results['decomposition'] = results
        
        return results
    
    def autocorrelation_analysis(self):
        """
        Análise de autocorrelação (ACF) e autocorrelação parcial (PACF).
        Ajuda a identificar ordens p e q do modelo ARIMA.
        """
        print("\n" + "="*70)
        print("ANÁLISE DE AUTOCORRELAÇÃO")
        print("="*70)
        
        from statsmodels.tsa.stattools import acf, pacf
        
        target_col = 'TOTAL_CASOS'
        series = self.merged_df.set_index('DATA')[target_col]
        
        # ACF e PACF
        nlags = min(24, len(series) // 3)  # Máximo 24 lags ou 1/3 dos dados
        acf_values = acf(series, nlags=nlags, fft=True)
        pacf_values = pacf(series, nlags=nlags, method='ols')
        
        # Identifica lags significativos
        # Usando intervalo de confiança aproximado: ±1.96/sqrt(n)
        conf_interval = 1.96 / np.sqrt(len(series))
        
        significant_acf_lags = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
        significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1
        
        print(f"\nLags significativos (|correlação| > {conf_interval:.3f}):")
        print(f"  ACF: {significant_acf_lags[:10].tolist()}")  # Primeiros 10
        print(f"  PACF: {significant_pacf_lags[:10].tolist()}")
        
        # Padrões típicos
        print(f"\nInterpretação:")
        if len(significant_pacf_lags) > 0 and significant_pacf_lags[0] <= 3:
            print(f"  → Padrão AR: p pode estar entre 1-{min(3, significant_pacf_lags[0])}")
        else:
            print(f"  → Padrão AR: p a determinar (usar auto_arima)")
        
        if len(significant_acf_lags) > 0:
            if significant_acf_lags[0] == 12 or 12 in significant_acf_lags[:5]:
                print(f"  → Sazonalidade anual detectada (lag 12)")
                print(f"  → Recomenda-se componente sazonal S=12 no SARIMA")
            print(f"  → Padrão MA: q a determinar (usar auto_arima)")
        
        results = {
            'acf_lags': significant_acf_lags[:10].tolist(),
            'pacf_lags': significant_pacf_lags[:10].tolist(),
            'seasonal_period': 12 if (12 in significant_acf_lags or 12 in significant_pacf_lags) else None
        }
        
        self.exploration_results['autocorrelation'] = results
        
        return results
    
    def exogenous_features_analysis(self):
        """
        Analisa variáveis exógenas.
        Verifica correlação e seleciona features mais relevantes.
        """
        print("\n" + "="*70)
        print("ANÁLISE DE VARIÁVEIS EXÓGENAS")
        print("="*70)
        
        target_col = 'TOTAL_CASOS'
        
        # Seleciona apenas colunas numéricas exógenas
        exclude_cols = ['DATA', 'ANO', 'MES', 'MES_NOME', target_col]
        exogenous_cols = [col for col in self.merged_df.columns if col not in exclude_cols]
        
        # Calcula correlações
        correlations = {}
        for col in exogenous_cols:
            if self.merged_df[col].dtype in [np.float64, np.int64]:
                corr = self.merged_df[target_col].corr(self.merged_df[col])
                correlations[col] = corr
        
        # Ordena por correlação absoluta
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 10 Variáveis Exógenas por Correlação com {target_col}:")
        print(f"{'Variável':<40} {'Correlação':>12}")
        print("-" * 55)
        for var, corr in sorted_corrs[:10]:
            print(f"{var:<40} {corr:>12.4f}")
        
        # Features recomendadas (correlação > 0.3 ou < -0.3)
        recommended = [(var, corr) for var, corr in sorted_corrs if abs(corr) > 0.3]
        
        print(f"\nVariáveis com |correlação| > 0.3 (recomendadas para SARIMAX):")
        for var, corr in recommended:
            print(f"  - {var}: {corr:.4f}")
        
        if len(recommended) == 0:
            print("  Nenhuma variável exógena com correlação > 0.3 encontrada")
            print("  → Considerar modelo ARIMA simples ou revisar features")
        
        results = {
            'correlations': dict(sorted_corrs),
            'recommended_features': [var for var, _ in recommended],
            'top_features': [var for var, _ in sorted_corrs[:10]]
        }
        
        self.exploration_results['exogenous'] = results
        
        return results
    
    def generate_report(self, output_path: str = "sarimax/data_exploration_report.md"):
        """Gera relatório completo da análise exploratória."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = ["# RELATÓRIO DE ANÁLISE EXPLORATÓRIA - SARIMAX\n"]
        report.append("## CRISP-DM Fase 2: Data Understanding\n")
        report.append(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Estatísticas básicas
        if 'basic_stats' in self.exploration_results:
            stats = self.exploration_results['basic_stats']
            report.append("### 1. Estatísticas Básicas\n")
            report.append("| Métrica | Valor |\n|---------|-------|")
            for key, value in stats.items():
                if isinstance(value, float):
                    report.append(f"| {key} | {value:.2f} |")
                else:
                    report.append(f"| {key} | {value} |")
            report.append("")
        
        # Estacionariedade
        if 'stationarity' in self.exploration_results:
            report.append("### 2. Testes de Estacionariedade\n")
            st = self.exploration_results['stationarity']
            report.append(f"- **ADF Test:** p-value = {st['adf']['pvalue']:.4f} ({'Estacionária' if st['adf']['is_stationary'] else 'Não-estacionária'})")
            kpss_pvalue = st['kpss']['pvalue'] if st['kpss']['pvalue'] is not None else None
            if kpss_pvalue is not None:
                report.append(f"- **KPSS Test:** p-value = {kpss_pvalue:.4f}")
            else:
                report.append("- **KPSS Test:** p-value = N/A")
            report.append("")
        
        # Decomposição
        if 'decomposition' in self.exploration_results:
            report.append("### 3. Decomposição Sazonal\n")
            dec = self.exploration_results['decomposition']
            report.append(f"- **Força da Sazonalidade:** {dec['seasonal_strength_add']:.3f}")
            report.append(f"- **Modelo Recomendado:** {dec['recommended_model'].upper()}")
            report.append("")
        
        # Autocorrelação
        if 'autocorrelation' in self.exploration_results:
            report.append("### 4. Análise de Autocorrelação\n")
            ac = self.exploration_results['autocorrelation']
            report.append(f"- **Período Sazonal Detectado:** {ac['seasonal_period'] if ac['seasonal_period'] else 'Nenhum'}")
            report.append("")
        
        # Variáveis exógenas
        if 'exogenous' in self.exploration_results:
            report.append("### 5. Variáveis Exógenas Recomendadas\n")
            ex = self.exploration_results['exogenous']
            for var in ex['recommended_features']:
                report.append(f"- {var}")
            report.append("")
        
        # Recomendações finais
        report.append("## Recomendações para Modelagem SARIMAX\n")
        report.append("### Configuração do auto_arima:\n")
        report.append("```python\n")
        report.append("from pmdarima import auto_arima\n")
        report.append("\n# Baseado na análise:\n")
        
        # Sazonalidade
        if 'autocorrelation' in self.exploration_results and self.exploration_results['autocorrelation'].get('seasonal_period'):
            report.append("seasonal = True  # Sazonalidade detectada\n")
            report.append("seasonal_periods = 12  # Mensal\n")
        else:
            report.append("seasonal = True  # Testar ambos\n")
            report.append("seasonal_periods = 12  # Tentar período 12\n")
        
        # Diferenciação
        if 'stationarity' in self.exploration_results:
            if not self.exploration_results['stationarity']['adf']['is_stationary']:
                report.append("d = 1  # Série não-estacionária\n")
            else:
                report.append("# d será determinado automaticamente\n")
        
        # Variáveis exógenas
        if 'exogenous' in self.exploration_results:
            features = self.exploration_results['exogenous']['recommended_features']
            if features:
                report.append(f"exogenous = df[{features}]  # Features recomendadas\n")
            else:
                report.append("# Revisar seleção de features exógenas\n")
        
        report.append("```\n")
        
        # Salva relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Relatório salvo em: {output_path}")
        
        return output_path
    
    def run_full_analysis(self):
        """Executa análise exploratória completa."""
        print("="*70)
        print("ANÁLISE EXPLORATÓRIA COMPLETA - SARIMAX")
        print("CRISP-DM Fase 2: Data Understanding")
        print("="*70)
        
        self.load_data()
        self.basic_statistics()
        self.missing_values_analysis()
        self.stationarity_tests()
        self.seasonal_decomposition()
        self.autocorrelation_analysis()
        self.exogenous_features_analysis()
        
        # Gera relatório
        report_path = self.generate_report()
        
        print("\n" + "="*70)
        print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
        print("="*70)
        print(f"\nRelatório salvo em: {report_path}")
        
        return self.exploration_results


if __name__ == "__main__":
    explorer = SARIMAXDataExplorer(
        target_path="data/raw_data/casos_mensal__criminal_series_2015_2024.csv",
        exogenous_path="data/raw_data/external_features_2015_2024.csv"
    )
    
    results = explorer.run_full_analysis()

