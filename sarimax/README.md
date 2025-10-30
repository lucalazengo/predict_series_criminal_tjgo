# SARIMAX Forecasting Pipeline - CRISP-DM

## 📋 Visão Geral

Implementação completa de um pipeline de previsão de séries temporais usando **SARIMAX** com **pmdarima** (auto_arima), seguindo a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

## 🎯 Metodologia CRISP-DM

Este projeto implementa todas as fases do CRISP-DM:

### 1. Business Understanding ✅
- **Objetivo**: Prever casos criminais mensais do TJGO usando modelo SARIMAX
- **Contexto**: Modelo estatístico adequado para séries temporais com sazonalidade e variáveis exógenas
- **Validação**: Decisões de modelagem baseadas em evidências estatísticas dos dados

### 2. Data Understanding ✅
- **Módulo**: `data_exploration.py`
- **Funcionalidades**:
  - Estatísticas descritivas detalhadas
  - Testes de estacionariedade (ADF, KPSS)
  - Decomposição sazonal (aditiva e multiplicativa)
  - Análise de autocorrelação (ACF, PACF)
  - Análise de variáveis exógenas
  - Geração de relatório automático

### 3. Data Preparation ✅
- **Módulo**: `data_preparation.py`
- **Funcionalidades**:
  - Carregamento e merge de datasets
  - Limpeza de dados (outliers, valores faltantes)
  - Criação de features de lag
  - Preparação de dados no formato SARIMAX

### 4. Modeling ✅
- **Módulo**: `sarimax_model.py`
- **Tecnologia**: `pmdarima.auto_arima`
- **Características**:
  - Busca automática de hiperparâmetros
  - Seleção de ordens (p, d, q) e (P, D, Q, s)
  - Suporte a variáveis exógenas
  - Otimização por AICc (adequado para amostras pequenas)
  - Modelo wrapper completo

### 5. Evaluation ✅
- **Módulo**: `evaluation.py`
- **Métricas**:
  - MAE, MSE, RMSE
  - MAPE, SMAPE
  - R²
  - AIC, BIC
- **Diagnósticos**:
  - Teste de Ljung-Box (autocorrelação dos resíduos)
  - Testes de normalidade (Jarque-Bera, Shapiro-Wilk)
  - Análise de heterocedasticidade
  - Validação cruzada temporal

### 6. Deployment ✅
- **Módulo**: `pipeline.py`
- **Script**: `run_sarimax.py`
- **Funcionalidades**:
  - Pipeline completo executável
  - Salvamento automático de artefatos
  - Relatórios gerados automaticamente

## 🚀 Como Usar

### Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

**Dependências principais adicionadas:**
- `pmdarima>=2.0.0` - Auto ARIMA/SARIMAX
- `statsmodels>=0.13.0` - Análises estatísticas
- `scipy>=1.9.0` - Testes estatísticos

### Execução Completa

```bash
# Execução completa do pipeline
python run_sarimax.py

# Com configuração personalizada
python run_sarimax.py --config configs/sarimax_config.yaml

# Apenas análise exploratória
python run_sarimax.py --exploration-only

# Pular análise exploratória (re-execução)
python run_sarimax.py --skip-exploration

# Modo verbose
python run_sarimax.py --verbose
```

### Execução Programática

```python
from sarimax.pipeline import SARIMAXPipeline

# Inicializa pipeline
pipeline = SARIMAXPipeline("configs/sarimax_config.yaml")

# Executa pipeline completo
results = pipeline.run_full_pipeline()

# Acessa resultados
print(f"Modelo: {results['model']['best_params']}")
print(f"Métricas: {results['evaluation']['metrics']}")
print(f"Previsões: {results['forecast']}")
```

## 📁 Estrutura de Arquivos

```
sarimax/
├── __init__.py              # Módulo package
├── data_exploration.py      # CRISP-DM Fase 2: Data Understanding
├── data_preparation.py       # CRISP-DM Fase 3: Data Preparation
├── sarimax_model.py          # CRISP-DM Fase 4: Modeling
├── evaluation.py             # CRISP-DM Fase 5: Evaluation
├── pipeline.py               # CRISP-DM Fase 6: Deployment
└── README.md                # Esta documentação
```

## ⚙️ Configuração

O arquivo de configuração está em `configs/sarimax_config.yaml` e inclui:

### Modelo SARIMAX
- **auto_arima**: Configurações de busca automática
  - Ordens máximas (p, d, q, P, D, Q)
  - Sazonalidade (seasonal, m=12)
  - Critério de informação (AICc)
  - Método de busca (stepwise)

### Variáveis Exógenas
- Features selecionadas baseadas em análise exploratória
- Criação automática de lags (configurável)

### Avaliação
- Métricas a calcular
- Testes de diagnóstico
- Validação cruzada temporal

## 📊 Saídas do Pipeline

O pipeline gera automaticamente:

1. **Modelo Treinado**: `outputs/sarimax/models/sarimax_model_*.joblib`
2. **Previsões**: `outputs/sarimax/predictions/forecast_*.csv`
3. **Métricas**: `outputs/sarimax/reports/metrics_*.json`
4. **Configuração**: `outputs/sarimax/reports/config_*.yaml`
5. **Análise Exploratória**: `sarimax/data_exploration_report.md`

## 🔍 Decisões Baseadas em Dados

### Estacionariedade
- Testes ADF e KPSS determinam necessidade de diferenciação
- `auto_arima` decide `d` e `D` automaticamente baseado nos testes

### Sazonalidade
- Decomposição sazonal identifica padrões anuais
- `seasonal=True` e `m=12` para dados mensais
- Força da sazonalidade calculada para validar componente sazonal

### Variáveis Exógenas
- Correlação com variável alvo determina features relevantes
- Features de lag criadas para capturar dependências temporais

### Ordens do Modelo
- `auto_arima` busca automaticamente melhor combinação de (p,d,q) e (P,D,Q,s)
- Critério AICc usado (adequado para amostras pequenas)

## 📈 Validação e Diagnósticos

### Testes de Resíduos
- **Ljung-Box**: Verifica se resíduos são ruído branco (ideal)
- **Normalidade**: Testa distribuição normal dos resíduos
- **Heterocedasticidade**: Verifica variância constante

### Validação Temporal
- Time Series Cross-Validation respeitando ordem temporal
- Evita vazamento de dados futuros

## 🎓 Fundamentos Estatísticos

### SARIMAX
- **SARIMA**: ARIMA com componente sazonal
- **SARIMAX**: SARIMA com variáveis exógenas (X)
- **Ordens**: (p,d,q)(P,D,Q)s
  - p, P: ordens AR (auto-regressivo)
  - d, D: diferenciações (não-sazonal e sazonal)
  - q, Q: ordens MA (média móvel)
  - s: período sazonal (12 para dados mensais)

### pmdarima.auto_arima
- Busca automática de hiperparâmetros ótimos
- Usa stepwise search para eficiência
- Testa múltiplas combinações e seleciona melhor por critério de informação

## 🔧 Customização

### Ajustar Busca do Modelo

Edite `configs/sarimax_config.yaml`:

```yaml
model:
  auto_arima:
    max_p: 5      # Aumentar para busca mais ampla
    max_q: 5
    seasonal: true
    m: 12
    information_criterion: "aicc"  # ou "aic", "bic", "hqic"
```

### Variáveis Exógenas

```yaml
model:
  exogenous_vars:
    enabled: true
    features: ["feature1", "feature2", ...]
    create_lags: true
    max_lags: 3
```

## 📝 Exemplo de Saída

```
SARIMAX FORECASTING PIPELINE - CASOS CRIMINAIS TJGO
Metodologia CRISP-DM
======================================================================

CRISP-DM FASE 2: DATA UNDERSTANDING
======================================================================
CARREGANDO DADOS
...

CRISP-DM FASE 3: DATA PREPARATION
======================================================================
Loading and merging datasets...
...

CRISP-DM FASE 4: MODELING
======================================================================
Fitting SARIMAX model using auto_arima...
Best order (p,d,q): (2, 1, 1)
Best seasonal order (P,D,Q,s): (1, 1, 1, 12)
AIC: 1234.56
AICc: 1245.67
BIC: 1267.89

CRISP-DM FASE 5: EVALUATION
======================================================================
...

RESULTADOS DA PREVISÃO - SARIMAX
======================================================================

Modelo Selecionado:
  Ordem (p,d,q): (2, 1, 1)
  Ordem Sazonal (P,D,Q,s): (1, 1, 1, 12)
  AIC: 1234.56
  BIC: 1267.89

Métricas de Avaliação:
  RMSE: 123.45
  MAPE: 5.67%
  R²: 0.89

PIPELINE CONCLUÍDO COM SUCESSO!
```

## 🐛 Troubleshooting

### Erro: "pmdarima not installed"
```bash
pip install pmdarima statsmodels scipy
```

### Aviso: "No valid data for metrics calculation"
- Verifique se há valores faltantes nos dados
- Ajuste `fill_missing` na configuração

### Modelo demorando muito para treinar
- Reduza `max_p`, `max_q`, `max_P`, `max_Q`
- Use `stepwise=True` (já é padrão)
- Reduza `max_order`

## 📚 Referências

- [CRISP-DM Methodology](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- [pmdarima Documentation](https://alkaline-ml.com/pmdarima/)
- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Time Series Analysis with Python](https://www.statsmodels.org/stable/tsa.html)

## ✅ Checklist de Implementação

- [x] Análise exploratória completa (CRISP-DM Fase 2)
- [x] Preparação de dados (CRISP-DM Fase 3)
- [x] Modelagem SARIMAX com auto_arima (CRISP-DM Fase 4)
- [x] Avaliação completa com diagnósticos (CRISP-DM Fase 5)
- [x] Pipeline executável (CRISP-DM Fase 6)
- [x] Configuração via YAML
- [x] Documentação completa
- [x] Decisões baseadas em dados

---

**Autor**: Implementação seguindo metodologia CRISP-DM  
**Data**: 2025  
**Versão**: 1.0.0

