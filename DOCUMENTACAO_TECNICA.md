# DOCUMENTAÇÃO TÉCNICA DETALHADA - PIPELINE PROPHET TJGO

## ÍNDICE

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Módulos Detalhados](#módulos-detalhados)
4. [Configurações](#configurações)
5. [Execução e Uso](#execução-e-uso)
6. [Troubleshooting](#troubleshooting)
7. [Extensibilidade](#extensibilidade)

## VISÃO GERAL

### Objetivo

Desenvolver um pipeline automatizado e reprodutível para previsão de séries temporais mensais de casos criminais utilizando Facebook Prophet, com suporte a variáveis exógenas, validação temporal, otimização de hiperparâmetros e geração de relatórios.

### Características Principais

- **Modular**: Arquitetura baseada em módulos independentes
- **Configurável**: Configurações via arquivos YAML
- **Reprodutível**: Versionamento e documentação completa
- **Escalável**: Fácil extensão para novos datasets
- **Robusto**: Tratamento de erros e validações

## ARQUITETURA DO SISTEMA

### Estrutura de Diretórios

```
predict_series_criminal_tjgo/
├── src/                          # Código fonte principal
│   ├── data/                     # Gerenciamento de dados
│   ├── models/                   # Modelos Prophet
│   ├── training/                 # Treinamento e otimização
│   ├── evaluation/               # Avaliação e métricas
│   └── utils/                    # Utilitários e helpers
├── configs/                      # Arquivos de configuração
├── data/                         # Dados de entrada
│   ├── raw_data/                 # Dados brutos
│   └── processed/                # Dados processados
├── artifacts/                    # Artefatos gerados
├── outputs/                      # Saídas do pipeline
├── tests/                        # Testes unitários
└── docs/                         # Documentação
```

### Fluxo de Dados

```
Dados Brutos → Validação → Pré-processamento → Treinamento → Avaliação → Relatórios
     ↓              ↓              ↓              ↓           ↓          ↓
   CSV Files → DataManager → ProphetModel → Evaluator → Visualizer → Artifacts
```

## MÓDULOS DETALHADOS

### 1. Módulo Data (`src/data/`)

#### DataManager

**Responsabilidade**: Gerenciamento completo do ciclo de vida dos dados

**Métodos Principais**:

- `load_target_series()`: Carrega série temporal principal
- `load_exogenous_features()`: Carrega variáveis exógenas
- `merge_datasets()`: Une datasets por data
- `validate_target_series()`: Valida série temporal
- `validate_exogenous_features()`: Valida features exógenas
- `remove_outliers()`: Remove outliers usando IQR
- `fill_missing_values()`: Preenche valores faltantes
- `create_lag_features()`: Cria features de lag
- `prepare_prophet_data()`: Formata dados para Prophet

**Exemplo de Uso**:

```python
from src.data import DataManager

config = ConfigManager.load_config('configs/criminal_cases_config.yaml')
data_manager = DataManager(config)
prophet_df, exogenous_features = data_manager.load_and_prepare_data()
```

#### DataValidator

**Responsabilidade**: Validação de qualidade e schema dos dados

**Validações Implementadas**:

- Presença de colunas obrigatórias
- Valores nulos
- Tipos de dados corretos
- Consistência temporal
- Faixas de valores esperadas

### 2. Módulo Models (`src/models/`)

#### ProphetModelWrapper

**Responsabilidade**: Wrapper para modelos Prophet com funcionalidades estendidas

**Métodos Principais**:

- `fit()`: Treina o modelo Prophet
- `predict()`: Gera previsões
- `get_model_components()`: Extrai componentes do modelo
- `get_feature_importance()`: Calcula importância das features
- `save_model()`: Salva modelo treinado
- `load_model()`: Carrega modelo salvo

**Configurações Suportadas**:

- Sazonalidade (anual, semanal, diária)
- Feriados (países específicos)
- Variáveis exógenas
- Hiperparâmetros personalizados

#### ProphetHyperparameterOptimizer

**Responsabilidade**: Otimização automática de hiperparâmetros usando Optuna

**Parâmetros Otimizados**:

- `changepoint_prior_scale`: Controle de mudanças de tendência
- `seasonality_prior_scale`: Força da sazonalidade
- `holidays_prior_scale`: Força dos efeitos de feriados

**Exemplo de Uso**:

```python
from src.models import ProphetHyperparameterOptimizer

optimizer = ProphetHyperparameterOptimizer(config)
best_params = optimizer.optimize(df, exogenous_features, n_trials=50)
```

### 3. Módulo Training (`src/training/`)

#### TrainingPipeline

**Responsabilidade**: Orquestração completa do processo de treinamento

**Etapas do Treinamento**:

1. **Preparação de Dados**: Divisão temporal
2. **Otimização**: Busca de hiperparâmetros ótimos
3. **Treinamento Final**: Modelo com parâmetros otimizados
4. **Validação Cruzada**: Avaliação temporal
5. **Análise de Features**: Importância das variáveis

#### TimeSeriesSplitter

**Responsabilidade**: Divisão temporal adequada para séries temporais

**Características**:

- Respeita ordem temporal
- Configurável (número de splits, horizonte)
- Evita vazamento de dados futuros

### 4. Módulo Evaluation (`src/evaluation/`)

#### ModelEvaluator

**Responsabilidade**: Avaliação abrangente do modelo

**Métricas Calculadas**:

- **RMSE**: Raiz do erro quadrático médio
- **MAE**: Erro absoluto médio
- **MAPE**: Erro percentual absoluto médio
- **SMAPE**: Erro percentual simétrico médio
- **R²**: Coeficiente de determinação

#### MetricsCalculator

**Responsabilidade**: Cálculo de métricas de avaliação

**Funcionalidades**:

- Cálculo robusto (tratamento de NaN)
- Múltiplas métricas simultâneas
- Validação de entrada

#### ComponentAnalyzer

**Responsabilidade**: Análise dos componentes do modelo Prophet

**Análises Incluídas**:

- Decomposição da série temporal
- Análise de tendência
- Análise de sazonalidade
- Efeitos de feriados

#### PerformanceAnalyzer

**Responsabilidade**: Análise de performance temporal

**Análises Incluídas**:

- Performance por período
- Análise de resíduos
- Detecção de padrões
- Comparação temporal

### 5. Módulo Utils (`src/utils/`)

#### ConfigManager

**Responsabilidade**: Gerenciamento de configurações

**Funcionalidades**:

- Carregamento de YAML
- Validação de configurações
- Configurações padrão
- Merge de configurações

#### Visualizer

**Responsabilidade**: Geração de visualizações

**Gráficos Suportados**:

- Gráfico de previsão
- Gráfico de componentes
- Análise de resíduos
- Gráficos de performance

#### ReportGenerator

**Responsabilidade**: Geração de relatórios

**Relatórios Gerados**:

- Relatório de performance
- Relatório de componentes
- Relatório de métricas
- Relatório executivo

#### ArtifactManager

**Responsabilidade**: Gerenciamento de artefatos

**Artefatos Gerenciados**:

- Modelos treinados
- Previsões
- Métricas
- Configurações
- Visualizações

## CONFIGURAÇÕES

### Estrutura de Configuração

#### Configuração de Dados

```yaml
data:
  raw_data_path: "data/raw_data/casos_mensal__criminal_series_2015_2024.csv"
  external_features_path: "data/raw_data/external_features_2015_2024.csv"
  target_column: "TOTAL_CASOS"
  date_column: "DATA"
  exogenous_features: ["feature1", "feature2"]
  test_size: 12
  validation_horizon: 3
  initial_train_size: 60
  period_step: 3
```

#### Configuração do Modelo

```yaml
model:
  growth: "linear"
  seasonality_mode: "additive"
  daily_seasonality: false
  weekly_seasonality: true
  yearly_seasonality: true
  holidays: "BR"
  changepoint_prior_scale: 0.05
  seasonality_prior_scale: 10.0
  holidays_prior_scale: 10.0
```

#### Configuração de Treinamento

```yaml
training:
  hyperparameter_optimization:
    enabled: true
    n_trials: 50
    timeout: 3600
    param_space:
      changepoint_prior_scale: [0.001, 0.5]
      seasonality_prior_scale: [0.1, 20.0]
      holidays_prior_scale: [0.1, 20.0]
    metric: "rmse"
  cv:
    n_splits: 5
    test_size: 12
    gap: 0
```

#### Configuração de Avaliação

```yaml
evaluation:
  metrics: ["rmse", "mae", "mape", "r2"]
  component_analysis:
    enabled: true
    components: ["trend", "seasonal", "holidays"]
  performance_analysis:
    enabled: true
    periods: ["monthly", "quarterly", "yearly"]
```

#### Configuração de Artefatos

```yaml
artifacts:
  output_dir: "artifacts"
  model_name: "prophet_model.pkl"
  forecast_name: "forecast.csv"
  metrics_name: "metrics.json"
  plots_dir: "plots"
```

### Configurações Específicas

#### Configuração para Casos Criminais

- **Arquivo**: `configs/criminal_cases_config.yaml`
- **Features Exógenas**: 6 variáveis selecionadas
- **Período**: 2015-2024
- **Otimização**: 5 trials (configuração rápida)

#### Configuração Padrão

- **Arquivo**: `configs/default_config.yaml`
- **Configuração**: Genérica para qualquer dataset
- **Otimização**: 50 trials (configuração completa)

## EXECUÇÃO E USO

### Pré-requisitos

#### Dependências Python

```bash
pip install -r requirements.txt
```

#### Dependências Principais

- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `prophet>=1.1.0`
- `scikit-learn>=1.1.0`
- `optuna>=3.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `plotly>=5.0.0`
- `pyyaml>=6.0`
- `loguru>=0.6.0`

### Execução Básica

#### 1. Execução Simples

```bash
python3 execute_pipeline.py
```

#### 2. Execução com Configuração Específica

```bash
python3 run_criminal_cases.py --config configs/criminal_cases_config.yaml
```

#### 3. Execução Completa

```bash
python3 main.py --config configs/criminal_cases_config.yaml
```

### Execução Avançada

#### Com Parâmetros Personalizados

```bash
python3 main.py \
  --config configs/criminal_cases_config.yaml \
  --output-dir custom_output \
  --n-trials 100 \
  --timeout 7200
```

#### Modo Debug

```bash
python3 main.py \
  --config configs/criminal_cases_config.yaml \
  --debug \
  --verbose
```

### Uso Programático

#### Exemplo Básico

```python
from src.data import DataManager
from src.models import ProphetModelWrapper
from src.training import TrainingPipeline
from src.evaluation import ModelEvaluator
from src.utils import ConfigManager

# Carregar configuração
config = ConfigManager.load_config('configs/criminal_cases_config.yaml')

# Inicializar componentes
data_manager = DataManager(config)
training_pipeline = TrainingPipeline(config)
evaluator = ModelEvaluator(config)

# Executar pipeline
prophet_df, exogenous_features = data_manager.load_and_prepare_data()
training_results = training_pipeline.run_training_pipeline(prophet_df, exogenous_features)
model_wrapper = training_results['model']

# Fazer previsões
forecast = model_wrapper.predict(prophet_df, horizon_months=12)

# Avaliar modelo
evaluation_results = evaluator.evaluate_model(model_wrapper, forecast, prophet_df)
```

#### Exemplo Avançado

```python
import optuna
from src.models import ProphetHyperparameterOptimizer

# Otimização personalizada
def objective(trial):
    optimizer = ProphetHyperparameterOptimizer(config)
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.1, 20.0),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1, 20.0)
    }
    return optimizer.optimize_with_params(df, exogenous_features, params)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## TROUBLESHOOTING

### Problemas Comuns

#### 1. Erro de Importação

**Problema**: `ModuleNotFoundError`
**Solução**:

```bash
# Instalar dependências
pip install -r requirements.txt

# Verificar Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Erro de Dados

**Problema**: `ValueError: operands could not be broadcast together`
**Solução**:

- Verificar formato dos dados
- Garantir que datas estão no formato correto
- Verificar se não há valores nulos

#### 3. Erro de Memória

**Problema**: `MemoryError`
**Solução**:

- Reduzir número de trials de otimização
- Usar configuração rápida
- Processar dados em chunks menores

#### 4. Erro de Configuração

**Problema**: `KeyError` em configurações
**Solução**:

- Verificar arquivo YAML
- Usar configuração padrão como base
- Validar estrutura de configuração

### Logs e Debugging

#### Configuração de Logs

```python
from loguru import logger

# Configurar nível de log
logger.add("logs/pipeline.log", level="DEBUG")

# Log personalizado
logger.info("Executando pipeline...")
logger.error("Erro encontrado: {error}", error=str(e))
```

#### Modo Debug

```bash
# Executar com debug
python3 main.py --config configs/criminal_cases_config.yaml --debug

# Logs detalhados
tail -f logs/pipeline.log
```

### Validação de Dados

#### Verificar Dados de Entrada

```python
# Verificar formato dos dados
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Verificar datas
print(df['DATA'].min(), df['DATA'].max())
print(df['DATA'].dtype)
```

#### Verificar Configuração

```python
from src.utils import ConfigManager

# Validar configuração
config = ConfigManager.load_config('configs/criminal_cases_config.yaml')
print(config.keys())
print(config['data'])
```

## EXTENSIBILIDADE

### Adicionando Novos Datasets

#### 1. Criar Nova Configuração

```yaml
# configs/new_dataset_config.yaml
data:
  raw_data_path: "data/raw_data/new_dataset.csv"
  target_column: "TARGET"
  date_column: "DATE"
  exogenous_features: ["feature1", "feature2"]
```

#### 2. Adaptar DataManager

```python
class CustomDataManager(DataManager):
    def load_target_series(self):
        # Implementação customizada
        pass
```

#### 3. Executar Pipeline

```bash
python3 main.py --config configs/new_dataset_config.yaml
```

### Adicionando Novas Métricas

#### 1. Estender MetricsCalculator

```python
class ExtendedMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, actual, predicted):
        # Implementação da nova métrica
        pass
```

#### 2. Atualizar Configuração

```yaml
evaluation:
  metrics: ["rmse", "mae", "mape", "r2", "custom_metric"]
```

### Adicionando Novos Modelos

#### 1. Criar Wrapper Personalizado

```python
class CustomModelWrapper:
    def __init__(self, config):
        self.config = config
  
    def fit(self, df):
        # Implementação do modelo
        pass
  
    def predict(self, df, horizon_months):
        # Implementação da previsão
        pass
```

#### 2. Integrar ao Pipeline

```python
# Atualizar TrainingPipeline para usar novo modelo
class ExtendedTrainingPipeline(TrainingPipeline):
    def train_model(self, df, exogenous_features):
        # Usar CustomModelWrapper
        pass
```

### Adicionando Novas Visualizações

#### 1. Estender Visualizer

```python
class ExtendedVisualizer(Visualizer):
    def create_custom_plot(self, data):
        # Implementação do novo gráfico
        pass
```

#### 2. Integrar ao Pipeline

```python
# Adicionar ao ReportGenerator
class ExtendedReportGenerator(ReportGenerator):
    def generate_custom_report(self, data):
        # Usar ExtendedVisualizer
        pass
```

## MONITORAMENTO E MANUTENÇÃO

### Monitoramento de Performance

#### Métricas de Monitoramento

- **Accuracy**: Precisão das previsões
- **Latency**: Tempo de execução
- **Memory Usage**: Uso de memória
- **Error Rate**: Taxa de erros

#### Alertas Automáticos

```python
def check_model_performance(metrics):
    if metrics['r2'] < 0.8:
        logger.warning("Model performance degraded")
        # Enviar alerta
  
    if metrics['mape'] > 20:
        logger.error("High prediction error")
        # Enviar alerta crítico
```

### Manutenção Preventiva

#### Retreinamento Automático

```python
def schedule_retraining():
    # Verificar se modelo precisa ser retreinado
    if should_retrain():
        logger.info("Starting model retraining")
        # Executar pipeline completo
```

#### Backup de Modelos

```python
def backup_models():
    # Fazer backup dos modelos
    # Versionar artefatos
    # Manter histórico de performance
```

## REFERÊNCIAS E RECURSOS

### Documentação Oficial

- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Optuna](https://optuna.org/)
- [Scikit-learn](https://scikit-learn.org/)

### Tutoriais e Exemplos

- [Prophet Tutorial](https://facebook.github.io/prophet/docs/quick_start.html)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

### Comunidade e Suporte

- [Prophet GitHub](https://github.com/facebook/prophet)
- [Optuna GitHub](https://github.com/optuna/optuna)

---


## Team and Contact

- **Autor** - Eng. Manuel Lucala Zengo
- **Mentorship** – UFG TI RESIDENCY
- **Team** - DIACDE TJGO
- **Methodology** – CRISP-DM adapted for time series
