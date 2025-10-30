# Prophet Forecasting Pipeline - Estrutura Final do Projeto

## 📁 Estrutura Completa

```
prophet_forecasting_pipeline/
├── 📁 src/                           # Código fonte modular
│   ├── 📁 data/                      # Módulo de dados
│   │   └── __init__.py              # DataLoader, Preprocessor, Validator, DataManager
│   ├── 📁 models/                    # Módulo de modelos
│   │   └── __init__.py              # ProphetModelWrapper, HyperparameterOptimizer
│   ├── 📁 training/                  # Módulo de treinamento
│   │   └── __init__.py              # TimeSeriesSplitter, Trainer, CrossValidator, Pipeline
│   ├── 📁 evaluation/                # Módulo de avaliação
│   │   └── __init__.py              # MetricsCalculator, ComponentAnalyzer, PerformanceAnalyzer
│   ├── 📁 utils/                     # Módulo de utilitários
│   │   └── __init__.py              # ReportGenerator, Visualizer, ArtifactManager, ConfigManager
│   └── __init__.py                   # Pacote principal
├── 📁 tests/                         # Testes unitários
│   └── test_pipeline.py             # Testes completos para todos os módulos
├── 📁 configs/                       # Configurações
│   ├── default_config.yaml          # Configuração padrão
│   └── criminal_cases_config.yaml   # Configuração específica para casos criminais
├── 📁 outputs/                       # Artefatos gerados (criado automaticamente)
│   ├── 📁 models/                   # Modelos treinados
│   ├── 📁 predictions/              # Previsões
│   └── 📁 reports/                  # Relatórios e métricas
├── 📁 logs/                          # Logs (criado automaticamente)
├── 📄 main.py                       # Pipeline genérico principal
├── 📄 run_criminal_cases.py         # Pipeline específico para casos criminais
├── 📄 example.py                    # Exemplo de uso com dados sintéticos
├── 📄 requirements.txt              # Dependências Python
├── 📄 setup.py                      # Configuração do pacote
├── 📄 README.md                     # Documentação principal
├── 📄 EXECUTION_GUIDE.md            # Guia detalhado de execução
├── 📄 QUICK_START.md                # Guia de início rápido
├── 📄 PROJECT_SUMMARY.md            # Resumo completo do projeto
└── 📄 feature_selection_results.md  # Resultados da análise de seleção de características
```

## Scripts Principais

### 1. **Pipeline Genérico**

```bash
python main.py --config configs/default_config.yaml
```

### 2. **Pipeline Específico para Casos Criminais**

```bash
python run_criminal_cases.py
```

### 3. **Exemplo com Dados Sintéticos**

```bash
python example.py
```

### 4. **Testes Unitários**

```bash
python -m pytest tests/ -v
```

## Funcionalidades Implementadas

### ✅ **Pipeline Automatizado e Reprodutível**

- Estrutura modular com `src/` seguindo padrões de ML
- Configuração via YAML para fácil reprodução
- Logging estruturado e rastreável
- Gerenciamento de artefatos automatizado

### ✅ **Suporte a Variáveis Exógenas**

- Integração completa com Prophet
- Suporte a múltiplas variáveis exógenas
- Criação automática de features de lag
- Validação de dados de entrada

### ✅ **Validação Temporal (Time Series Cross-Validation)**

- TimeSeriesSplit personalizado
- Validação cruzada temporal adequada
- Métricas de avaliação abrangentes
- Análise de performance temporal

### ✅ **Otimização de Hiperparâmetros**

- Integração com Optuna
- Otimização automática de parâmetros Prophet
- Validação temporal durante otimização
- Configuração flexível de trials

### ✅ **Geração de Relatórios**

- Relatórios HTML interativos
- Métricas detalhadas (MAE, RMSE, MAPE, R², SMAPE)
- Análise de componentes do modelo
- Diagnósticos de resíduos

### ✅ **Salvamento de Artefatos**

- Modelos treinados (.joblib)
- Previsões com intervalos (.csv)
- Métricas de avaliação (.json)
- Configurações (.yaml)

## Dados Utilizados

### Série Temporal Principal

- **Arquivo**: `data/raw_data/casos_mensal__criminal_series_2015_2024.csv`
- **Período**: 2015-2024 (120 meses)
- **Variável**: TOTAL_CASOS

### Variáveis Exógenas Selecionadas

Baseadas na análise de seleção de características:

1. `atendimento_pre_hospitalar` (atual + lags 1-3)
2. `pessoa_localizada_lag_3`
3. `lesao_corporal_seguida_de_morte` (atual + lag 2)
4. `tentativa_de_feminicidio_lag_2`
5. `morte_de_agente_do_estado_lag_1`
6. `suicidio_de_agente_do_estado`

## Configuração

### Arquivo Principal

`configs/criminal_cases_config.yaml`

### Principais Configurações

- **Horizonte de Previsão**: 12 meses
- **Validação Cruzada**: 5 splits
- **Otimização**: 50 trials
- **Métricas**: MAE, RMSE, MAPE, R²
- **Feriados**: Brasil habilitado

## Resultados Esperados

### Artefatos Gerados

- **Modelo**: `outputs/models/prophet_model_*.joblib`
- **Previsões**: `outputs/predictions/forecast_*.csv`
- **Métricas**: `outputs/reports/metrics_*.json`
- **Relatório**: `outputs/reports/prophet_report_*.html`

### Visualizações

- Gráfico de previsão com intervalos de confiança
- Análise de componentes do modelo
- Diagnósticos de resíduos
- Métricas de performance

## Testes Implementados

### Cobertura de Testes

- ✅ Módulo de dados (DataLoader, Preprocessor, Validator)
- ✅ Módulo de modelos (ProphetWrapper, Optimizer)
- ✅ Módulo de treinamento (Trainer, CrossValidator)
- ✅ Módulo de avaliação (MetricsCalculator, Analyzer)
- ✅ Módulo de utilitários (ConfigManager, Visualizer)
- ✅ Testes de integração end-to-end

## Documentação

### Arquivos de Documentação

- **README.md**: Visão geral do projeto
- **EXECUTION_GUIDE.md**: Guia detalhado de execução
- **QUICK_START.md**: Guia de início rápido
- **PROJECT_SUMMARY.md**: Resumo completo do projeto
- **Docstrings**: Documentação inline completa

## Como Executar

### 1. Instalação

```bash
pip install -r requirements.txt
```

### 2. Execução Rápida

```bash
python run_criminal_cases.py
```

### 3. Execução com Opções

```bash
python run_criminal_cases.py --quick --verbose
```

### 4. Executar Testes

```bash
python -m pytest tests/ -v
```

## 🏆 Conclusão

O projeto implementa com sucesso um **pipeline completo e profissional** para previsão de séries temporais usando Prophet, com todas as funcionalidades solicitadas:

- ✅ Pipeline automatizado e reprodutível
- ✅ Suporte a variáveis exógenas
- ✅ Validação temporal adequada
- ✅ Otimização de hiperparâmetros
- ✅ Geração de relatórios completos
- ✅ Salvamento de artefatos
- ✅ Código modular e testado
- ✅ Documentação completa

O pipeline está pronto para uso em produção e pode ser facilmente adaptado para outros projetos de forecasting.
