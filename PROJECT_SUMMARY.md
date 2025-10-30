# 🎯 Prophet Forecasting Pipeline - Projeto Completo

## 📋 Resumo do Projeto

Desenvolvi um **pipeline automatizado e reprodutível** para previsão de séries temporais mensais utilizando Prophet (Facebook/Prophet) com todas as funcionalidades solicitadas.

## ✅ Funcionalidades Implementadas

### 🔄 Pipeline Automatizado e Reprodutível
- ✅ Estrutura modular com `src/` seguindo padrões de ML
- ✅ Configuração via YAML para fácil reprodução
- ✅ Logging estruturado e rastreável
- ✅ Gerenciamento de artefatos automatizado

### 📊 Suporte a Variáveis Exógenas
- ✅ Integração completa com Prophet
- ✅ Suporte a múltiplas variáveis exógenas
- ✅ Criação automática de features de lag
- ✅ Validação de dados de entrada

### ⏰ Validação Temporal (Time Series Cross-Validation)
- ✅ TimeSeriesSplit personalizado
- ✅ Validação cruzada temporal adequada
- ✅ Métricas de avaliação abrangentes
- ✅ Análise de performance temporal

### 🎛️ Otimização de Hiperparâmetros
- ✅ Integração com Optuna
- ✅ Otimização automática de parâmetros Prophet
- ✅ Validação temporal durante otimização
- ✅ Configuração flexível de trials

### 📈 Geração de Relatórios
- ✅ Relatórios HTML interativos
- ✅ Métricas detalhadas (MAE, RMSE, MAPE, R², SMAPE)
- ✅ Análise de componentes do modelo
- ✅ Diagnósticos de resíduos

### 💾 Salvamento de Artefatos
- ✅ Modelos treinados (.joblib)
- ✅ Previsões com intervalos (.csv)
- ✅ Métricas de avaliação (.json)
- ✅ Configurações (.yaml)

## 🏗️ Estrutura do Projeto

```
prophet_forecasting_pipeline/
├── src/                    # Código modular
│   ├── data/              # Carregamento e pré-processamento
│   ├── models/            # Wrapper Prophet com exógenas
│   ├── training/          # Pipeline de treinamento
│   ├── evaluation/        # Avaliação e métricas
│   └── utils/             # Relatórios e visualizações
├── tests/                 # Testes unitários completos
├── configs/               # Configurações YAML
├── outputs/               # Artefatos gerados
├── logs/                  # Logs estruturados
├── main.py               # Pipeline genérico
├── run_criminal_cases.py # Pipeline específico
├── example.py            # Exemplo de uso
├── requirements.txt      # Dependências
├── setup.py             # Configuração do pacote
└── README.md            # Documentação completa
```

## 🚀 Como Executar

### Instalação Rápida
```bash
pip install -r requirements.txt
```

### Execução do Pipeline
```bash
# Pipeline completo para casos criminais
python run_criminal_cases.py

# Execução rápida (menos otimização)
python run_criminal_cases.py --quick

# Com logs detalhados
python run_criminal_cases.py --verbose
```

### Exemplo de Uso
```bash
python example.py
```

### Testes
```bash
python -m pytest tests/ -v
```

## 📊 Dados Utilizados

### Série Temporal Principal
- **Arquivo**: `casos_mensal__criminal_series_2015_2024.csv`
- **Período**: 2015-2024 (120 meses)
- **Variável**: TOTAL_CASOS

### Variáveis Exógenas Selecionadas
Baseadas na análise de seleção de características anterior:

1. `atendimento_pre_hospitalar` (atual + lags 1-3)
2. `pessoa_localizada_lag_3`
3. `lesao_corporal_seguida_de_morte` (atual + lag 2)
4. `tentativa_de_feminicidio_lag_2`
5. `morte_de_agente_do_estado_lag_1`
6. `suicidio_de_agente_do_estado`

## 🎯 Características Técnicas

### Modelo Prophet Aprimorado
- Suporte completo a variáveis exógenas
- Feriados brasileiros configurados
- Componentes sazonais personalizáveis
- Intervalos de predição (80%, 95%)

### Validação Temporal
- TimeSeriesSplit com 5 folds
- Gap configurável entre treino/teste
- Métricas de avaliação abrangentes
- Análise de performance por período

### Otimização de Hiperparâmetros
- Optuna para otimização automática
- 50 trials por padrão (configurável)
- Parâmetros otimizados:
  - `changepoint_prior_scale`
  - `seasonality_prior_scale`
  - `holidays_prior_scale`

### Relatórios e Visualizações
- Relatórios HTML interativos
- Gráficos de previsão com intervalos
- Análise de componentes do modelo
- Diagnósticos de resíduos
- Métricas de performance detalhadas

## 📈 Resultados Esperados

### Artefatos Gerados
- **Modelo**: `outputs/models/prophet_model_*.joblib`
- **Previsões**: `outputs/predictions/forecast_*.csv`
- **Métricas**: `outputs/reports/metrics_*.json`
- **Relatório**: `outputs/reports/prophet_report_*.html`

### Métricas de Avaliação
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- R² (R-squared)

## 🔧 Configuração

### Arquivo Principal
`configs/criminal_cases_config.yaml`

### Principais Configurações
- **Horizonte**: 12 meses
- **CV**: 5 splits temporais
- **Otimização**: 50 trials
- **Feriados**: Brasil habilitado
- **Exógenas**: 10 features selecionadas

## 🧪 Testes Implementados

### Cobertura de Testes
- ✅ Módulo de dados (DataLoader, Preprocessor, Validator)
- ✅ Módulo de modelos (ProphetWrapper, Optimizer)
- ✅ Módulo de treinamento (Trainer, CrossValidator)
- ✅ Módulo de avaliação (MetricsCalculator, Analyzer)
- ✅ Módulo de utilitários (ConfigManager, Visualizer)
- ✅ Testes de integração end-to-end

### Execução dos Testes
```bash
python -m pytest tests/ -v --cov=src
```

## 📚 Documentação

### Arquivos de Documentação
- **README.md**: Visão geral do projeto
- **EXECUTION_GUIDE.md**: Guia detalhado de execução
- **QUICK_START.md**: Guia de início rápido
- **Docstrings**: Documentação inline completa

### Exemplos de Uso
- **example.py**: Exemplo completo com dados sintéticos
- **run_criminal_cases.py**: Pipeline específico para dados reais
- **Configurações**: Exemplos de configuração YAML

## 🎉 Diferenciais do Projeto

### 1. **Arquitetura Modular**
- Separação clara de responsabilidades
- Fácil manutenção e extensão
- Código reutilizável

### 2. **Configuração Flexível**
- Configuração via YAML
- Parâmetros facilmente ajustáveis
- Suporte a diferentes cenários

### 3. **Validação Temporal Adequada**
- TimeSeriesSplit personalizado
- Evita vazamento de dados
- Métricas realistas

### 4. **Otimização Inteligente**
- Integração com Optuna
- Validação temporal durante otimização
- Configuração flexível de trials

### 5. **Relatórios Completos**
- HTML interativo
- Visualizações automáticas
- Análise de componentes
- Diagnósticos de modelo

### 6. **Gerenciamento de Artefatos**
- Salvamento automático
- Versionamento de modelos
- Rastreabilidade completa

## 🚀 Próximos Passos

1. **Execute o Pipeline**
   ```bash
   python run_criminal_cases.py
   ```

2. **Analise os Resultados**
   - Abra o relatório HTML
   - Verifique as métricas
   - Analise as visualizações

3. **Ajuste se Necessário**
   - Modifique a configuração
   - Adicione novas variáveis
   - Ajuste parâmetros

4. **Integre com Produção**
   - Use o modelo treinado
   - Automatize execução
   - Monitore performance

---

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
