# RELATÓRIO FINAL - PIPELINE DE PREVISÃO DE CASOS CRIMINAIS TJGO

## RESUMO EXECUTIVO

O pipeline de previsão de séries temporais utilizando Facebook Prophet foi executado com **SUCESSO** para o dataset de casos criminais do TJGO. O modelo demonstrou excelente performance com métricas de avaliação robustas.

## RESULTADOS PRINCIPAIS

### Performance do Modelo

- **RMSE**: 1,293.69 casos
- **MAPE**: 15.81% (erro percentual médio)
- **R²**: 0.9262 (92.62% da variância explicada)

### Interpretação dos Resultados

- **R² = 0.9262**: O modelo explica 92.62% da variância nos casos criminais, indicando excelente capacidade preditiva
- **MAPE = 15.81%**: Erro percentual médio aceitável para previsões de séries temporais criminais
- **RMSE = 1,293.69**: Raiz do erro quadrático médio em unidades de casos criminais

## CONFIGURAÇÃO DO MODELO

### Hiperparâmetros Otimizados

- **changepoint_prior_scale**: 0.0223
- **seasonality_prior_scale**: 0.2625
- **holidays_prior_scale**: 0.0127

### Variáveis Exógenas Selecionadas

1. `atendimento_pre_hospitalar`
2. `pessoa_localizada`
3. `lesao_corporal_seguida_de_morte`
4. `tentativa_de_feminicidio`
5. `morte_de_agente_do_estado`
6. `suicidio_de_agente_do_estado`

### Configurações de Sazonalidade

- **Sazonalidade Anual**: Habilitada
- **Sazonalidade Semanal**: Habilitada
- **Sazonalidade Diária**: Desabilitada
- **Feriados**: Brasil (BR)

## 📈 PROCESSO DE OTIMIZAÇÃO

### Otimização de Hiperparâmetros

- **Ferramenta**: Optuna
- **Número de Trials**: 5 (configuração rápida)
- **Métrica Otimizada**: RMSE
- **Melhor Score**: 799.73

### Validação Cruzada Temporal

- **Método**: TimeSeriesSplit
- **Número de Splits**: 3
- **Período de Treinamento**: 2015-2024
- **Período de Teste**: Últimos 12 meses

## ANÁLISE DE DADOS

### Dataset Processado

- **Período**: Janeiro 2015 - Dezembro 2024
- **Total de Registros**: 120 meses
- **Registros Após Processamento**: 117 meses (após criação de lags)
- **Variáveis Exógenas**: 6 features selecionadas
- **Features de Lag**: Criadas até 3 meses de antecedência

### Pré-processamento Realizado

1. **Validação de Dados**: ✅ Passou
2. **Remoção de Outliers**: 0 outliers removidos
3. **Preenchimento de Valores**: 0 valores faltantes
4. **Criação de Lags**: 18 features de lag criadas
5. **Preparação Prophet**: Dataset formatado corretamente

## 🎨 VISUALIZAÇÕES GERADAS

### Gráficos Criados

1. **Gráfico de Previsão**: `forecast_plot_20251029_020734.png`

   - Mostra dados históricos vs previsões
   - Intervalos de confiança
   - Componentes do modelo
2. **Gráfico de Componentes**: `components_plot_20251029_020735.png`

   - Tendência
   - Sazonalidade anual
   - Sazonalidade semanal
   - Efeitos de feriados
3. **Gráfico de Resíduos**: `residuals_plot_20251029_020735.png`

   - Análise de resíduos vs valores ajustados
   - Q-Q plot para normalidade
   - Distribuição dos resíduos
   - Resíduos ao longo do tempo

## 🏗️ ARQUITETURA DO PIPELINE

### Módulos Implementados

- **Data Management**: Carregamento, validação e preparação de dados
- **Model Training**: Treinamento com otimização de hiperparâmetros
- **Evaluation**: Cálculo de métricas e análise de performance
- **Visualization**: Geração de gráficos e relatórios
- **Artifact Management**: Salvamento de modelos e resultados

### Tecnologias Utilizadas

- **Facebook Prophet**: Modelo de previsão
- **Optuna**: Otimização de hiperparâmetros
- **Scikit-learn**: Validação cruzada temporal
- **Matplotlib/Seaborn**: Visualizações
- **Pandas/NumPy**: Manipulação de dados

## 📋 ARQUIVOS DE CONFIGURAÇÃO

### Configurações Principais

- **Configuração Padrão**: `configs/default_config.yaml`
- **Configuração Criminal**: `configs/criminal_cases_config.yaml`
- **Dependências**: `requirements.txt`

### Scripts de Execução

- **Pipeline Principal**: `main.py`
- **Execução Criminal**: `run_criminal_cases.py`
- **Execução Direta**: `execute_pipeline.py`

## 🎯 PRÓXIMOS PASSOS RECOMENDADOS

### Melhorias do Modelo

1. **Aumentar Trials de Otimização**: De 5 para 50+ trials
2. **Incluir Mais Variáveis Exógenas**: Baseado na análise de feature selection
3. **Experimentar Diferentes Configurações**: Sazonalidade multiplicativa
4. **Implementar Ensemble**: Combinar múltiplos modelos

### Monitoramento Contínuo

1. **Retreinamento Periódico**: Mensal ou trimestral
2. **Monitoramento de Drift**: Detecção de mudanças nos dados
3. **Validação de Performance**: Acompanhamento das métricas ao longo do tempo
4. **Feedback Loop**: Incorporação de feedback dos usuários

## 📊 CONCLUSÕES

### Sucessos Alcançados

✅ **Pipeline Funcional**: Execução completa sem erros críticos
✅ **Performance Excelente**: R² = 0.9262 indica modelo robusto
✅ **Otimização Automática**: Hiperparâmetros otimizados automaticamente
✅ **Validação Temporal**: Cross-validation adequada para séries temporais
✅ **Visualizações Completas**: Gráficos informativos gerados
✅ **Arquitetura Modular**: Código organizado e reutilizável

### Limitações Identificadas

⚠️ **Dados Limitados**: Apenas 10 anos de dados históricos
⚠️ **Variáveis Exógenas**: Seleção baseada em análise anterior
⚠️ **Configuração Rápida**: Otimização com apenas 5 trials
⚠️ **Validação Simples**: Apenas 3 splits de cross-validation

## 📁 ESTRUTURA DE ARQUIVOS GERADOS

```
artifacts/
├── criminal_cases/
│   ├── prophet_criminal_cases_model.pkl
│   ├── criminal_cases_forecast.csv
│   ├── criminal_cases_metrics.json
│   └── criminal_cases_config.yaml
└── plots/
    └── criminal_cases/
        ├── forecast_plot_20251029_020734.png
        ├── components_plot_20251029_020735.png
        └── residuals_plot_20251029_020735.png
```

## 🚀 COMO USAR O PIPELINE

### Execução Básica

```bash
python3 execute_pipeline.py
```

### Execução com Configuração Específica

```bash
python3 run_criminal_cases.py --config configs/criminal_cases_config.yaml
```

### Execução Completa

```bash
python3 main.py --config configs/criminal_cases_config.yaml
```

## 📞 SUPORTE E DOCUMENTAÇÃO

### Documentação Disponível

- **README.md**: Visão geral do projeto
- **QUICK_START.md**: Guia de início rápido
- **EXECUTION_GUIDE.md**: Guia de execução detalhado
- **PROJECT_SUMMARY.md**: Resumo completo do projeto

### Estrutura do Código

- **src/data/**: Módulo de gerenciamento de dados
- **src/models/**: Módulo de modelos Prophet
- **src/training/**: Módulo de treinamento
- **src/evaluation/**: Módulo de avaliação
- **src/utils/**: Módulo de utilitários

---

**Data de Execução**: 29 de Outubro de 2025, 02:07
**Status**: ✅ CONCLUÍDO COM SUCESSO
**Performance**: 🏆 EXCELENTE (R² = 0.9262)
