# 🔮 Pipeline de Previsão de Séries Temporais - Casos Criminais TJGO

Pipeline automatizado e reprodutível para previsão de séries temporais mensais utilizando **Facebook Prophet**, desenvolvido especificamente para análise e previsão de casos criminais do Tribunal de Justiça de Goiás (TJGO).

## 📋 Sumário

- [Características](#-características)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação Rápida](#-instalação-rápida)
- [Guia de Execução](#-guia-de-execução)
- [Configuração](#-configuração)
- [Resultados e Saídas](#-resultados-e-saídas)
- [Documentação](#-documentação)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Contribuindo](#-contribuindo)

---

## ✨ Características

### Funcionalidades Principais

- ✅ **Modelo Prophet Completo**: Implementação com suporte completo a variáveis exógenas
- ✅ **Validação Temporal**: Time Series Cross-Validation (TimeSeriesSplit) para evitar vazamento de dados
- ✅ **Otimização Automática**: Busca de hiperparâmetros usando Optuna (50+ trials)
- ✅ **Análise Detalhada de Features**: Importância relativa, correlações e contribuições
- ✅ **Métricas Completas**: MAE, RMSE, MAPE, SMAPE, R²
- ✅ **Visualizações**: Gráficos de previsão, componentes e análise de resíduos
- ✅ **Relatórios Detalhados**: HTML interativo e relatório técnico completo em Markdown
- ✅ **Gestão de Artefatos**: Salvamento automático de modelos, previsões e métricas
- ✅ **Design Modular**: Estrutura organizada em módulos (`src/`)
- ✅ **Documentação Completa**: Guias detalhados em português

### Melhorias Recentes (Outubro 2025)

- 🚀 Otimização robusta: 50 trials (aumentado de 5)
- 🚀 Validação expandida: 5 splits de cross-validation
- 🚀 Análise aprofundada de importância de variáveis exógenas
- 🚀 Relatório técnico detalhado com análises aprofundadas
- 🚀 Geração automática de relatórios após execução

---

## 📁 Estrutura do Projeto

```
predict_series_criminal_tjgo/
├── src/                          # Código fonte modular
│   ├── data/                    # Carregamento e pré-processamento de dados
│   │   └── __init__.py          # DataManager, DataProcessor, DataValidator
│   ├── models/                  # Wrapper do modelo Prophet
│   │   └── __init__.py          # ProphetModelWrapper, ProphetHyperparameterOptimizer
│   ├── training/                 # Pipeline de treinamento
│   │   └── __init__.py          # TrainingPipeline, TimeSeriesSplitter
│   ├── evaluation/              # Avaliação e métricas
│   │   └── __init__.py          # ModelEvaluator, FeatureImportanceAnalyzer, MetricsCalculator
│   └── utils/                   # Utilidades e relatórios
│       └── __init__.py          # ReportGenerator, Visualizer, ArtifactManager
│
├── configs/                      # Arquivos de configuração
│   ├── default_config.yaml      # Configuração padrão
│   └── criminal_cases_config.yaml  # Configuração específica para casos criminais
│
├── data/                         # Dados do projeto
│   ├── raw_data/                # Dados brutos
│   │   ├── casos_mensal__criminal_series_2015_2024.csv
│   │   └── external_features_2015_2024.csv
│   └── processed/               # Dados processados
│
├── outputs/                      # Resultados gerados
│   ├── models/                  # Modelos treinados (.joblib)
│   ├── predictions/             # Previsões (.csv)
│   └── reports/                 # Relatórios e visualizações
│       ├── *.html               # Relatórios HTML
│       ├── *.md                 # Relatórios Markdown
│       ├── *.png                # Gráficos
│       └── *.json               # Métricas e análises
│
├── logs/                        # Arquivos de log
│
├── tests/                       # Testes unitários
│
├── execute_pipeline.py          # Script principal de execução ⭐
├── generate_detailed_report.py  # Gerador de relatório técnico detalhado
├── main.py                      # Script alternativo de execução
│
├── requirements.txt             # Dependências Python
├── setup.py                     # Configuração do pacote
│
├── README.md                    # Este arquivo
├── GUIA_EXECUCAO.md            # Guia detalhado de execução 📘
├── DOCUMENTACAO_TECNICA.md     # Documentação técnica completa
├── RELATORIO_EXECUCAO_FINAL.md # Relatório de execução
└── RESUMO_IMPLEMENTACOES.md    # Resumo das implementações
```

---

## 🔧 Pré-requisitos

### Sistema

- **Python**: 3.8 ou superior
- **Sistema Operacional**: macOS, Linux ou Windows
- **Memória RAM**: Mínimo 4GB (recomendado 8GB+)
- **Espaço em Disco**: ~500MB

### Dependências Python

Todas as dependências estão listadas em `requirements.txt`:

```txt
pandas>=1.3.0
numpy>=1.21.0
prophet>=1.1.0
scikit-learn>=1.0.0
optuna>=3.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
pyyaml>=5.4.0
loguru>=0.6.0
joblib>=1.0.0
```

---

## 🚀 Instalação Rápida

### 1. Clonar/Baixar o Projeto

```bash
cd /caminho/para/projeto
```

### 2. Criar Ambiente Virtual (Recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# ou venv\Scripts\activate  # Windows
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Verificar Instalação

```bash
python3 -c "import pandas, numpy, prophet; print('✅ Dependências instaladas!')"
```

---

## 📖 Guia de Execução

### Execução Básica (Recomendada)

```bash
python3 execute_pipeline.py
```

Este comando executa o pipeline completo:
1. ✅ Carrega e prepara os dados
2. ✅ Valida a qualidade dos dados
3. ✅ Treina o modelo Prophet com otimização de hiperparâmetros (50 trials)
4. ✅ Gera previsões futuras
5. ✅ Calcula todas as métricas de avaliação
6. ✅ Analisa importância das variáveis exógenas
7. ✅ Gera visualizações (gráficos)
8. ✅ Salva todos os artefatos
9. ✅ Gera relatório HTML
10. ✅ Gera relatório técnico detalhado em Markdown

**⏱️ Tempo Estimado:** 30-60 minutos (com 50 trials)

### Execução Rápida (Para Testes)

1. Editar `configs/criminal_cases_config.yaml`:
```yaml
training:
  hyperparameter_optimization:
    n_trials: 5  # Reduzir para execução mais rápida
```

2. Executar:
```bash
python3 execute_pipeline.py
```

**⏱️ Tempo Estimado:** 5-10 minutos

### Gerar Apenas Relatório Detalhado

Se você já executou o pipeline e quer gerar apenas o relatório detalhado:

```bash
python3 generate_detailed_report.py
```

### Documentação Detalhada

Para instruções completas e detalhadas, consulte:
- **[GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)** - Guia passo a passo completo

---

## ⚙️ Configuração

### Arquivo de Configuração Principal

`configs/criminal_cases_config.yaml`

### Parâmetros Principais

```yaml
# Otimização de Hiperparâmetros
training:
  hyperparameter_optimization:
    enabled: true
    n_trials: 50              # Número de tentativas (aumentar = mais robustez, mais tempo)
    
# Validação Cruzada
training:
  cv:
    n_splits: 5               # Número de folds para TimeSeriesSplit
    
# Horizonte de Previsão
forecasting:
  horizon_months: 12          # Meses futuros para prever
  
# Métricas de Avaliação
evaluation:
  metrics:
    - "mae"                   # Mean Absolute Error
    - "rmse"                  # Root Mean Squared Error
    - "mape"                  # Mean Absolute Percentage Error
    - "smape"                 # Symmetric MAPE
    - "r2"                    # R-squared
```

### Variáveis Exógenas

O modelo utiliza as seguintes variáveis exógenas (configuradas em `criminal_cases_config.yaml`):

- `atendimento_pre_hospitalar`
- `pessoa_localizada`
- `lesao_corporal_seguida_de_morte`
- `tentativa_de_feminicidio`
- `morte_de_agente_do_estado`
- `suicidio_de_agente_do_estado`

---

## 📊 Resultados e Saídas

### Estrutura de Saída

```
outputs/
├── models/
│   └── prophet_model_YYYYMMDD_HHMMSS.joblib      # Modelo treinado
│
├── predictions/
│   └── forecast_YYYYMMDD_HHMMSS.csv              # Previsões futuras
│
└── reports/
    ├── forecast_plot_YYYYMMDD_HHMMSS.png          # Gráfico de previsão
    ├── components_plot_YYYYMMDD_HHMMSS.png       # Decomposição da série
    ├── residuals_plot_YYYYMMDD_HHMMSS.png         # Análise de resíduos
    ├── metrics_YYYYMMDD_HHMMSS.json                # Métricas calculadas
    ├── feature_analysis_YYYYMMDD_HHMMSS.json      # Análise de features
    ├── config_YYYYMMDD_HHMMSS.yaml                # Configuração usada
    ├── report_YYYYMMDD_HHMMSS.html                # Relatório HTML
    └── RELATORIO_DETALHADO_COMPLETO_*.md          # Relatório técnico detalhado
```

### Métricas de Performance

Os resultados típicos obtidos:

| Métrica | Valor | Classificação |
|---------|-------|---------------|
| **MAE** | ~648 casos | Excelente |
| **RMSE** | ~831 casos | Excelente |
| **MAPE** | ~8.08% | Excelente (< 10%) |
| **SMAPE** | ~8.62% | Excelente |
| **R²** | ~0.9695 | Excelente (> 0.90) |

**Interpretação:**
- **R² = 0.9695**: O modelo explica **96.95%** da variância dos dados
- **MAPE = 8.08%**: Erro percentual muito baixo (< 10% é excelente)
- **MAE = 647.95**: Erro absoluto médio de aproximadamente 648 casos/mês

### Visualizações Geradas

1. **Gráfico de Previsão**: Série temporal real vs prevista com intervalos de confiança
2. **Gráfico de Componentes**: Decomposição em tendência, sazonalidade e efeitos exógenos
3. **Gráfico de Resíduos**: Análise dos erros de previsão

### Relatórios

1. **Relatório HTML**: Relatório interativo com todas as métricas e visualizações incorporadas
2. **Relatório Markdown Detalhado**: Relatório técnico completo com:
   - Análise estatística dos dados
   - Análise detalhada de cada métrica
   - Análise dos componentes do modelo
   - Diagnóstico completo do modelo
   - Análise de importância de features
   - Conclusões e recomendações

---

## 📚 Documentação

### Documentos Disponíveis

1. **[GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)** ⭐
   - Guia passo a passo de instalação e execução
   - Troubleshooting completo
   - Exemplos de uso
   - Checklist de execução

2. **[DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)**
   - Documentação técnica detalhada
   - Arquitetura do sistema
   - Descrição de módulos
   - API e interfaces

3. **[RELATORIO_EXECUCAO_FINAL.md](RELATORIO_EXECUCAO_FINAL.md)**
   - Relatório de execução do pipeline
   - Resultados obtidos
   - Análise dos resultados

4. **[RESUMO_IMPLEMENTACOES.md](RESUMO_IMPLEMENTACOES.md)**
   - Resumo de todas as implementações
   - Status das funcionalidades
   - Melhorias realizadas

### Comentários no Código

Todo o código está documentado em português com:
- Docstrings detalhadas
- Comentários explicativos
- Type hints
- Exemplos de uso

---

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Principais

- **Prophet**: Modelo de previsão de séries temporais do Facebook
- **Optuna**: Framework de otimização de hiperparâmetros
- **Scikit-learn**: Machine learning e validação cruzada
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Matplotlib/Seaborn/Plotly**: Visualizações

### Estrutura e Padrões

- **Design Modular**: Separação clara de responsabilidades
- **Type Hints**: Anotação de tipos para melhor manutenibilidade
- **Loguru**: Sistema de logging robusto
- **YAML**: Configurações em formato legível
- **Joblib**: Serialização de modelos

---

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/

# Executar testes com cobertura
python -m pytest tests/ --cov=src
```

---

## 🔍 Troubleshooting

### Problemas Comuns

1. **Erro de Importação**
   ```bash
   pip install -r requirements.txt
   ```

2. **Arquivos de Dados Não Encontrados**
   - Verificar se os arquivos existem em `data/raw_data/`
   - Verificar nomes dos arquivos na configuração

3. **Execução Muito Lenta**
   - Reduzir `n_trials` na configuração
   - Verificar recursos do sistema (RAM, CPU)

### Mais Informações

Consulte a seção **Troubleshooting** em [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) para soluções detalhadas.

---

## 📈 Performance Esperada

### Resultados Típicos

Com a configuração padrão (50 trials, 5 splits):

- **R²**: > 0.90 (Excelente)
- **MAPE**: < 10% (Excelente)
- **Tempo de Execução**: 30-60 minutos
- **Arquivos Gerados**: ~10 arquivos (modelo, previsões, relatórios, visualizações)

### Benchmark

| Configuração | Trials | Tempo | R² Esperado |
|--------------|--------|-------|-------------|
| Rápida | 5 | ~10 min | > 0.85 |
| Padrão | 50 | ~45 min | > 0.90 |
| Extensa | 100+ | ~2h | > 0.92 |

---

## 🤝 Contribuindo

### Estrutura de Desenvolvimento

1. Criar branch para nova feature
2. Desenvolver e testar
3. Documentar alterações
4. Submeter pull request

### Padrões de Código

- Seguir estrutura modular em `src/`
- Manter documentação atualizada
- Adicionar testes para novas funcionalidades
- Usar type hints

---

## 📝 Licença

Este projeto foi desenvolvido para análise de casos criminais do TJGO.

---

## 👥 Autores

Desenvolvido como parte do projeto de Residência em TI - TJGO.

---

## 📞 Suporte

Para dúvidas ou problemas:

1. Consultar [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)
2. Verificar logs em `logs/prophet_pipeline.log`
3. Revisar documentação técnica

---

## 🎯 Próximos Passos

Após executar o pipeline:

1. ✅ Analisar relatório HTML gerado
2. ✅ Revisar relatório técnico detalhado
3. ✅ Examinar visualizações
4. ✅ Ajustar configuração se necessário
5. ✅ Retreinar com novos dados periodicamente

---

## 🌟 Destaques

- 🏆 **Performance Excelente**: R² > 0.96, MAPE < 10%
- 🎯 **Análise Detalhada**: Relatórios técnicos completos e aprofundados
- 🔬 **Otimização Robusta**: 50+ trials para encontrar melhores hiperparâmetros
- 📊 **Visualizações Completas**: Gráficos interativos e informativos
- 📚 **Documentação Completa**: Guias detalhados em português
- 🔄 **Reprodutível**: Configuração versionada e artefatos salvos

---

**Última atualização:** Outubro 2025

**Versão:** 2.0.0
