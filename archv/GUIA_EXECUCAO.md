# 📘 GUIA DETALHADO DE EXECUÇÃO - PIPELINE PROPHET CASOS CRIMINAIS TJGO

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Configuração](#configuração)
4. [Execução Básica](#execução-básica)
5. [Execução Avançada](#execução-avançada)
6. [Interpretação dos Resultados](#interpretação-dos-resultados)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Pré-requisitos

### Requisitos do Sistema

- **Python**: 3.8 ou superior
- **Sistema Operacional**: macOS, Linux ou Windows
- **Memória RAM**: Mínimo 4GB (recomendado 8GB+)
- **Espaço em Disco**: ~500MB para dados e resultados

### Dependências Python

O projeto requer as seguintes bibliotecas principais:
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `prophet` >= 1.1.0
- `scikit-learn` >= 1.0.0
- `optuna` >= 3.0.0
- `matplotlib` >= 3.4.0
- `seaborn` >= 0.11.0
- `plotly` >= 5.0.0
- `pyyaml` >= 5.4.0
- `loguru` >= 0.6.0

---

## 📥 Instalação

### Passo 1: Clonar/Baixar o Projeto

Se você já tem o projeto localmente, pule para o Passo 2.

```bash
# Se usando Git
git clone <url-do-repositorio>
cd predict_series_criminal_tjgo

# Ou navegue até o diretório do projeto
cd /caminho/para/predict_series_criminal_tjgo
```

### Passo 2: Criar Ambiente Virtual (Recomendado)

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No macOS/Linux:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### Passo 3: Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Ou instalar individualmente (se necessário)
pip install pandas numpy prophet scikit-learn optuna matplotlib seaborn plotly pyyaml loguru
```

### Passo 4: Verificar Instalação

```bash
# Verificar versão do Python
python3 --version  # Deve ser 3.8 ou superior

# Testar importação das bibliotecas principais
python3 -c "import pandas, numpy, prophet; print('Dependências instaladas com sucesso!')"
```

---

## ⚙️ Configuração

### Estrutura de Arquivos Necessários

Certifique-se de que os seguintes arquivos de dados existem:

```
data/
└── raw_data/
    ├── casos_mensal__criminal_series_2015_2024.csv
    └── external_features_2015_2024.csv
```

### Verificar Dados

```bash
# Verificar se os arquivos de dados existem
ls -la data/raw_data/

# Verificar conteúdo (primeiras linhas)
head -5 data/raw_data/casos_mensal__criminal_series_2015_2024.csv
head -5 data/raw_data/external_features_2015_2024.csv
```

### Configuração do Pipeline

O arquivo de configuração principal está em `configs/criminal_cases_config.yaml`.

**Parâmetros Principais:**

```yaml
# Número de trials para otimização (padrão: 50)
training:
  hyperparameter_optimization:
    n_trials: 50  # Aumente para mais robustez (pode levar mais tempo)

# Splits de validação cruzada
training:
  cv:
    n_splits: 5  # Número de folds para TimeSeriesSplit

# Horizonte de previsão
forecasting:
  horizon_months: 12  # Meses futuros para prever
```

**⚠️ Nota:** Aumentar `n_trials` melhora a qualidade da otimização, mas aumenta significativamente o tempo de execução.

---

## 🚀 Execução Básica

### Método 1: Execução Direta (Recomendado)

```bash
# Executar o pipeline completo
python3 execute_pipeline.py
```

Este script:
1. Carrega os dados
2. Prepara e valida os dados
3. Treina o modelo Prophet com otimização de hiperparâmetros
4. Gera previsões
5. Calcula métricas de avaliação
6. Gera visualizações
7. Salva todos os artefatos
8. Gera relatório HTML e relatório técnico detalhado

### Método 2: Usando o Script Principal

```bash
python3 main.py --config configs/criminal_cases_config.yaml
```

### Tempo de Execução Esperado

- **Execução Rápida (5 trials)**: ~5-10 minutos
- **Execução Completa (50 trials)**: ~30-60 minutos
- **Execução Extensa (100+ trials)**: 1-3 horas

---

## 🎯 Execução Avançada

### Executar Apenas o Gerador de Relatórios

Se você já executou o pipeline e quer gerar apenas o relatório detalhado:

```bash
python3 generate_detailed_report.py
```

O script automaticamente encontra os arquivos mais recentes em `outputs/reports/`.

### Personalizar Configuração

1. **Editar configuração:**

```bash
# Editar o arquivo de configuração
nano configs/criminal_cases_config.yaml
# ou
vim configs/criminal_cases_config.yaml
```

2. **Principais parâmetros para ajustar:**

```yaml
# Reduzir trials para execução mais rápida
training:
  hyperparameter_optimization:
    n_trials: 10  # Reduzido de 50 para 10

# Aumentar horizonte de previsão
forecasting:
  horizon_months: 24  # Prever 2 anos à frente
```

3. **Executar com configuração personalizada:**

```bash
python3 execute_pipeline.py
# O script usa automaticamente configs/criminal_cases_config.yaml
```

### Executar com Dados Diferentes

1. **Preparar seus dados:**
   - Formato CSV com coluna de data e coluna de valores
   - Formato de data: `YYYY-MM-DD` ou similar

2. **Atualizar configuração:**

```yaml
data:
  target_series_path: "data/raw_data/seu_arquivo.csv"
  target_column: "SUA_COLUNA_ALVO"
  date_column: "SUA_COLUNA_DATA"
```

3. **Executar normalmente:**

```bash
python3 execute_pipeline.py
```

---

## 📊 Interpretação dos Resultados

### Estrutura de Saída

Após a execução, você encontrará:

```
outputs/
├── models/
│   └── prophet_model_YYYYMMDD_HHMMSS.joblib
├── predictions/
│   └── forecast_YYYYMMDD_HHMMSS.csv
└── reports/
    ├── forecast_plot_YYYYMMDD_HHMMSS.png
    ├── components_plot_YYYYMMDD_HHMMSS.png
    ├── residuals_plot_YYYYMMDD_HHMMSS.png
    ├── metrics_YYYYMMDD_HHMMSS.json
    ├── feature_analysis_YYYYMMDD_HHMMSS.json
    ├── config_YYYYMMDD_HHMMSS.yaml
    ├── report_YYYYMMDD_HHMMSS.html
    └── RELATORIO_DETALHADO_COMPLETO_YYYYMMDD_HHMMSS.md
```

### Métricas Principais

Os resultados são salvos em `outputs/reports/metrics_*.json`:

```json
{
  "mae": 647.95,      // Erro Absoluto Médio (menor é melhor)
  "rmse": 831.30,     // Raiz do Erro Quadrático Médio (menor é melhor)
  "mape": 8.08,       // Erro Percentual Absoluto Médio (menor é melhor)
  "smape": 8.62,      // Erro Percentual Simétrico (menor é melhor)
  "r2": 0.9695        // Coeficiente de Determinação (maior é melhor, máximo 1.0)
}
```

**Interpretação Rápida:**
- **R² > 0.90**: Excelente! Modelo explica mais de 90% da variância
- **MAPE < 10%**: Excelente precisão percentual
- **MAE e RMSE**: Avaliar em relação à média da série (quanto menor, melhor)

### Visualizações

1. **forecast_plot_*.png**: Gráfico de previsão com dados reais vs previstos
2. **components_plot_*.png**: Decomposição da série (tendência, sazonalidade)
3. **residuals_plot_*.png**: Análise de resíduos (erros do modelo)

### Relatórios

1. **report_*.html**: Relatório HTML interativo com todas as métricas e visualizações
2. **RELATORIO_DETALHADO_COMPLETO_*.md**: Relatório técnico completo em Markdown com análises aprofundadas

---

## 🔍 Troubleshooting

### Erro: "ModuleNotFoundError"

```bash
# Problema: Biblioteca não encontrada
# Solução: Instalar dependências
pip install -r requirements.txt
```

### Erro: "FileNotFoundError: data/raw_data/..."

```bash
# Problema: Arquivos de dados não encontrados
# Solução: Verificar estrutura de diretórios
ls -la data/raw_data/

# Criar diretórios se não existirem
mkdir -p data/raw_data
```

### Erro: "MemoryError" durante otimização

```bash
# Problema: Muitos trials consumindo muita memória
# Solução: Reduzir número de trials na configuração
# Editar configs/criminal_cases_config.yaml
training:
  hyperparameter_optimization:
    n_trials: 20  # Reduzir de 50
```

### Execução muito lenta

```bash
# Problema: Otimização com muitos trials
# Solução 1: Reduzir trials
# Solução 2: Executar em servidor com mais recursos
# Solução 3: Usar configuração rápida para testes
```

### Erro: "Prophet raised an exception"

```bash
# Problema: Erro no modelo Prophet
# Solução: Verificar dados
# - Coluna de data está no formato correto?
# - Há valores faltantes?
# - Há valores infinitos ou NaN?

python3 -c "
import pandas as pd
df = pd.read_csv('data/raw_data/casos_mensal__criminal_series_2015_2024.csv')
print(df.info())
print(df.isnull().sum())
"
```

### Avisos sobre Encoding

```bash
# Problema: Encoding de caracteres especiais
# Solução: Os scripts já incluem # -*- coding: utf-8 -*-
# Se persistir, verificar encoding dos arquivos CSV
file -I data/raw_data/*.csv
```

### Não encontra análise de features

```bash
# Problema: Arquivo feature_analysis não gerado
# Solução: Verificar se a execução completou com sucesso
# A análise de features é gerada automaticamente durante a avaliação
```

---

## 📚 Comandos Úteis

### Verificar Status da Execução

```bash
# Ver logs mais recentes
tail -f logs/prophet_pipeline.log

# Ver arquivos gerados
ls -lth outputs/reports/ | head -10
```

### Limpar Resultados Anteriores

```bash
# Limpar apenas modelos antigos (manter dados)
find outputs/models -name "*.joblib" -mtime +7 -delete

# Limpar todos os outputs (CUIDADO!)
rm -rf outputs/models/* outputs/predictions/* outputs/reports/*
```

### Validar Dados Antes de Executar

```bash
# Verificar estrutura dos dados
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw_data/casos_mensal__criminal_series_2015_2024.csv')
print('Linhas:', len(df))
print('Colunas:', df.columns.tolist())
print('Período:', df['DATA'].min(), 'a', df['DATA'].max())
print('Total casos:', df['TOTAL_CASOS'].sum())
"
```

---

## 🎓 Exemplos de Uso

### Exemplo 1: Execução Rápida para Testes

1. Editar `configs/criminal_cases_config.yaml`:
```yaml
training:
  hyperparameter_optimization:
    n_trials: 5  # Reduzido para execução rápida
```

2. Executar:
```bash
python3 execute_pipeline.py
```

### Exemplo 2: Execução Completa com Otimização Robusta

1. Manter configuração padrão (50 trials)

2. Executar:
```bash
python3 execute_pipeline.py
```

3. Aguardar conclusão (~30-60 minutos)

### Exemplo 3: Gerar Apenas Relatório Detalhado

```bash
# Executar pipeline completo primeiro
python3 execute_pipeline.py

# Depois, gerar apenas relatório (se necessário)
python3 generate_detailed_report.py
```

---

## ✅ Checklist de Execução

Antes de executar, verifique:

- [ ] Python 3.8+ instalado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivos de dados existem em `data/raw_data/`
- [ ] Configuração atualizada (se necessário)
- [ ] Espaço em disco suficiente (~500MB)

Durante a execução:

- [ ] Pipeline inicia sem erros
- [ ] Dados são carregados corretamente
- [ ] Modelo treina sem erros
- [ ] Previsões são geradas
- [ ] Arquivos são salvos em `outputs/`

Após a execução:

- [ ] Relatório HTML gerado
- [ ] Relatório Markdown detalhado gerado
- [ ] Visualizações geradas (3 gráficos)
- [ ] Métricas salvas em JSON
- [ ] Modelo salvo em `.joblib`

---

## 📞 Suporte

Se encontrar problemas:

1. Verificar seções de Troubleshooting acima
2. Consultar logs em `logs/prophet_pipeline.log`
3. Verificar documentação técnica em `DOCUMENTACAO_TECNICA.md`
4. Revisar exemplos neste guia

---

**Última atualização:** Outubro 2025



