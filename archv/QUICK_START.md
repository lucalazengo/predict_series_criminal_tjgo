# 🚀 Prophet Forecasting Pipeline - Guia Rápido

## ⚡ Execução Rápida

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Pipeline Completo
```bash
python run_criminal_cases.py
```

### 3. Execução Rápida (Menos Otimização)
```bash
python run_criminal_cases.py --quick
```

### 4. Executar com Logs Detalhados
```bash
python run_criminal_cases.py --verbose
```

## 📁 Estrutura do Projeto

```
prophet_forecasting_pipeline/
├── src/                    # Código modular
│   ├── data/              # Carregamento de dados
│   ├── models/            # Modelo Prophet
│   ├── training/          # Treinamento e otimização
│   ├── evaluation/        # Avaliação e métricas
│   └── utils/             # Relatórios e visualizações
├── configs/               # Configurações
├── outputs/               # Resultados gerados
├── tests/                 # Testes unitários
├── main.py               # Pipeline genérico
├── run_criminal_cases.py # Pipeline específico
└── example.py            # Exemplo de uso
```

## 🎯 Funcionalidades Principais

✅ **Carregamento Automático de Dados**
- CSV com séries temporais e variáveis exógenas
- Pré-processamento automático
- Validação de dados

✅ **Modelo Prophet Aprimorado**
- Suporte a variáveis exógenas
- Feriados brasileiros
- Componentes sazonais

✅ **Validação Temporal**
- Time Series Cross-Validation
- Métricas abrangentes (MAE, RMSE, MAPE, R²)
- Análise de componentes

✅ **Otimização Automática**
- Hiperparâmetros com Optuna
- Validação temporal durante otimização

✅ **Relatórios Completos**
- HTML interativo
- Visualizações automáticas
- Métricas de performance

## 📊 Dados Utilizados

### Série Temporal Principal
- **Arquivo**: `data/raw_data/casos_mensal__criminal_series_2015_2024.csv`
- **Período**: 2015-2024 (120 meses)
- **Variável**: TOTAL_CASOS

### Variáveis Exógenas Selecionadas
Baseadas na análise de seleção de características:

1. `atendimento_pre_hospitalar` (atual e lags 1-3)
2. `pessoa_localizada_lag_3`
3. `lesao_corporal_seguida_de_morte` (atual e lag 2)
4. `tentativa_de_feminicidio_lag_2`
5. `morte_de_agente_do_estado_lag_1`
6. `suicidio_de_agente_do_estado`

## 🔧 Configuração

### Arquivo Principal
`configs/criminal_cases_config.yaml`

### Principais Configurações
- **Horizonte de Previsão**: 12 meses
- **Validação Cruzada**: 5 splits
- **Otimização**: 50 trials
- **Métricas**: MAE, RMSE, MAPE, R²

## 📈 Resultados Esperados

### Artefatos Gerados
- **Modelo**: `outputs/models/prophet_model_*.joblib`
- **Previsões**: `outputs/predictions/forecast_*.csv`
- **Métricas**: `outputs/reports/metrics_*.json`
- **Relatório**: `outputs/reports/prophet_report_*.html`

### Visualizações
- Gráfico de previsão
- Análise de componentes
- Diagnósticos de resíduos

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/ -v

# Executar teste específico
python -m pytest tests/test_pipeline.py::TestDataModule -v
```

## 📝 Exemplo de Uso

```python
from src.data import DataManager
from src.training import TrainingPipeline

# Carregar dados
data_manager = DataManager(config)
df, features = data_manager.load_and_prepare_data()

# Treinar modelo
pipeline = TrainingPipeline(config)
results = pipeline.run_training_pipeline(df, features)

# Fazer previsão
forecast = results['model'].predict(df, horizon_months=12)
```

## 🚨 Troubleshooting

### Problemas Comuns

1. **Erro de Importação**
   ```bash
   pip install -r requirements.txt
   ```

2. **Arquivo de Dados Não Encontrado**
   - Verificar caminhos em `configs/criminal_cases_config.yaml`
   - Confirmar que os arquivos existem

3. **Memória Insuficiente**
   - Usar `--quick` para menos otimização
   - Reduzir `n_trials` na configuração

4. **Convergência Lenta**
   - Ajustar `changepoint_prior_scale` para 0.01
   - Reduzir `seasonality_prior_scale` para 1.0

### Logs
- **Arquivo**: `logs/prophet_pipeline.log`
- **Nível**: INFO (use `--verbose` para DEBUG)

## 📚 Documentação Completa

- **Guia Detalhado**: `EXECUTION_GUIDE.md`
- **README**: `README.md`
- **Exemplo**: `example.py`

## 🎉 Próximos Passos

1. **Execute o Pipeline**
   ```bash
   python run_criminal_cases.py
   ```

2. **Analise os Resultados**
   - Abra o relatório HTML
   - Verifique as métricas de performance
   - Analise as visualizações

3. **Ajuste se Necessário**
   - Modifique a configuração
   - Adicione novas variáveis exógenas
   - Ajuste parâmetros do Prophet

4. **Integre com Produção**
   - Use o modelo treinado
   - Automatize a execução
   - Monitore performance

---

**Desenvolvido com ❤️ para previsão de séries temporais criminais**
