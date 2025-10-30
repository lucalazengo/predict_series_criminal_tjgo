# Prophet Forecasting Pipeline - Guia de Execução

## Visão Geral

Este pipeline automatizado para previsão de séries temporais utiliza o Facebook Prophet com suporte completo a variáveis exógenas, validação temporal, otimização de hiperparâmetros e geração de relatórios.

## Estrutura do Projeto

```
prophet_forecasting_pipeline/
├── src/                    # Código fonte modular
│   ├── data/              # Carregamento e pré-processamento de dados
│   ├── models/            # Wrapper do Prophet com variáveis exógenas
│   ├── training/          # Pipeline de treinamento e otimização
│   ├── evaluation/        # Avaliação e métricas
│   └── utils/             # Utilitários, relatórios e visualizações
├── tests/                 # Testes unitários
├── configs/               # Arquivos de configuração
├── outputs/               # Artefatos gerados
│   ├── models/           # Modelos treinados
│   ├── predictions/      # Previsões
│   └── reports/          # Relatórios e análises
├── logs/                  # Arquivos de log
├── main.py               # Script principal
├── example.py            # Exemplo de uso
├── requirements.txt      # Dependências
└── setup.py             # Configuração do pacote
```

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Instalar o Pacote (Opcional)

```bash
pip install -e .
```

## Configuração

### Arquivo de Configuração Principal

Edite `configs/default_config.yaml` para personalizar:

- **Caminhos dos dados**: Especifique os arquivos CSV de entrada
- **Parâmetros do modelo**: Configure Prophet e variáveis exógenas
- **Treinamento**: Defina validação cruzada e otimização
- **Avaliação**: Escolha métricas e análises
- **Saída**: Configure salvamento de artefatos

### Exemplo de Configuração Mínima

```yaml
data:
  target_series_path: "data/target.csv"
  exogenous_features_path: "data/features.csv"
  target_column: "TOTAL_CASOS"
  date_column: "DATA"

model:
  prophet_params:
    growth: "linear"
    yearly_seasonality: true
  exogenous_vars:
    enabled: true
    features: ["feature1", "feature2"]

training:
  cv:
    n_splits: 5
  hyperparameter_optimization:
    enabled: true
    n_trials: 50

output:
  base_dir: "outputs"
  save_model: true
  save_predictions: true
```

## Execução

### 1. Execução Básica

```bash
python main.py --config configs/default_config.yaml
```

### 2. Execução com Opções

```bash
python main.py \
  --config configs/default_config.yaml \
  --output-dir my_outputs \
  --verbose
```

### 3. Executar Exemplo

```bash
python example.py
```

### 4. Executar Testes

```bash
python -m pytest tests/ -v
```

## Formato dos Dados

### Dados da Série Temporal (Target)

```csv
DATA,TOTAL_CASOS
2020-01-01,1000
2020-02-01,1100
2020-03-01,1200
...
```

### Dados de Variáveis Exógenas

```csv
data,feature1,feature2,feature3
2020-01-01,50.5,20.3,100.1
2020-02-01,52.1,21.0,98.5
2020-03-01,49.8,19.7,102.3
...
```

## Funcionalidades Principais

### 1. Carregamento e Pré-processamento de Dados

- Carregamento automático de CSV
- Detecção e remoção de outliers
- Preenchimento de valores faltantes
- Criação de features de lag
- Validação de esquemas de dados

### 2. Modelo Prophet Aprimorado

- Suporte completo a variáveis exógenas
- Configuração de feriados
- Componentes sazonais personalizáveis
- Intervalos de predição

### 3. Validação Temporal

- Time Series Cross-Validation
- Métricas de avaliação abrangentes
- Análise de componentes do modelo
- Diagnósticos de resíduos

### 4. Otimização de Hiperparâmetros

- Integração com Optuna
- Otimização automática de parâmetros
- Validação temporal durante otimização

### 5. Relatórios e Visualizações

- Relatórios HTML interativos
- Gráficos de previsão
- Análise de componentes
- Métricas de performance

## Métricas de Avaliação

O pipeline calcula automaticamente:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **R²**: R-squared

## Artefatos Gerados

### Modelos
- Modelo Prophet treinado (`.joblib`)
- Configuração otimizada (`.yaml`)

### Previsões
- Previsões com intervalos de confiança (`.csv`)
- Métricas de avaliação (`.json`)

### Relatórios
- Relatório HTML interativo
- Gráficos de previsão (`.png`)
- Análise de componentes
- Diagnósticos de modelo

## Personalização Avançada

### Adicionando Novas Métricas

```python
# Em src/evaluation/metrics_calculator.py
def _calculate_custom_metric(self, actual, predicted):
    # Implementar nova métrica
    return custom_value
```

### Criando Visualizações Customizadas

```python
# Em src/utils/visualizer.py
def create_custom_plot(self, data, save_path=None):
    # Implementar nova visualização
    pass
```

### Configurando Feriados Customizados

```yaml
model:
  holidays:
    enabled: true
    country: "BR"
    custom_holidays:
      - name: "Feriado Customizado"
        date: "2024-01-01"
        lower_window: -1
        upper_window: 1
```

## Troubleshooting

### Problemas Comuns

1. **Erro de Importação**
   ```bash
   # Verificar se todas as dependências estão instaladas
   pip install -r requirements.txt
   ```

2. **Erro de Caminho de Dados**
   ```bash
   # Verificar se os arquivos existem e os caminhos estão corretos
   ls -la data/
   ```

3. **Erro de Memória**
   ```yaml
   # Reduzir número de trials na otimização
   training:
     hyperparameter_optimization:
       n_trials: 10
   ```

4. **Convergência Lenta**
   ```yaml
   # Ajustar parâmetros do Prophet
   model:
     prophet_params:
       changepoint_prior_scale: 0.01
       seasonality_prior_scale: 1.0
   ```

### Logs e Debugging

- Logs são salvos em `logs/prophet_pipeline.log`
- Use `--verbose` para logging detalhado
- Verifique os logs para identificar problemas

## Exemplos de Uso

### Exemplo 1: Previsão Simples

```python
from src.data import DataManager
from src.models import ProphetModelWrapper

# Carregar dados
data_manager = DataManager(config)
df, features = data_manager.load_and_prepare_data()

# Treinar modelo
model = ProphetModelWrapper(config)
model.fit(df, features)

# Fazer previsão
forecast = model.predict(df, horizon_months=12)
```

### Exemplo 2: Pipeline Completo

```python
from src.training import TrainingPipeline
from src.evaluation import ModelEvaluator

# Pipeline completo
training_pipeline = TrainingPipeline(config)
results = training_pipeline.run_training_pipeline(df, features)

# Avaliação
evaluator = ModelEvaluator(config)
evaluation = evaluator.evaluate_model(results['model'], forecast, df)
```

## Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente testes unitários
4. Execute `pytest` para verificar
5. Submeta um Pull Request

## Licença

Este projeto está licenciado sob a MIT License.

## Suporte

Para suporte e dúvidas:

- Abra uma issue no GitHub
- Consulte a documentação completa
- Verifique os logs de execução
