# Documentação da Implementação SARIMAX - CRISP-DM

## Resumo Executivo

Esta implementação segue rigorosamente a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining) para criar um pipeline completo de previsão de séries temporais usando **SARIMAX** com `pmdarima` (auto_arima).

**Princípio Fundamental**: Todas as decisões de modelagem são baseadas em evidências estatísticas dos dados, garantindo que o modelo seja adequado e validado.

---

## Fase 1: Business Understanding ✅

### Objetivo do Negócio
Prever casos criminais mensais do Tribunal de Justiça de Goiás (TJGO) utilizando modelo estatístico SARIMAX, que é apropriado para:
- Séries temporais com padrões sazonais
- Incorporação de variáveis exógenas
- Modelagem de dependências temporais complexas

### Contexto Técnico
- **Modelo**: SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
- **Ferramenta**: `pmdarima.auto_arima` para busca automática de hiperparâmetros
- **Metodologia**: CRISP-DM para garantir qualidade e reprodutibilidade

---

## Fase 2: Data Understanding ✅

### Módulo: `data_exploration.py`

Análise exploratória profunda que fundamenta todas as decisões:

#### 2.1 Estatísticas Descritivas
- Média, mediana, desvio padrão
- Coeficiente de variação (CV) - indica variabilidade
- Assimetria e curtose - distribuição dos dados
- **Decisão**: Determina transformações necessárias

#### 2.2 Testes de Estacionariedade
**Testes Implementados**:
- **ADF (Augmented Dickey-Fuller)**: Testa se série tem raiz unitária
- **KPSS**: Testa se série é estacionária em torno de tendência

**Decisão Baseada em Dados**:
```python
# Se ADF p-value > 0.05 OU KPSS p-value < 0.05:
# → Série NÃO é estacionária
# → Necessária diferenciação (d > 0)
# → auto_arima testa d=0,1,2 automaticamente
```

#### 2.3 Decomposição Sazonal
**Análises**:
- Decomposição aditiva e multiplicativa
- Cálculo da força da sazonalidade
- Comparação de resíduos entre modelos

**Decisão Baseada em Dados**:
```python
seasonal_strength = var(seasonal) / (var(seasonal) + var(residual))

if seasonal_strength > 0.64:
    # Sazonalidade FORTE
    seasonal = True, m = 12
elif seasonal_strength > 0.36:
    # Sazonalidade MODERADA
    seasonal = True, testar m = 12
else:
    # Sazonalidade FRACA
    # Considerar ARIMA simples (seasonal=False)
```

#### 2.4 Análise de Autocorrelação
**Análises**:
- ACF (Autocorrelation Function)
- PACF (Partial Autocorrelation Function)

**Decisão Baseada em Dados**:
- Identifica padrões AR (via PACF)
- Identifica padrões MA (via ACF)
- Detecta período sazonal (ex: lag 12 em dados mensais)

#### 2.5 Análise de Variáveis Exógenas
**Análises**:
- Correlação de Pearson com variável alvo
- Ranking por correlação absoluta
- Seleção baseada em |correlação| > 0.3

**Decisão Baseada em Dados**:
```python
# Features com correlação significativa são selecionadas
# para incluir no modelo SARIMAX como variáveis exógenas
```

---

## Fase 3: Data Preparation ✅

### Módulo: `data_preparation.py`

#### 3.1 Carregamento e Merge
- Carrega série alvo e variáveis exógenas
- Merge por data com validação
- Ordenação temporal garantida

#### 3.2 Limpeza de Dados
**Operações**:
- Remoção de outliers (opcional, configurável)
- Preenchimento de valores faltantes (forward fill por padrão)
- Validação de integridade

**Decisão**: Para SARIMAX, outliers podem ser informativos, então remoção é opcional.

#### 3.3 Criação de Features de Lag
**Decisão Baseada em Dados**:
```python
# Análise exploratória identifica dependências temporais
# Cria lags até max_lags períodos
# Captura efeitos defasados de variáveis exógenas
```

#### 3.4 Preparação para SARIMAX
- Extrai série alvo como `pd.Series` com índice temporal
- Prepara variáveis exógenas como `pd.DataFrame`
- Garante alinhamento temporal

---

## Fase 4: Modeling ✅

### Módulo: `sarimax_model.py`

#### 4.1 Busca Automática com auto_arima

**Configuração Baseada em Análise Exploratória**:

```python
auto_arima(
    y,                          # Série alvo
    exogenous=exog,            # Variáveis exógenas (se disponíveis)
    max_p=5, max_d=2, max_q=5,  # Ordens máximas ARIMA
    max_P=2, max_D=1, max_Q=2, # Ordens máximas sazonais
    seasonal=True,             # Baseado em decomposição sazonal
    m=12,                       # Período sazonal (mensal)
    information_criterion='aicc',  # Para amostras pequenas
    stepwise=True,              # Busca eficiente
    test='adf',                 # Teste de estacionariedade
    ...
)
```

**Decisões Automáticas do auto_arima**:
1. Testa diferentes combinações de (p,d,q)(P,D,Q)s
2. Seleciona melhor por critério de informação (AICc)
3. Considera resultados dos testes de estacionariedade
4. Otimiza para melhor ajuste estatístico

#### 4.2 Modelo SARIMAX Resultante

O modelo treinado contém:
- **Ordem não-sazonal**: (p, d, q)
- **Ordem sazonal**: (P, D, Q, s)
- **Parâmetros estimados**: Coeficientes AR, MA, e exógenos
- **Estatísticas do modelo**: AIC, AICc, BIC, log-likelihood

---

## Fase 5: Evaluation ✅

### Módulo: `evaluation.py`

#### 5.1 Métricas de Precisão

**Métricas Calculadas**:
- **MAE**: Erro absoluto médio
- **RMSE**: Raiz do erro quadrático médio
- **MAPE**: Erro percentual absoluto médio
- **SMAPE**: Erro percentual simétrico médio
- **R²**: Coeficiente de determinação

**Interpretação**:
- RMSE e MAE em unidades da variável alvo
- MAPE e SMAPE em percentual (ideal < 10% para séries temporais)
- R² próximo de 1 indica bom ajuste

#### 5.2 Diagnósticos de Resíduos

**Teste de Ljung-Box**:
```python
# H0: Resíduos são ruído branco (não autocorrelacionados)
# p-value > 0.05: Resíduos são ruído branco (IDEAL)
# p-value < 0.05: Há autocorrelação (modelo pode ser melhorado)
```

**Teste de Normalidade**:
```python
# Jarque-Bera (amostras grandes) ou Shapiro-Wilk (amostras pequenas)
# H0: Resíduos seguem distribuição normal
# p-value > 0.05: Normalidade confirmada (IDEAL)
```

**Teste de Heterocedasticidade**:
```python
# Compara variâncias em diferentes períodos
# Variância constante é ideal (homoskedasticidade)
```

#### 5.3 Validação Temporal

**Time Series Cross-Validation**:
- Respeita ordem temporal
- Evita vazamento de dados
- Múltiplas divisões treino/teste
- Agrega métricas por split

---

## Fase 6: Deployment ✅

### Módulo: `pipeline.py` + `run_sarimax.py`

#### 6.1 Pipeline Orquestrado

O pipeline executa automaticamente:
1. Data Understanding (exploração)
2. Data Preparation
3. Modeling (treinamento)
4. Evaluation (avaliação)
5. Geração de previsões
6. Salvamento de artefatos

#### 6.2 Artefatos Gerados

1. **Modelo Treinado**: `.joblib` para reutilização
2. **Previsões**: CSV com previsões e intervalos de confiança
3. **Métricas**: JSON com todas as métricas calculadas
4. **Configuração**: YAML usado na execução
5. **Relatório Exploratório**: Markdown com análise detalhada

---

## Validação das Decisões

### Como Validar se o Modelo Performou Bem

#### 1. Critérios Estatísticos

**✅ Resíduos são Ruído Branco**:
```python
# Teste de Ljung-Box
p_value > 0.05  # ✓ Modelo capturou padrões
```

**✅ Resíduos são Normais**:
```python
# Teste de normalidade
p_value > 0.05  # ✓ Suposições do modelo atendidas
```

**✅ AIC/BIC Aceitáveis**:
```python
# Comparar com modelos alternativos
# Menor AIC/BIC geralmente indica melhor modelo
```

#### 2. Critérios de Precisão

**✅ MAPE < 15%**:
```python
MAPE < 15  # Bom para séries temporais
MAPE < 10  # Excelente
```

**✅ R² > 0.7**:
```python
R² > 0.7  # Bom ajuste
R² > 0.9  # Excelente ajuste
```

#### 3. Critérios de Negócio

**✅ Previsões Realistas**:
- Valores dentro de faixas esperadas
- Tendências consistentes com histórico
- Sazonalidade preservada

**✅ Intervalos de Confiança Apropriados**:
- 95% CI não muito amplos (modelo confiável)
- Cobertura real próxima de 95%

---

## Exemplo de Validação

Após executar o pipeline:

```python
results = pipeline.run_full_pipeline()

# Verifica métricas
metrics = results['evaluation']['metrics']
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"R²: {metrics['r2']:.4f}")

# Verifica diagnósticos
residuals_analysis = results['evaluation']['residual_analysis']

# Ljung-Box
if residuals_analysis['ljung_box']['is_white_noise']:
    print("✓ Resíduos são ruído branco (modelo adequado)")
else:
    print("✗ Resíduos têm autocorrelação (considerar melhorias)")

# Normalidade
if residuals_analysis['normality']['is_normal']:
    print("✓ Resíduos são normais (suposições atendidas)")
else:
    print("✗ Resíduos não são normais (considerar transformações)")
```

---

## Decisões Alinhadas aos Dados - Resumo

| Decisão | Evidência dos Dados | Implementação |
|---------|-------------------|----------------|
| Diferenciação (d) | Testes ADF/KPSS | `auto_arima` decide automaticamente |
| Sazonalidade | Decomposição sazonal | `seasonal=True, m=12` |
| Ordem AR (p) | PACF | `auto_arima` busca 0-5 |
| Ordem MA (q) | ACF | `auto_arima` busca 0-5 |
| Variáveis Exógenas | Correlação > 0.3 | Incluídas no modelo |
| Features de Lag | Análise exploratória | `max_lags=3` |
| Critério de Informação | Amostra pequena | AICc usado |

---

## Conclusão

Esta implementação garante que:

1. ✅ **Todos os passos seguem CRISP-DM**
2. ✅ **Decisões são baseadas em evidências estatísticas**
3. ✅ **Modelo é validado com múltiplos critérios**
4. ✅ **Pipeline é reprodutível e documentado**
5. ✅ **Artefatos são salvos automaticamente**

**O modelo só será considerado adequado se passar em todos os testes estatísticos e apresentar métricas aceitáveis para o contexto de negócio.**

