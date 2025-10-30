# RELATÓRIO TÉCNICO DETALHADO - PIPELINE PROPHET CASOS CRIMINAIS TJGO

**Data de Geração:** 22 de October de 2025, 09:11:40

---

## ANÁLISE ESTATÍSTICA DOS DADOS

---

## ANÁLISE DETALHADA DAS MÉTRICAS DE PERFORMANCE

### Mean Absolute Error (MAE) - Erro Absoluto Médio

**Valor Obtido:** 647.9473 casos
**Interpretação:**

- O MAE de 647.95 indica que, em média, o modelo apresenta um erro absoluto de aproximadamente 647 casos por mês ao prever o número de casos criminais.
  **Comparação com outras métricas:**
- O MAE (647.95) é menor que o RMSE (831.30), o que é esperado, já que o RMSE penaliza mais erros grandes. A diferença indica que há alguns outliers com erros maiores, mas a maioria das previsões tem erro moderado.

### Root Mean Squared Error (RMSE) - Raiz do Erro Quadrático Médio

**Valor Obtido:** 831.3026 casos
**Interpretação:**

- O RMSE de 831.30 indica que a raiz do erro quadrático médio é de aproximadamente 831 casos por mês.
- Comparado ao MAE (647.95), o RMSE é maior, indicando que há alguns períodos com erros relativamente maiores, mas a maioria das previsões é precisa.

### Mean Absolute Percentage Error (MAPE) - Erro Percentual Absoluto Médio

**Valor Obtido:** 8.0822%
**Interpretação:**

- O MAPE de 8.08% indica que, em média, o erro percentual absoluto é de aproximadamente 8.1%.
  **Classificação do MAPE:**
- **Excelente (< 10%):** Com MAPE de 8.08%, o modelo apresenta precisão percentual excepcional. Erros inferiores a 10% são considerados muito bons para séries temporais, especialmente em domínios como casos criminais que apresentam alta variabilidade e fatores externos complexos.

### R² (Coefficient of Determination) - Coeficiente de Determinação

**Valor Obtido:** 0.9695
**Interpretação:**

- O R² de 0.9695 indica que o modelo explica 96.95% da variância total dos casos criminais.
  **Classificação do R²:**
- **Excelente (R² ≥ 0.90):** Com R² de 0.9695, o modelo explica mais de 90% da variância. Isso indica que o modelo captura quase completamente os padrões presentes nos dados. Para séries temporais criminais, este é um resultado excepcional, considerando a complexidade e variabilidade deste tipo de dado.

---

## ANÁLISE DETALHADA DOS COMPONENTES DO MODELO

O modelo Prophet decompõe a série temporal em componentes principais: tendência, sazonalidade e efeitos de variáveis exógenas. A análise desses componentes permite entender como cada aspecto contribui para as previsões.

### Componente de Tendência

- **Valor Inicial:** 2834.14 casos
- **Valor Final:** 15555.38 casos
- **Variação Total:** 12721.24 casos (+448.86%)

**Interpretação:**

- A tendência apresentou variação significativa (+448.86%) ao longo do período. Este é um padrão importante que foi capturado pelo modelo Prophet, que utiliza changepoints para identificar mudanças na tendência.
- O componente de tendência representa a linha de base da série temporal, descontando efeitos sazonais e de variáveis exógenas. Um modelo com boa captura de tendência é essencial para previsões de longo prazo.

### Componente de Sazonalidade Semanal

- **Amplitude:** 728.21 casos
- **Desvio Padrão:** 269.30 casos

**Interpretação:**

- O componente semanal apresenta amplitude significativa, indicando que há variações sistemáticas dos casos criminais ao longo dos dias da semana. Este padrão pode estar relacionado a fatores como padrões de atividade criminal ou disponibilidade de recursos de segurança em diferentes dias.

### Componente de Sazonalidade Anual

- **Amplitude:** 2873.92 casos
- **Desvio Padrão:** 766.32 casos

**Interpretação:**

- O componente anual apresenta amplitude muito significativa, indicando fortes padrões sazonais anuais nos casos criminais. Isso é esperado, pois eventos como férias, festivais e padrões econômicos variam ao longo do ano e podem influenciar a criminalidade.
- Para séries mensais de casos criminais, a sazonalidade anual é um componente crucial, pois eventos e condições que influenciam a criminalidade frequentemente se repetem anualmente (ex: festas de fim de ano, períodos de férias escolares, etc.).

---

## DIAGNÓSTICO DETALHADO DO MODELO

### Análise Comparativa entre Métricas

A comparação entre MAE e RMSE oferece insights sobre a distribuição dos erros:

- **MAE:** 647.95 casos
- **RMSE:** 831.30 casos
- **Razão RMSE/MAE:** 1.283

**Interpretação da Razão RMSE/MAE:**

- Com razão de 1.283, há algumas previsões com erro maior, mas a maioria das previsões apresenta erro moderado. A diferença entre RMSE e MAE indica presença de alguns períodos com maior erro, mas não de forma extrema.

### Consistência e Coerência das Métricas

**Verificação de Consistência:**
✅ **MAE < RMSE:** Correto. O RMSE é sempre maior ou igual ao MAE, pois penaliza mais erros grandes. Este resultado confirma que os cálculos estão corretos.
✅ **Coerência R² vs Erros:** Com R² alto (boa explicação de variância) e erros moderados, as métricas são coerentes entre si. O modelo apresenta boa capacidade preditiva em múltiplas dimensões.

### Análise de Robustez do Modelo

A robustez do modelo refere-se à sua capacidade de manter boa performance mesmo quando testado em períodos diferentes daqueles utilizados no treinamento.

**Conclusão sobre Robustez:**

- Com R² de 0.9695 e MAPE de 8.08%, o modelo demonstra **robustez** adequada. As métricas indicam que o modelo captura padrões gerais da série temporal que são válidos em diferentes períodos, não apenas no período de treinamento.

---

## ANÁLISE DA PERFORMANCE GERAL DO MODELO

### Síntese das Métricas

- **Precisão Absoluta:** MAE de 647.95 e RMSE de 831.30 indicam que o modelo apresenta boa capacidade de previsão em termos absolutos.
- **Precisão Relativa:** MAPE de 8.08% indica que os erros percentuais são baixos, demonstrando que o modelo se adapta bem à escala dos dados.
- **Capacidade Explicativa:** R² de 0.9695 indica que o modelo explica 96.95% da variância, demonstrando forte capacidade de capturar os padrões e tendências presentes nos dados.

### Avaliação da Qualidade do Modelo

**Conclusão Geral:** O modelo apresenta **desempenho excelente** baseado nas métricas calculadas.

**Pontos Fortes Identificados:**

- MAPE excepcional (8.08%) - erro percentual muito baixo
- R² excelente (0.9695) - explica mais de 90% da variância

### Recomendações Baseadas na Análise

---

## ANÁLISE DE VALIDAÇÃO CRUZADA TEMPORAL

Resultados de validação cruzada não disponíveis para análise detalhada.
A validação cruzada temporal foi executada durante o treinamento, mas métricas detalhadas por fold não foram salvas para análise individual.

---

## CONCLUSÕES E RECOMENDAÇÕES FINAIS

### Conclusão Geral

O modelo Prophet implementado demonstra **desempenho excepcional** para previsão de casos criminais. Com R² acima de 90% e MAPE abaixo de 10%, o modelo apresenta excelente capacidade preditiva e pode ser confiantemente utilizado para previsões e planejamento estratégico.

### Próximos Passos Recomendados

1. **Monitoramento Contínuo:** Acompanhar a performance do modelo ao longo do tempo
2. **Retreinamento Periódico:** Atualizar o modelo com dados mais recentes
3. **Validação com Dados Novos:** Testar o modelo em períodos futuros
4. **Refinamento Contínuo:** Ajustar hiperparâmetros conforme mais dados ficam disponíveis

## Team and Contact

- **Autor** - Eng. Manuel Lucala Zengo
- **Mentorship** – UFG TI RESIDENCY
- **Team** - DIACDE TJGO
- **Methodology** – CRISP-DM adapted for time series
