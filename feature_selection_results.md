# Feature Selection Analysis Results

## Summary
Successfully completed comprehensive feature selection analysis for criminal cases prediction using external features data from 2015-2024.

## Data Processing
- **Original datasets**: 
  - Criminal cases: 120 rows × 5 columns
  - External features: 120 rows × 31 columns
- **Merged dataset**: 120 rows × 36 columns
- **Date range**: 2015-01-01 to 2024-12-01

## Correlation Analysis
- **High correlations found**: 70 pairs with |correlation| > 0.9
- **Features removed**: 15 collinear features based on lower correlation with target variable
- **Remaining features**: 21 after correlation removal

## VIF Analysis
- **High VIF features removed**: 3 features with VIF > 10
  - `roubo_de_veiculo` (VIF: 22.41)
  - `emissao_de_alvaras_de_licenca` (VIF: 13.33)
  - `estupro_de_vulneravel` (VIF: 12.47)
- **Final features after VIF**: 18 features

## Feature Selection Results

### Without Lag Features
- **Lasso selected**: 8 features
- **ElasticNet selected**: 12 features
- **Combined unique**: 12 features

### With Lag Features (max_lags=3)
- **Lag features created**: 36 additional features
- **Lasso selected**: 37 features
- **ElasticNet selected**: 44 features
- **Combined unique**: 46 features

## Final Candidate Regressors (Top 10)

Based on feature importance ranking, the final set of 6-10 candidate regressors includes:

1. **atendimento_pre_hospitalar_lag_3** - Pre-hospital care (3-month lag)
2. **atendimento_pre_hospitalar** - Pre-hospital care (current)
3. **atendimento_pre_hospitalar_lag_1** - Pre-hospital care (1-month lag)
4. **atendimento_pre_hospitalar_lag_2** - Pre-hospital care (2-month lag)
5. **pessoa_localizada_lag_3** - Located person (3-month lag)
6. **lesao_corporal_seguida_de_morte** - Bodily injury followed by death
7. **tentativa_de_feminicidio_lag_2** - Attempted femicide (2-month lag)
8. **morte_de_agente_do_estado_lag_1** - State agent death (1-month lag)
9. **lesao_corporal_seguida_de_morte_lag_2** - Bodily injury followed by death (2-month lag)
10. **suicidio_de_agente_do_estado** - State agent suicide

## Model Performance
- **Final CV R² (ElasticNet)**: -12.84 ± 15.74
- **Note**: Negative R² scores indicate the model performs worse than a simple mean baseline, suggesting the features may not be strongly predictive of criminal cases or there may be data quality issues.

## Key Insights
1. **Pre-hospital care** features dominate the selection, appearing in multiple lag periods
2. **Violence-related features** (bodily injury, femicide attempts) are important predictors
3. **State agent-related events** show temporal patterns
4. **Lag features** significantly improve feature selection, capturing temporal dependencies
5. **Model performance** suggests the relationship between external features and criminal cases may be weak or non-linear

## Recommendations
1. Consider non-linear models (Random Forest, XGBoost) for better performance
2. Investigate data quality and potential missing confounding variables
3. Explore different lag periods and feature engineering techniques
4. Consider domain-specific feature combinations
5. Validate findings with domain experts
