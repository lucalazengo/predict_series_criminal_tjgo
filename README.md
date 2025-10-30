# Time Series Forecasting Pipeline – TJGO Criminal Cases

An automated and reproducible pipeline for monthly time series forecasting using **Facebook Prophet**, specifically developed for analyzing and forecasting criminal cases at the Court of Justice of Goiás (TJGO).

## Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Installation](#-quick-installation)
- [Execution Guide](#-execution-guide)
- [Configuration](#-configuration)
- [Results and Outputs](#-results-and-outputs)
- [Documentation](#-documentation)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

---

## Features

### Core Capabilities

- ✅ **Full Prophet Model**: Implementation with full support for exogenous variables
- ✅ **Temporal Validation**: Time Series Cross-Validation (`TimeSeriesSplit`) to prevent data leakage
- ✅ **Automatic Optimization**: Hyperparameter tuning using Optuna (50+ trials)
- ✅ **Detailed Feature Analysis**: Relative importance, correlations, and contributions
- ✅ **Comprehensive Metrics**: MAE, RMSE, MAPE, SMAPE, R²
- ✅ **Visualizations**: Forecast plots, component decomposition, and residual analysis
- ✅ **Detailed Reports**: Interactive HTML and complete technical report in Markdown
- ✅ **Artifact Management**: Automatic saving of models, forecasts, and metrics
- ✅ **Modular Design**: Organized code structure (`src/`)
- ✅ **Complete Documentation**: Detailed guides in Portuguese

### Recent Improvements (October 2025)

- Robust optimization: 50 trials (increased from 5)
- Expanded validation: 5 cross-validation splits
- In-depth analysis of exogenous variable importance
- Detailed technical report with advanced diagnostics
- Automatic report generation after execution

---

## Project Structure

```
predict_series_criminal_tjgo/
├── src/                          # Modular source code
│   ├── data/                    # Data loading and preprocessing
│   │   └── __init__.py          # DataManager, DataProcessor, DataValidator
│   ├── models/                  # Prophet model wrapper
│   │   └── __init__.py          # ProphetModelWrapper, ProphetHyperparameterOptimizer
│   ├── training/                # Training pipeline
│   │   └── __init__.py          # TrainingPipeline, TimeSeriesSplitter
│   ├── evaluation/              # Evaluation and metrics
│   │   └── __init__.py          # ModelEvaluator, FeatureImportanceAnalyzer, MetricsCalculator
│   └── utils/                   # Utilities and reporting
│       └── __init__.py          # ReportGenerator, Visualizer, ArtifactManager
│
├── configs/                      # Configuration files
│   ├── default_config.yaml      # Default configuration
│   └── criminal_cases_config.yaml  # Configuration for criminal cases
│
├── data/                         # Project data
│   ├── raw_data/                # Raw data
│   │   ├── casos_mensal__criminal_series_2015_2024.csv
│   │   └── external_features_2015_2024.csv
│   └── processed/               # Processed data
│
├── outputs/                      # Generated results
│   ├── models/                  # Trained models (.joblib)
│   ├── predictions/             # Forecasts (.csv)
│   └── reports/                 # Reports and visualizations
│       ├── *.html               # HTML reports
│       ├── *.md                 # Markdown reports
│       ├── *.png                # Plots
│       └── *.json               # Metrics and analyses
│
├── logs/                        # Log files
│
├── tests/                       # Unit tests
│
├── execute_pipeline.py          # Main execution script ⭐
├── generate_detailed_report.py  # Detailed technical report generator
├── main.py                      # Alternative execution script
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── README.md                    # This file
├── GUIA_EXECUCAO.md            # Detailed execution guide 📘
├── DOCUMENTACAO_TECNICA.md     # Full technical documentation
├── RELATORIO_EXECUCAO_FINAL.md # Execution report
└── RESUMO_IMPLEMENTACOES.md    # Implementation summary
```

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk Space**: ~500MB

### Python Dependencies

All dependencies are listed in `requirements.txt`:

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

## Quick Installation

### 1. Clone/Download the Project

```bash
cd /path/to/project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python3 -c "import pandas, numpy, prophet; print(' Dependencies installed!')"
```

---

## Execution Guide

### Basic Execution (Recommended)

```bash
python3 execute_pipeline.py
```

This command runs the full pipeline:

1. Loads and prepares data
2. Validates data quality
3. Trains Prophet model with hyperparameter optimization (50 trials)
4. Generates future forecasts
5. Computes all evaluation metrics
6. Analyzes exogenous variable importance
7. Generates visualizations (plots)
8. Saves all artifacts
9. Generates HTML report
10. Generates detailed Markdown technical report

**Estimated Time:** 30–60 minutes (with 50 trials)

### Fast Execution (For Testing)

1. Edit `configs/criminal_cases_config.yaml`:

```yaml
training:
  hyperparameter_optimization:
    n_trials: 5  # Reduce for faster execution
```

2. Run:

```bash
python3 execute_pipeline.py
```

**Estimated Time:** 5–10 minutes

### Generate Only the Detailed Report

If you’ve already run the pipeline and only want to regenerate the detailed report:

```bash
python3 generate_detailed_report.py
```

### Full Documentation

For complete step-by-step instructions, see:

- **[GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)** – Comprehensive execution guide

---

## Configuration

### Main Configuration File

`configs/criminal_cases_config.yaml`

### Key Parameters

```yaml
# Hyperparameter Optimization
training:
  hyperparameter_optimization:
    enabled: true
    n_trials: 50              # Number of trials (higher = more robust, slower)

# Cross-Validation
training:
  cv:
    n_splits: 5               # Number of TimeSeriesSplit folds

# Forecast Horizon
forecasting:
  horizon_months: 12          # Number of future months to forecast

# Evaluation Metrics
evaluation:
  metrics:
    - "mae"                   # Mean Absolute Error
    - "rmse"                  # Root Mean Squared Error
    - "mape"                  # Mean Absolute Percentage Error
    - "smape"                 # Symmetric MAPE
    - "r2"                    # R-squared
```

### Exogenous Variables

The model uses the following exogenous variables (configured in `criminal_cases_config.yaml`):

- `atendimento_pre_hospitalar`
- `pessoa_localizada`
- `lesao_corporal_seguida_de_morte`
- `tentativa_de_feminicidio`
- `morte_de_agente_do_estado`
- `suicidio_de_agente_do_estado`

---

## Results and Outputs

### Output Structure

```
outputs/
├── models/
│   └── prophet_model_YYYYMMDD_HHMMSS.joblib      # Trained model
│
├── predictions/
│   └── forecast_YYYYMMDD_HHMMSS.csv              # Future forecasts
│
└── reports/
    ├── forecast_plot_YYYYMMDD_HHMMSS.png          # Forecast plot
    ├── components_plot_YYYYMMDD_HHMMSS.png       # Time series decomposition
    ├── residuals_plot_YYYYMMDD_HHMMSS.png         # Residual analysis
    ├── metrics_YYYYMMDD_HHMMSS.json                # Computed metrics
    ├── feature_analysis_YYYYMMDD_HHMMSS.json      # Feature importance analysis
    ├── config_YYYYMMDD_HHMMSS.yaml                # Configuration used
    ├── report_YYYYMMDD_HHMMSS.html                # Interactive HTML report
    └── RELATORIO_DETALHADO_COMPLETO_*.md          # Full technical report
```

### Performance Metrics

Typical results achieved:

| Metric          | Value      | Rating            |
| --------------- | ---------- | ----------------- |
| **MAE**   | ~648 cases | Excellent         |
| **RMSE**  | ~831 cases | Excellent         |
| **MAPE**  | ~8.08%     | Excellent (<10%)  |
| **SMAPE** | ~8.62%     | Excellent         |
| **R²**   | ~0.9695    | Excellent (>0.90) |

**Interpretation:**

- **R² = 0.9695**: The model explains **96.95%** of the data variance
- **MAPE = 8.08%**: Very low percentage error (<10% is excellent)
- **MAE = 647.95**: Average absolute error of approximately 648 cases/month

### Generated Visualizations

1. **Forecast Plot**: Actual vs. predicted time series with confidence intervals
2. **Component Plot**: Decomposition into trend, seasonality, and exogenous effects
3. **Residual Plot**: Forecast error analysis

### Reports

1. **HTML Report**: Interactive report with embedded metrics and visualizations
2. **Detailed Markdown Report**: Comprehensive technical document including:
   - Statistical data analysis
   - In-depth metric interpretation
   - Model component diagnostics
   - Full model validation
   - Feature importance analysis
   - Conclusions and recommendations

---

## Team and Contact

- **Author** – Eng. Manuel Lucala Zengo
- **Mentorship** – UFG TI Residency Program
- **Team** – DIACDE TJGO
- **Methodology** – CRISP-DM adapted for time series
