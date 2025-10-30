#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Analysis for Criminal Cases Prediction
- Loads criminal cases data and external features
- Performs correlation analysis and removes collinearity
- Applies VIF analysis
- Performs Lasso/ElasticNet feature selection with TimeSeriesSplit
- Creates lag features and returns final candidate regressors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

def load_and_join_data():
    """Load criminal cases data and external features, then join them"""
    print("Loading data...")
    
    # Load criminal cases data
    casos_df = pd.read_csv('/Users/universo369/Documents/RESIDENCIA EM TI - TJGO + TJGO/TJGO/predict_series_criminal_tjgo/data/raw_data/casos_mensal__criminal_series_2015_2024.csv')
    
    # Load external features
    external_df = pd.read_csv('/Users/universo369/Documents/RESIDENCIA EM TI - TJGO + TJGO/TJGO/predict_series_criminal_tjgo/data/raw_data/external_features_2015_2024.csv')
    
    print("Criminal cases data shape: {}".format(casos_df.shape))
    print("External features data shape: {}".format(external_df.shape))
    
    # Convert date columns to datetime
    casos_df['DATA'] = pd.to_datetime(casos_df['DATA'])
    external_df['data'] = pd.to_datetime(external_df['data'])
    
    # Join datasets on date
    merged_df = pd.merge(casos_df, external_df, left_on='DATA', right_on='data', how='inner')
    
    print("Merged data shape: {}".format(merged_df.shape))
    print("Date range: {} to {}".format(merged_df['DATA'].min(), merged_df['DATA'].max()))
    
    return merged_df

def calculate_correlations(df, target_col='TOTAL_CASOS'):
    """Calculate correlations and identify highly correlated features"""
    print("\nCalculating correlations...")
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols + [target_col]].corr()
    
    # Find highly correlated pairs (|correlation| > 0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print("Found {} highly correlated pairs (|corr| > 0.9):".format(len(high_corr_pairs)))
    for pair in high_corr_pairs:
        print("  {} <-> {}: {:.3f}".format(pair['feature1'], pair['feature2'], pair['correlation']))
    
    return corr_matrix, high_corr_pairs

def remove_collinear_features(df, high_corr_pairs, target_col='TOTAL_CASOS'):
    """Remove collinear features based on correlation threshold"""
    print("\nRemoving collinear features...")
    
    features_to_remove = set()
    
    for pair in high_corr_pairs:
        feat1, feat2 = pair['feature1'], pair['feature2']
        
        # Skip if target column is involved
        if feat1 == target_col or feat2 == target_col:
            continue
            
        # Calculate correlation with target to decide which to keep
        corr1_target = abs(df[feat1].corr(df[target_col]))
        corr2_target = abs(df[feat2].corr(df[target_col]))
        
        # Remove the feature with lower correlation to target
        if corr1_target < corr2_target:
            features_to_remove.add(feat1)
            print("  Removing {} (corr with target: {:.3f})".format(feat1, corr1_target))
        else:
            features_to_remove.add(feat2)
            print("  Removing {} (corr with target: {:.3f})".format(feat2, corr2_target))
    
    # Remove collinear features
    df_cleaned = df.drop(columns=list(features_to_remove))
    
    print("Removed {} collinear features".format(len(features_to_remove)))
    print("Remaining features: {}".format(df_cleaned.shape[1]))
    
    return df_cleaned, list(features_to_remove)

def calculate_vif(df, target_col='TOTAL_CASOS'):
    """Calculate Variance Inflation Factor for remaining features"""
    print("\nCalculating VIF...")
    
    # Select numeric features (excluding target and date columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [target_col, 'ANO', 'MES']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols
    vif_data["VIF"] = [variance_inflation_factor(df[feature_cols].values, i) 
                      for i in range(len(feature_cols))]
    
    # Sort by VIF
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("VIF Analysis:")
    print(vif_data.to_string(index=False))
    
    # Remove features with VIF > 10
    high_vif_features = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
    
    if high_vif_features:
        print("\nRemoving {} features with VIF > 10:".format(len(high_vif_features)))
        for feat in high_vif_features:
            print("  {} (VIF: {:.2f})".format(feat, vif_data[vif_data['Feature']==feat]['VIF'].iloc[0]))
        
        df_vif_cleaned = df.drop(columns=high_vif_features)
    else:
        print("\nNo features with VIF > 10 found")
        df_vif_cleaned = df.copy()
    
    return df_vif_cleaned, vif_data, high_vif_features

def create_lag_features(df, target_col='TOTAL_CASOS', max_lags=3):
    """Create lag features for time series analysis"""
    print("\nCreating lag features (max_lags={})...".format(max_lags))
    
    df_lagged = df.copy()
    
    # Get feature columns (excluding target and date columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [target_col, 'ANO', 'MES']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create lag features
    for lag in range(1, max_lags + 1):
        for col in feature_cols:
            df_lagged['{}_lag_{}'.format(col, lag)] = df[col].shift(lag)
    
    # Remove rows with NaN values (first max_lags rows)
    df_lagged = df_lagged.dropna()
    
    print("Created {} lag features".format(len(feature_cols) * max_lags))
    print("Data shape after lag creation: {}".format(df_lagged.shape))
    
    return df_lagged

def perform_feature_selection(df, target_col='TOTAL_CASOS', n_splits=5):
    """Perform Lasso and ElasticNet feature selection with TimeSeriesSplit"""
    print("\nPerforming feature selection with TimeSeriesSplit (n_splits={})...".format(n_splits))
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [target_col, 'ANO', 'MES']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Lasso CV
    print("Running Lasso CV...")
    lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=2000)
    lasso_cv.fit(X_scaled, y)
    
    # ElasticNet CV
    print("Running ElasticNet CV...")
    elastic_cv = ElasticNetCV(cv=tscv, random_state=42, max_iter=2000, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    elastic_cv.fit(X_scaled, y)
    
    # Get selected features
    lasso_features = [feature_cols[i] for i in range(len(feature_cols)) if abs(lasso_cv.coef_[i]) > 1e-6]
    elastic_features = [feature_cols[i] for i in range(len(feature_cols)) if abs(elastic_cv.coef_[i]) > 1e-6]
    
    print("\nLasso selected {} features:".format(len(lasso_features)))
    for feat in lasso_features:
        coef_idx = feature_cols.index(feat)
        print("  {}: {:.4f}".format(feat, lasso_cv.coef_[coef_idx]))
    
    print("\nElasticNet selected {} features:".format(len(elastic_features)))
    for feat in elastic_features:
        coef_idx = feature_cols.index(feat)
        print("  {}: {:.4f}".format(feat, elastic_cv.coef_[coef_idx]))
    
    # Combine features from both methods
    combined_features = list(set(lasso_features + elastic_features))
    
    print("\nCombined unique features: {}".format(len(combined_features)))
    
    return {
        'lasso_features': lasso_features,
        'elastic_features': elastic_features,
        'combined_features': combined_features,
        'lasso_cv': lasso_cv,
        'elastic_cv': elastic_cv,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

def evaluate_features(df, selected_features, target_col='TOTAL_CASOS', n_splits=5):
    """Evaluate selected features using cross-validation"""
    print("\nEvaluating selected features...")
    
    X = df[selected_features].values
    y = df[target_col].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Lasso evaluation
    lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=2000)
    lasso_cv.fit(X_scaled, y)
    
    # ElasticNet evaluation
    elastic_cv = ElasticNetCV(cv=tscv, random_state=42, max_iter=2000, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    elastic_cv.fit(X_scaled, y)
    
    # Cross-validation scores
    lasso_scores = []
    elastic_scores = []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Lasso
        lasso_cv.fit(X_train, y_train)
        lasso_pred = lasso_cv.predict(X_test)
        lasso_scores.append(r2_score(y_test, lasso_pred))
        
        # ElasticNet
        elastic_cv.fit(X_train, y_train)
        elastic_pred = elastic_cv.predict(X_test)
        elastic_scores.append(r2_score(y_test, elastic_pred))
    
    print("Lasso CV R2 scores: {:.4f} ± {:.4f}".format(np.mean(lasso_scores), np.std(lasso_scores)))
    print("ElasticNet CV R2 scores: {:.4f} ± {:.4f}".format(np.mean(elastic_scores), np.std(elastic_scores)))
    
    return {
        'lasso_scores': lasso_scores,
        'elastic_scores': elastic_scores,
        'lasso_cv': lasso_cv,
        'elastic_cv': elastic_cv
    }

def main():
    """Main analysis pipeline"""
    print("="*60)
    print("CRIMINAL CASES FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # Step 1: Load and join data
    df = load_and_join_data()
    
    # Step 2: Calculate correlations and remove collinearity
    corr_matrix, high_corr_pairs = calculate_correlations(df)
    df_cleaned, removed_corr = remove_collinear_features(df, high_corr_pairs)
    
    # Step 3: Apply VIF analysis
    df_vif_cleaned, vif_data, removed_vif = calculate_vif(df_cleaned)
    
    # Step 4: Feature selection without lags
    print("\n" + "="*40)
    print("FEATURE SELECTION WITHOUT LAGS")
    print("="*40)
    
    selection_results = perform_feature_selection(df_vif_cleaned)
    
    # Step 5: Create lag features and repeat selection
    print("\n" + "="*40)
    print("FEATURE SELECTION WITH LAGS")
    print("="*40)
    
    df_lagged = create_lag_features(df_vif_cleaned)
    selection_results_lagged = perform_feature_selection(df_lagged)
    
    # Step 6: Final candidate selection
    print("\n" + "="*40)
    print("FINAL CANDIDATE REGRESSORS")
    print("="*40)
    
    # Combine features from both analyses
    all_candidates = list(set(selection_results['combined_features'] + 
                             selection_results_lagged['combined_features']))
    
    # Limit to 6-10 features
    if len(all_candidates) > 10:
        # Evaluate all candidates and select top 10
        eval_results = evaluate_features(df_lagged, all_candidates)
        
        # Get feature importance from ElasticNet
        feature_importance = []
        for feat in all_candidates:
            if feat in selection_results_lagged['feature_cols']:
                idx = selection_results_lagged['feature_cols'].index(feat)
                importance = abs(selection_results_lagged['elastic_cv'].coef_[idx])
                feature_importance.append((feat, importance))
        
        # Sort by importance and select top 10
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        final_candidates = [feat for feat, _ in feature_importance[:10]]
    else:
        final_candidates = all_candidates
    
    print("\nFinal candidate regressors ({} features):".format(len(final_candidates)))
    for i, feat in enumerate(final_candidates, 1):
        print("  {:2d}. {}".format(i, feat))
    
    # Final evaluation
    print("\nFinal evaluation with {} features:".format(len(final_candidates)))
    final_eval = evaluate_features(df_lagged, final_candidates)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print("Original features: {}".format(df.shape[1] - 5))  # Excluding date columns and target
    print("After correlation removal: {} features removed".format(len(removed_corr)))
    print("After VIF removal: {} features removed".format(len(removed_vif)))
    print("Features without lags: {}".format(len(selection_results['combined_features'])))
    print("Features with lags: {}".format(len(selection_results_lagged['combined_features'])))
    print("Final candidates: {}".format(len(final_candidates)))
    print("Final CV R2 (ElasticNet): {:.4f} ± {:.4f}".format(np.mean(final_eval['elastic_scores']), np.std(final_eval['elastic_scores'])))
    
    return {
        'final_candidates': final_candidates,
        'df_processed': df_lagged,
        'selection_results': selection_results,
        'selection_results_lagged': selection_results_lagged,
        'final_evaluation': final_eval
    }

if __name__ == "__main__":
    results = main()
