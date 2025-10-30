#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils module for Prophet forecasting pipeline.
Handles reporting, visualization, artifact management, and utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import yaml
import json
from pathlib import Path
from datetime import datetime
import joblib
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """Generates comprehensive reports for the forecasting pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_config = config['output']
        
    def generate_report(self, training_results: Dict, evaluation_results: Dict, 
                       forecast: pd.DataFrame, actual_data: pd.DataFrame) -> str:
        """Generate comprehensive HTML report."""
        logger.info("Generating comprehensive report...")
        
        report_html = self._create_html_template()
        
        # Add executive summary
        report_html += self._add_executive_summary(training_results, evaluation_results)
        
        # Add metrics section
        report_html += self._add_metrics_section(evaluation_results)
        
        # Add component analysis
        if 'component_analysis' in evaluation_results:
            report_html += self._add_component_analysis(evaluation_results['component_analysis'])
        
        # Add performance analysis
        report_html += self._add_performance_analysis(evaluation_results['performance_analysis'])
        
        # Add forecast visualization
        report_html += self._add_forecast_visualization(forecast, actual_data)
        
        # Add model diagnostics
        report_html += self._add_model_diagnostics(training_results, evaluation_results)
        
        # Close HTML template
        report_html += self._close_html_template()
        
        # Save report
        report_path = self._save_report(report_html)
        
        logger.info(f"Report generated successfully: {report_path}")
        
        return report_path
    
    def _create_html_template(self) -> str:
        """Create HTML template for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prophet Forecasting Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                h3 {{ color: #7f8c8d; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 5px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Prophet Forecasting Pipeline Report</h1>
            <p>Generated on: {timestamp}</p>
        """
    
    def _add_executive_summary(self, training_results: Dict, evaluation_results: Dict) -> str:
        """Add executive summary section."""
        metrics = evaluation_results['metrics']
        
        summary = f"""
        <h2>Executive Summary</h2>
        <div class="metric">
            <div class="metric-value">{metrics.get('mae', 'N/A'):.2f}</div>
            <div class="metric-label">Mean Absolute Error (MAE)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('rmse', 'N/A'):.2f}</div>
            <div class="metric-label">Root Mean Squared Error (RMSE)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('mape', 'N/A'):.2f}%</div>
            <div class="metric-label">Mean Absolute Percentage Error (MAPE)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('r2', 'N/A'):.3f}</div>
            <div class="metric-label">R-squared (R²)</div>
        </div>
        """
        
        return summary
    
    def _add_metrics_section(self, evaluation_results: Dict) -> str:
        """Add metrics section."""
        metrics = evaluation_results['metrics']
        
        metrics_html = "<h2>Model Performance Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            metrics_html += f"<tr><td>{metric.upper()}</td><td>{formatted_value}</td></tr>"
        
        metrics_html += "</table>"
        
        return metrics_html
    
    def _add_component_analysis(self, component_analysis: Dict) -> str:
        """Add component analysis section."""
        html = "<h2>Model Component Analysis</h2>"
        
        if 'component_contributions' in component_analysis:
            html += "<h3>Component Contributions</h3><table><tr><th>Component</th><th>Contribution</th></tr>"
            
            for component, contribution in component_analysis['component_contributions'].items():
                html += f"<tr><td>{component}</td><td>{contribution:.3f}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _add_performance_analysis(self, performance_analysis: Dict) -> str:
        """Add performance analysis section."""
        html = "<h2>Performance Analysis</h2>"
        
        if 'temporal_performance' in performance_analysis:
            html += "<h3>Temporal Performance</h3>"
            
            if 'yearly' in performance_analysis['temporal_performance']:
                html += "<h4>Performance by Year</h4><table><tr><th>Year</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>"
                
                for year_data in performance_analysis['temporal_performance']['yearly']:
                    html += f"<tr><td>{year_data['year']}</td><td>{year_data.get('rmse', 'N/A'):.3f}</td><td>{year_data.get('mape', 'N/A'):.2f}%</td><td>{year_data.get('r2', 'N/A'):.3f}</td></tr>"
                
                html += "</table>"
        
        return html
    
    def _add_forecast_visualization(self, forecast: pd.DataFrame, actual_data: pd.DataFrame) -> str:
        """Add forecast visualization section."""
        html = "<h2>Forecast Visualization</h2>"
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=actual_data['ds'],
            y=actual_data['y'],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals if available
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title='Time Series Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        # Convert to HTML
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id="forecast-plot")
        
        html += f'<div class="plot-container">{plot_html}</div>'
        
        return html
    
    def _add_model_diagnostics(self, training_results: Dict, evaluation_results: Dict) -> str:
        """Add model diagnostics section."""
        html = "<h2>Model Diagnostics</h2>"
        
        # Feature importance
        if 'feature_importance' in training_results:
            html += "<h3>Feature Importance</h3><table><tr><th>Feature</th><th>Importance</th></tr>"
            
            for feature, importance in training_results['feature_importance'].items():
                html += f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>"
            
            html += "</table>"
        
        # Model parameters
        if 'best_params' in training_results and training_results['best_params']:
            html += "<h3>Optimized Parameters</h3><table><tr><th>Parameter</th><th>Value</th></tr>"
            
            for param, value in training_results['best_params'].items():
                html += f"<tr><td>{param}</td><td>{value:.4f}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _close_html_template(self) -> str:
        """Close HTML template."""
        return "</body></html>"
    
    def _save_report(self, html_content: str) -> str:
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"prophet_report_{timestamp}.html"
        
        output_dir = Path(self.output_config['base_dir']) / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)


class Visualizer:
    """Creates visualizations for the forecasting pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_config = config['output']
        
    def create_forecast_plot(self, forecast: pd.DataFrame, actual_data: pd.DataFrame, 
                           save_path: Optional[str] = None) -> str:
        """Create forecast visualization plot."""
        logger.info("Creating forecast plot...")
        
        plt.figure(figsize=(15, 8))
        
        # Plot actual data
        plt.plot(actual_data['ds'], actual_data['y'], label='Actual', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals if available
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            plt.fill_between(
                forecast['ds'], 
                forecast['yhat_lower'], 
                forecast['yhat_upper'], 
                alpha=0.3, 
                color='red',
                label='Confidence Interval'
            )
        
        plt.title('Time Series Forecast', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/reports/forecast_plot_{timestamp}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast plot saved: {save_path}")
        
        return save_path
    
    def create_components_plot(self, forecast: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create components visualization plot."""
        logger.info("Creating components plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prophet Model Components', fontsize=16, fontweight='bold')
        
        # Trend component
        if 'trend' in forecast.columns:
            axes[0, 0].plot(forecast['ds'], forecast['trend'], color='blue')
            axes[0, 0].set_title('Trend Component')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Yearly seasonality
        yearly_cols = [col for col in forecast.columns if col.startswith('yearly')]
        if yearly_cols:
            axes[0, 1].plot(forecast['ds'], forecast[yearly_cols[0]], color='green')
            axes[0, 1].set_title('Yearly Seasonality')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Weekly seasonality
        weekly_cols = [col for col in forecast.columns if col.startswith('weekly')]
        if weekly_cols:
            axes[1, 0].plot(forecast['ds'], forecast[weekly_cols[0]], color='orange')
            axes[1, 0].set_title('Weekly Seasonality')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Holidays component
        if 'holidays' in forecast.columns:
            axes[1, 1].plot(forecast['ds'], forecast['holidays'], color='purple')
            axes[1, 1].set_title('Holidays Component')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/reports/components_plot_{timestamp}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Components plot saved: {save_path}")
        
        return save_path
    
    def create_residuals_plot(self, actual: np.ndarray, predicted: np.ndarray, 
                            save_path: Optional[str] = None) -> str:
        """Create residuals analysis plot."""
        logger.info("Creating residuals plot...")
        
        # Ensure arrays have the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Residuals Analysis', fontsize=16, fontweight='bold')
        
        # Residuals vs Fitted
        axes[0, 0].scatter(predicted, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/reports/residuals_plot_{timestamp}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residuals plot saved: {save_path}")
        
        return save_path


class ArtifactManager:
    """Manages saving and loading of pipeline artifacts."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_config = config['output']
        
    def save_model(self, model_wrapper, filepath: Optional[str] = None) -> str:
        """Save trained model."""
        logger.info("Saving model...")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/models/prophet_model_{timestamp}.joblib"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model_wrapper.save_model(filepath)
        
        logger.info(f"Model saved: {filepath}")
        
        return filepath
    
    def save_predictions(self, forecast: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """Save predictions."""
        logger.info("Saving predictions...")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/predictions/forecast_{timestamp}.csv"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(filepath, index=False)
        
        logger.info(f"Predictions saved: {filepath}")
        
        return filepath
    
    def save_metrics(self, metrics: Dict, filepath: Optional[str] = None) -> str:
        """Save metrics."""
        logger.info("Saving metrics...")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/reports/metrics_{timestamp}.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved: {filepath}")
        
        return filepath
    
    def save_config(self, config: Dict, filepath: Optional[str] = None) -> str:
        """Save configuration."""
        logger.info("Saving configuration...")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/reports/config_{timestamp}.yaml"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved: {filepath}")
        
        return filepath


class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        return config
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate configuration structure."""
        logger.info("Validating configuration...")
        
        required_sections = ['data', 'model', 'training', 'evaluation', 'output']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        logger.info("Configuration validation passed")
        
        return True
    
    @staticmethod
    def update_config(config: Dict, updates: Dict) -> Dict:
        """Update configuration with new values."""
        logger.info("Updating configuration...")
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        updated_config = deep_update(config.copy(), updates)
        
        logger.info("Configuration updated successfully")
        
        return updated_config
