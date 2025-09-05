# Autonomous Business Intelligence Agent

A prototype autonomous pattern recognition and decision-making system that can independently discover, synthesize, and act on business intelligence.

## Features

- **Autonomous Data Analysis**: Automatically detects anomalies, trends, and patterns in data
- **Multi-Agent Architecture**: Specialized agents for planning, metrics, drilldown, and tool execution
- **Action Classification**: Maps insights to appropriate business actions with confidence scoring
- **Interactive Dashboard**: Streamlit-based UI for data upload, analysis, and action approval
- **Data Source Connectors**: Support for various data sources (Salesforce, BigQuery, Snowflake, MySQL)

## Quick Start

1. **Generate Sample Data**: Click "Generate New CSV" to create dummy data
2. **Run Analysis**: Click "Run Agent" to start autonomous analysis
3. **Review Results**: View insights, patterns, decisions, and actions in the dashboard
4. **Approve Actions**: Review and approve pending actions based on confidence scores

## Architecture

- **PlannerAgent**: Creates analysis plans and coordinates other agents
- **MetricAgent**: Performs autonomous data analysis and anomaly detection
- **DrilldownAgent**: Enhances insights with additional context
- **ToolAgent**: Executes actions using various business tools

## Data Sources

The system supports connecting to various data sources:
- Salesforce
- BigQuery
- Snowflake
- MySQL
- CSV files

## Deployment

This app is designed to run on Streamlit Community Cloud. Simply connect your GitHub repository to deploy.

## Requirements

- Python 3.9+
- Streamlit
- Pandas, NumPy
- Scikit-learn, Statsmodels
- Plotly for visualizations
- Pydantic for data validation
