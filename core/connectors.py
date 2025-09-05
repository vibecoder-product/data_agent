from typing import Dict, Any, Optional
import pandas as pd
from core.generate import generate_dummy_data


class DataConnector:
	"""Base class for data connectors."""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
	
	def connect(self) -> bool:
		"""Test connection to data source."""
		return True
	
	def get_schema(self) -> Dict[str, Any]:
		"""Get available tables/objects and their schemas."""
		return {}
	
	def query(self, query: str) -> pd.DataFrame:
		"""Execute query and return DataFrame."""
		raise NotImplementedError


class SalesforceConnector(DataConnector):
	"""Simulated Salesforce connector."""
	
	def get_schema(self) -> Dict[str, Any]:
		return {
			"objects": {
				"Opportunities": {
					"fields": ["Id", "Amount", "StageName", "CloseDate", "AccountId", "OwnerId"],
					"sample_query": "SELECT Id, Amount, StageName, CloseDate FROM Opportunities WHERE CloseDate >= LAST_N_DAYS:30"
				},
				"Accounts": {
					"fields": ["Id", "Name", "Industry", "BillingCountry", "AnnualRevenue"],
					"sample_query": "SELECT Id, Name, Industry, AnnualRevenue FROM Accounts"
				},
				"Leads": {
					"fields": ["Id", "Company", "Status", "LeadSource", "CreatedDate"],
					"sample_query": "SELECT Id, Company, Status, LeadSource FROM Leads WHERE CreatedDate >= LAST_N_DAYS:30"
				}
			}
		}
	
	def query(self, query: str) -> pd.DataFrame:
		# Simulate Salesforce data
		return generate_dummy_data(500, random_seed=hash(query) % 1000)


class BigQueryConnector(DataConnector):
	"""Simulated BigQuery connector."""
	
	def get_schema(self) -> Dict[str, Any]:
		return {
			"datasets": {
				"analytics": {
					"tables": {
						"user_events": {
							"fields": ["user_id", "event_type", "timestamp", "page_url", "session_id"],
							"sample_query": "SELECT user_id, event_type, DATE(timestamp) as date FROM analytics.user_events WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)"
						},
						"transactions": {
							"fields": ["transaction_id", "user_id", "amount", "product_id", "timestamp"],
							"sample_query": "SELECT transaction_id, amount, DATE(timestamp) as date FROM analytics.transactions WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)"
						}
					}
				},
				"marketing": {
					"tables": {
						"campaign_performance": {
							"fields": ["campaign_id", "impressions", "clicks", "conversions", "date"],
							"sample_query": "SELECT campaign_id, impressions, clicks, conversions, date FROM marketing.campaign_performance WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)"
						}
					}
				}
			}
		}
	
	def query(self, query: str) -> pd.DataFrame:
		# Simulate BigQuery data
		return generate_dummy_data(500, random_seed=hash(query) % 1000)


class SnowflakeConnector(DataConnector):
	"""Simulated Snowflake connector."""
	
	def get_schema(self) -> Dict[str, Any]:
		return {
			"databases": {
				"ANALYTICS": {
					"schemas": {
						"PUBLIC": {
							"tables": {
								"customer_behavior": {
									"fields": ["customer_id", "action", "timestamp", "channel"],
									"sample_query": "SELECT customer_id, action, DATE(timestamp) as date FROM ANALYTICS.PUBLIC.customer_behavior WHERE DATE(timestamp) >= DATEADD(day, -30, CURRENT_DATE())"
								},
								"sales_data": {
									"fields": ["order_id", "customer_id", "amount", "product_category", "order_date"],
									"sample_query": "SELECT order_id, amount, product_category, DATE(order_date) as date FROM ANALYTICS.PUBLIC.sales_data WHERE DATE(order_date) >= DATEADD(day, -30, CURRENT_DATE())"
								}
							}
						}
					}
				}
			}
		}
	
	def query(self, query: str) -> pd.DataFrame:
		# Simulate Snowflake data
		return generate_dummy_data(500, random_seed=hash(query) % 1000)


class MySQLConnector(DataConnector):
	"""Simulated MySQL connector."""
	
	def get_schema(self) -> Dict[str, Any]:
		return {
			"tables": {
				"orders": {
					"fields": ["id", "customer_id", "total_amount", "status", "created_at"],
					"sample_query": "SELECT id, total_amount, status, DATE(created_at) as date FROM orders WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
				},
				"users": {
					"fields": ["id", "email", "registration_date", "last_login"],
					"sample_query": "SELECT id, email, DATE(registration_date) as date FROM users WHERE registration_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
				}
			}
		}
	
	def query(self, query: str) -> pd.DataFrame:
		# Simulate MySQL data
		return generate_dummy_data(500, random_seed=hash(query) % 1000)


def get_connector(source_type: str, config: Dict[str, Any]) -> Optional[DataConnector]:
	"""Factory function to create appropriate connector."""
	connectors = {
		"salesforce": SalesforceConnector,
		"bigquery": BigQueryConnector,
		"snowflake": SnowflakeConnector,
		"mysql": MySQLConnector,
	}
	
	connector_class = connectors.get(source_type.lower())
	if connector_class:
		return connector_class(config)
	return None

