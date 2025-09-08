import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import random


def generate_dummy_data(num_rows: int = 500, random_seed: Optional[int] = 42) -> pd.DataFrame:
	"""Generate a dynamic synthetic business dataset with varying columns and structures."""
	if random_seed is not None:
		np.random.seed(random_seed)
		random.seed(random_seed)

	# Define different dataset templates
	dataset_templates = [
		{
			"name": "E-commerce Analytics",
			"columns": {
				"date": "datetime",
				"product_category": "categorical",
				"brand": "categorical", 
				"region": "categorical",
				"sales_volume": "numeric",
				"revenue": "numeric",
				"profit_margin": "numeric",
				"customer_rating": "numeric",
				"return_rate": "numeric",
				"inventory_level": "numeric"
			}
		},
		{
			"name": "Marketing Campaign",
			"columns": {
				"campaign_date": "datetime",
				"campaign_type": "categorical",
				"target_audience": "categorical",
				"platform": "categorical",
				"impressions": "numeric",
				"clicks": "numeric",
				"conversions": "numeric",
				"cost_per_click": "numeric",
				"engagement_rate": "numeric",
				"roi": "numeric"
			}
		},
		{
			"name": "Financial Metrics",
			"columns": {
				"transaction_date": "datetime",
				"account_type": "categorical",
				"transaction_category": "categorical",
				"amount": "numeric",
				"fees": "numeric",
				"balance": "numeric",
				"credit_score": "numeric",
				"risk_level": "categorical",
				"processing_time": "numeric"
			}
		},
		{
			"name": "HR Analytics",
			"columns": {
				"employee_id": "numeric",
				"department": "categorical",
				"position": "categorical",
				"hire_date": "datetime",
				"salary": "numeric",
				"performance_score": "numeric",
				"attendance_rate": "numeric",
				"training_hours": "numeric",
				"satisfaction_score": "numeric"
			}
		},
		{
			"name": "Supply Chain",
			"columns": {
				"order_date": "datetime",
				"supplier": "categorical",
				"product_line": "categorical",
				"warehouse": "categorical",
				"quantity": "numeric",
				"unit_cost": "numeric",
				"shipping_cost": "numeric",
				"delivery_time": "numeric",
				"quality_score": "numeric"
			}
		},
		{
			"name": "Customer Support",
			"columns": {
				"ticket_date": "datetime",
				"priority": "categorical",
				"category": "categorical",
				"agent": "categorical",
				"resolution_time": "numeric",
				"customer_satisfaction": "numeric",
				"ticket_count": "numeric",
				"escalation_rate": "numeric"
			}
		}
	]

	# Randomly select a template
	template = random.choice(dataset_templates)
	
	# Generate data based on selected template
	data = []
	start_date = datetime.today().date() - timedelta(days=num_rows)
	
	# Define categorical values for each category type
	categorical_values = {
		"product_category": ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"],
		"brand": ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"],
		"region": ["North", "South", "East", "West", "Central"],
		"campaign_type": ["Social Media", "Email", "Search", "Display", "Video"],
		"target_audience": ["Gen Z", "Millennials", "Gen X", "Boomers"],
		"platform": ["Facebook", "Instagram", "Google", "Twitter", "LinkedIn"],
		"account_type": ["Savings", "Checking", "Credit", "Investment"],
		"transaction_category": ["Food", "Transport", "Entertainment", "Shopping", "Bills"],
		"risk_level": ["Low", "Medium", "High"],
		"department": ["Sales", "Marketing", "Engineering", "HR", "Finance"],
		"position": ["Manager", "Senior", "Junior", "Director", "Analyst"],
		"supplier": ["Supplier1", "Supplier2", "Supplier3", "Supplier4"],
		"product_line": ["LineA", "LineB", "LineC", "LineD"],
		"warehouse": ["Warehouse1", "Warehouse2", "Warehouse3"],
		"priority": ["Low", "Medium", "High", "Critical"],
		"category": ["Technical", "Billing", "General", "Feature Request"],
		"agent": ["Agent1", "Agent2", "Agent3", "Agent4", "Agent5"]
	}

	for i in range(num_rows):
		row = {}
		
		for col_name, col_type in template["columns"].items():
			if col_type == "datetime":
				# Generate date with some randomness
				days_offset = i + random.randint(-5, 5)
				row[col_name] = start_date + timedelta(days=days_offset)
				
			elif col_type == "categorical":
				# Find appropriate categorical values
				category_key = None
				for key in categorical_values:
					if key in col_name.lower() or col_name.lower() in key:
						category_key = key
						break
				
				if category_key and category_key in categorical_values:
					row[col_name] = random.choice(categorical_values[category_key])
				else:
					# Generate generic categorical values
					row[col_name] = f"Category{random.randint(1, 5)}"
					
			elif col_type == "numeric":
				# Generate numeric values with realistic ranges
				if "id" in col_name.lower():
					row[col_name] = random.randint(1000, 9999)
				elif "rate" in col_name.lower() or "score" in col_name.lower():
					row[col_name] = round(random.uniform(0, 10), 2)
				elif "cost" in col_name.lower() or "price" in col_name.lower():
					row[col_name] = round(random.uniform(10, 1000), 2)
				elif "time" in col_name.lower() or "hours" in col_name.lower():
					row[col_name] = round(random.uniform(1, 100), 1)
				elif "amount" in col_name.lower() or "revenue" in col_name.lower() or "salary" in col_name.lower():
					row[col_name] = round(random.uniform(1000, 100000), 2)
				elif "count" in col_name.lower() or "volume" in col_name.lower() or "quantity" in col_name.lower():
					row[col_name] = random.randint(1, 1000)
				else:
					row[col_name] = round(random.uniform(0, 1000), 2)
		
		data.append(row)

	df = pd.DataFrame(data)
	
	# Add some anomalies to make it interesting for analysis
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	for col in numeric_cols:
		if len(df) > 10:  # Only add anomalies if we have enough data
			anomaly_count = max(1, len(df) // 20)  # 5% anomalies
			anomaly_indices = random.sample(range(len(df)), anomaly_count)
			
			for idx in anomaly_indices:
				# Create anomalies by multiplying by extreme values
				multiplier = random.choice([0.1, 0.2, 3.0, 5.0])
				original_value = df.iloc[idx, df.columns.get_loc(col)]
				new_value = original_value * multiplier
				
				# Ensure non-negative values for certain columns
				if "rate" in col.lower() or "score" in col.lower():
					new_value = max(0, new_value)
				
				# Convert to appropriate type
				if df[col].dtype == 'int64':
					df.iloc[idx, df.columns.get_loc(col)] = int(new_value)
				else:
					df.iloc[idx, df.columns.get_loc(col)] = new_value

	return df


def generate_and_save_csv(path: str, num_rows: int = 500, random_seed: Optional[int] = 42) -> str:
	df = generate_dummy_data(num_rows=num_rows, random_seed=random_seed)
	
	# Add a proper serial number column instead of using pandas index
	df.insert(0, 'serial_no', range(1, len(df) + 1))
	
	# Save without pandas index to avoid "Unnamed: 0" column
	df.to_csv(path, index=False)
	return path



