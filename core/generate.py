import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_dummy_data(num_rows: int = 500, random_seed: Optional[int] = 42) -> pd.DataFrame:
	"""Generate a synthetic business dataset with time series and segments.

	Columns: date, channel, region, product, sessions, conversions, revenue, cost, cpc, conversion_rate, margin
	"""
	if random_seed is not None:
		np.random.seed(random_seed)

	start_date = datetime.today().date() - timedelta(days=num_rows)
	dates = pd.date_range(start=start_date, periods=num_rows, freq="D")
	channels = ["Search", "Social", "Email", "Direct", "Affiliate"]
	regions = ["NA", "EU", "APAC"]
	products = ["A", "B", "C"]

	# Base seasonal patterns
	weekly = (np.sin(np.arange(num_rows) * 2 * np.pi / 7) + 1.2) * 0.5
	trend = np.linspace(1.0, 1.3, num_rows)

	sessions_base = 1000 * trend * (1.0 + 0.2 * weekly)

	data = []
	for i, d in enumerate(dates):
		for ch in channels:
			for rg in regions:
				for pr in products:
					if len(data) >= num_rows:
						break
					noise = np.random.normal(0, 60)
					mult = 1.0
					if ch == "Search":
						mult *= 1.4
					if rg == "EU":
						mult *= 0.9
					if pr == "B":
						mult *= 1.1

					sessions = max(10, sessions_base[i] * mult + noise)
					conv_rate = np.clip(0.02 + 0.01 * weekly[i] + np.random.normal(0, 0.002), 0.005, 0.12)
					conversions = max(1, int(sessions * conv_rate))
					cpc = np.clip(0.6 + 0.2 * np.random.rand(), 0.4, 2.0)
					cost = float(sessions / 3.0 * cpc)
					rev_per_conv = 40 + 10 * (1 if pr == "A" else 0) + 5 * (1 if ch == "Email" else 0)
					revenue = float(conversions * rev_per_conv * (0.95 + 0.1 * np.random.rand()))
					margin = float(revenue - cost)
					data.append({
						"date": d.date(),
						"channel": ch,
						"region": rg,
						"product": pr,
						"sessions": int(sessions),
						"conversions": conversions,
						"revenue": round(revenue, 2),
						"cost": round(cost, 2),
						"cpc": round(cpc, 2),
						"conversion_rate": round(conversions / max(sessions, 1), 4),
						"margin": round(margin, 2),
					})
				if len(data) >= num_rows:
					break
			if len(data) >= num_rows:
				break
		if len(data) >= num_rows:
			break

	df = pd.DataFrame(data)

	# Inject a few anomalies
	for col in ["sessions", "revenue", "cost", "conversion_rate", "margin"]:
		idx = np.random.choice(df.index, size=max(1, num_rows // 25), replace=False)
		if col == "conversion_rate":
			df.loc[idx, col] *= np.random.choice([0.2, 0.4, 1.8, 2.2], size=len(idx))
		else:
			df.loc[idx, col] *= np.random.choice([0.2, 0.4, 1.8, 2.2], size=len(idx))
			df[col] = df[col].clip(lower=0.001)

	return df


def generate_and_save_csv(path: str, num_rows: int = 500, random_seed: Optional[int] = 42) -> str:
	df = generate_dummy_data(num_rows=num_rows, random_seed=random_seed)
	df.to_csv(path, index=False)
	return path



