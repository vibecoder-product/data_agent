from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Insight:
	metric: str
	dimension: str
	segment: str
	description: str
	impact: float
	urgency: float
	confidence: float
	kind: str  # change_point, anomaly, trend_shift, efficiency
	context: Dict[str, Any]


@dataclass
class Pattern:
	pattern: str
	metric: str
	dimension: str
	segment: str
	description: str
	strength: float
	context: Dict[str, Any]


def robust_z_scores(series: pd.Series) -> pd.Series:
	median = np.median(series)
	mad = np.median(np.abs(series - median)) + 1e-9
	z = 0.6745 * (series - median) / mad
	return z


def stl_residual_anomalies(y: pd.Series) -> Tuple[pd.Series, pd.Series]:
	"""Return residual and robust z-scores. Fallback to simple detrend when short."""
	if len(y) < 14:
		# Simple detrend using rolling median for short sequences
		roll = y.rolling(window=max(3, len(y)//3)).median().bfill().ffill()
		residual = y - roll
		z = robust_z_scores(residual)
		return residual, z
	# Simple detrend using rolling median for longer sequences
	period = max(7, min(30, max(7, len(y)//6)))
	roll = y.rolling(window=period).median().bfill().ffill()
	residual = y - roll
	z = robust_z_scores(residual)
	return residual, z


def detect_anomalies(y: pd.Series, window: int = 14, z_thresh: float = 2.0) -> List[Dict[str, Any]]:
	"""Detect anomalies using rolling median and z-score."""
	anomalies = []
	
	if len(y) < window * 2:
		# Fallback for short time series
		mean_val = y.mean()
		std_val = y.std()
		if std_val > 0:
			z_scores = np.abs((y - mean_val) / std_val)
			anomaly_indices = np.where(z_scores > z_thresh)[0]
			for idx in anomaly_indices:
				confidence = min(z_scores[idx] / (z_thresh * 2), 1.0)  # Normalize confidence
				anomaly_value = y.iloc[idx]
				deviation = (anomaly_value - mean_val) / mean_val * 100 if mean_val != 0 else 0
				
				# Create descriptive message
				if anomaly_value > mean_val:
					direction = "spike"
					description = f"Unusual {direction} detected: {anomaly_value:.2f} is {deviation:.1f}% above the average ({mean_val:.2f})"
				else:
					direction = "drop"
					description = f"Unusual {direction} detected: {anomaly_value:.2f} is {abs(deviation):.1f}% below the average ({mean_val:.2f})"
				
				anomalies.append({
					"index": int(idx),
					"value": float(anomaly_value),
					"z_score": float(z_scores[idx]),
					"confidence": confidence,
					"description": description,
					"direction": direction,
					"deviation_percent": float(deviation),
					"baseline_mean": float(mean_val)
				})
		return anomalies
	
	# Rolling median approach
	rolling_median = y.rolling(window=window, center=True).median()
	rolling_std = y.rolling(window=window, center=True).std()
	
	# Fill NaN values
	rolling_median = rolling_median.ffill().bfill()
	rolling_std = rolling_std.ffill().bfill()
	
	# Calculate z-scores
	z_scores = np.abs((y - rolling_median) / rolling_std)
	z_scores = z_scores.fillna(0)
	
	# Find anomalies
	anomaly_indices = np.where(z_scores > z_thresh)[0]
	
	for idx in anomaly_indices:
		confidence = min(z_scores.iloc[idx] / (z_thresh * 2), 1.0)  # Normalize confidence
		anomaly_value = y.iloc[idx]
		expected_value = rolling_median.iloc[idx]
		deviation = (anomaly_value - expected_value) / expected_value * 100 if expected_value != 0 else 0
		
		# Create descriptive message
		if anomaly_value > expected_value:
			direction = "spike"
			description = f"Anomalous {direction} detected at position {idx}: {anomaly_value:.2f} is {deviation:.1f}% above the expected trend ({expected_value:.2f})"
		else:
			direction = "drop"
			description = f"Anomalous {direction} detected at position {idx}: {anomaly_value:.2f} is {abs(deviation):.1f}% below the expected trend ({expected_value:.2f})"
		
		anomalies.append({
			"index": int(idx),
			"value": float(anomaly_value),
			"z_score": float(z_scores.iloc[idx]),
			"confidence": confidence,
			"description": description,
			"direction": direction,
			"deviation_percent": float(deviation),
			"expected_value": float(expected_value)
		})
	
	return anomalies


def change_points(y: pd.Series, window: int = 14, z_thresh: float = 3.0) -> List[Dict[str, Any]]:
	"""Detect change points using rolling statistics."""
	changes = []
	
	if len(y) < window * 2:
		return changes
	
	# Calculate rolling statistics
	rolling_mean = y.rolling(window=window, center=True).mean()
	rolling_std = y.rolling(window=window, center=True).std()
	
	# Fill NaN values
	rolling_mean = rolling_mean.ffill().bfill()
	rolling_std = rolling_std.ffill().bfill()
	
	# Calculate z-scores for change detection
	z_scores = np.abs((y - rolling_mean) / rolling_std)
	z_scores = z_scores.fillna(0)
	
	# Find change points
	change_indices = np.where(z_scores > z_thresh)[0]
	
	for idx in change_indices:
		confidence = min(z_scores.iloc[idx] / (z_thresh * 2), 1.0)  # Normalize confidence
		change_value = y.iloc[idx]
		expected_value = rolling_mean.iloc[idx]
		deviation = (change_value - expected_value) / expected_value * 100 if expected_value != 0 else 0
		
		# Create descriptive message
		if change_value > expected_value:
			direction = "increase"
			description = f"Significant {direction} detected at position {idx}: {change_value:.2f} is {deviation:.1f}% above the rolling average ({expected_value:.2f})"
		else:
			direction = "decrease"
			description = f"Significant {direction} detected at position {idx}: {change_value:.2f} is {abs(deviation):.1f}% below the rolling average ({expected_value:.2f})"
		
		changes.append({
			"index": int(idx),
			"value": float(change_value),
			"z_score": float(z_scores.iloc[idx]),
			"confidence": confidence,
			"description": description,
			"direction": direction,
			"deviation_percent": float(deviation),
			"expected_value": float(expected_value)
		})
	
	return changes


def analyze_dataframe(df: pd.DataFrame, date_col: str = "date") -> List[Insight]:
	insights: List[Insight] = []
	if date_col in df.columns:
		df[date_col] = pd.to_datetime(df[date_col])

	metrics = [c for c in ["sessions", "conversions", "revenue", "cost", "conversion_rate", "margin"] if c in df.columns]
	dimensions = [c for c in ["channel", "region", "product"] if c in df.columns]

	# Global analysis (aggregated by date)
	if date_col in df.columns and metrics:
		g = df.groupby(date_col)[metrics].sum().sort_index()
		for m in metrics:
			y = g[m].astype(float)
			resid, z = stl_residual_anomalies(y)
			outliers = z[np.abs(z) > 2.5]
			if not outliers.empty:
				impact = float(np.sum(np.abs(outliers)))
				insights.append(Insight(
					metric=m,
					dimension="(all)",
					segment="(all)",
					description=f"Anomalies detected in {m} on {list(outliers.index.date)}",
					impact=impact,
					urgency=0.7,
					confidence=0.7,
					kind="anomaly",
					context={"dates": [str(d.date()) for d in outliers.index]}
				))
			# Change points
			bkpts = change_points(y, window=min(10, max(5, len(y)//3)), z_thresh=2.5)
			if bkpts:
				insights.append(Insight(
					metric=m,
					dimension="(all)",
					segment="(all)",
					description=f"Change points in {m} at indices {bkpts}",
					impact=float(len(bkpts)),
					urgency=0.6,
					confidence=0.65,
					kind="change_point",
					context={"indices": bkpts}
				))

	# Segment drill-down: which dimension segments contribute most to anomalies
	for dim in dimensions:
		try:
			for seg, seg_df in df.groupby(dim):
				if date_col in seg_df.columns and metrics:
					g = seg_df.groupby(date_col)[metrics].sum().sort_index()
					for m in metrics:
						y = g[m].astype(float)
						_, z = stl_residual_anomalies(y)
						outliers = z[np.abs(z) > 2.5]
						if not outliers.empty:
							impact = float(np.sum(np.abs(outliers)))
							insights.append(Insight(
								metric=m,
								dimension=dim,
								segment=str(seg),
								description=f"Segment {dim}={seg} shows anomalies in {m} on {list(outliers.index.date)}",
								impact=impact,
								urgency=0.75,
								confidence=0.7,
								kind="anomaly",
								context={"dates": [str(d.date()) for d in outliers.index]}
							))
		except Exception:
			continue

	# Efficiency insights
	if set(["revenue", "cost"]).issubset(df.columns):
		by = df.groupby(date_col).agg({"revenue": "sum", "cost": "sum"}).sort_index()
		if len(by) >= 14:
			roas = (by["revenue"] / (by["cost"] + 1e-6)).replace([np.inf, -np.inf], np.nan).dropna()
			if len(roas) >= 14:
				_, z = stl_residual_anomalies(roas)
				outliers = z[np.abs(z) > 2.5]
				if not outliers.empty:
					insights.append(Insight(
						metric="ROAS",
						dimension="(all)",
						segment="(all)",
						description=f"Marketing efficiency shifts (ROAS anomalies) on {list(outliers.index.date)}",
						impact=float(np.sum(np.abs(outliers))),
						urgency=0.7,
						confidence=0.68,
						kind="efficiency",
						context={}
					))

	# Rank by a simple score
	insights = sorted(insights, key=lambda x: (x.impact * 0.5 + x.urgency * 0.3 + x.confidence * 0.2), reverse=True)
	return insights


def analyze_auto(df: pd.DataFrame, date_col: str = "date") -> List[Insight]:
	"""Autonomously pick numeric metrics and categorical dimensions, then analyze."""
	insights: List[Insight] = []
	local_df = df.copy()
	if date_col in local_df.columns:
		local_df[date_col] = pd.to_datetime(local_df[date_col], errors="coerce")
		local_df = local_df.dropna(subset=[date_col])

	numeric_cols = [c for c in local_df.select_dtypes(include=[np.number]).columns if c != date_col]
	cat_cols = [c for c in local_df.select_dtypes(include=["object", "category"]).columns]

	# Global analysis
	if date_col in local_df.columns and numeric_cols:
		g = local_df.groupby(date_col)[numeric_cols].sum(min_count=1).sort_index()
		for m in numeric_cols:
			y = g[m].astype(float).fillna(0.0)
			if len(y) < 7:
				continue
			_, z = stl_residual_anomalies(y)
			outliers = z[np.abs(z) > 2.5]
			if not outliers.empty:
				insights.append(Insight(
					metric=m,
					dimension="(all)",
					segment="(all)",
					description=f"Anomalies detected in {m} on {list(outliers.index.date)}",
					impact=float(np.sum(np.abs(outliers))),
					urgency=0.7,
					confidence=0.65,
					kind="anomaly",
					context={}
				))
			bkpts = change_points(y, window=min(10, max(5, len(y)//3)), z_thresh=2.5)
			if bkpts:
				insights.append(Insight(
					metric=m,
					dimension="(all)",
					segment="(all)",
					description=f"Change points in {m} at indices {bkpts}",
					impact=float(len(bkpts)),
					urgency=0.6,
					confidence=0.6,
					kind="change_point",
					context={}
				))

	# Drilldown by categorical columns
	for dim in cat_cols:
		try:
			for seg, seg_df in local_df.groupby(dim):
				if date_col not in seg_df.columns:
					continue
				g = seg_df.groupby(date_col)[numeric_cols].sum(min_count=1).sort_index()
				for m in numeric_cols:
					y = g[m].astype(float).fillna(0.0)
					if len(y) < 7:
						continue
					_, z = stl_residual_anomalies(y)
					outliers = z[np.abs(z) > 2.5]
					if not outliers.empty:
						insights.append(Insight(
							metric=m,
							dimension=dim,
							segment=str(seg),
							description=f"Segment {dim}={seg} shows anomalies in {m}",
							impact=float(np.sum(np.abs(outliers))),
							urgency=0.75,
							confidence=0.65,
							kind="anomaly",
							context={}
						))
		except Exception:
			continue

	# Rank
	insights = sorted(insights, key=lambda x: (x.impact * 0.5 + x.urgency * 0.3 + x.confidence * 0.2), reverse=True)
	return insights


def discover_patterns(df: pd.DataFrame, date_col: str = "date") -> List[Pattern]:
	"""Discover recurring patterns: seasonality strength, trend direction, correlations, recurring segments."""
	patterns: List[Pattern] = []
	local_df = df.copy()
	if date_col in local_df.columns:
		local_df[date_col] = pd.to_datetime(local_df[date_col], errors="coerce")
		local_df = local_df.dropna(subset=[date_col])

	numeric_cols = [c for c in local_df.select_dtypes(include=[np.number]).columns if c != date_col]
	cat_cols = [c for c in local_df.select_dtypes(include=["object", "category"]).columns]

	# Global seasonality and trend per metric
	if date_col in local_df.columns and numeric_cols:
		g = local_df.groupby(date_col)[numeric_cols].sum(min_count=1).sort_index()
		for m in numeric_cols:
			y = g[m].astype(float).fillna(0.0)
			if len(y) < 14:
				continue
			# Simple seasonal strength estimation using rolling mean
			period = max(7, min(30, max(7, len(y)//6)))
			rolling_mean = y.rolling(window=period, center=True).mean()
			seasonal_var = np.var(rolling_mean.dropna())
			total_var = np.var(y) + 1e-9
			seasonal_strength = float(min(1.0, max(0.0, seasonal_var / total_var)))
			trend_slope = float(np.polyfit(np.arange(len(y)), y.values, 1)[0])
			if seasonal_strength > 0.15:
				patterns.append(Pattern(
					pattern="seasonality",
					metric=m,
					dimension="(all)",
					segment="(all)",
					description=f"{m} shows seasonality (strength={seasonal_strength:.2f})",
					strength=seasonal_strength,
					context={}
				))
			patterns.append(Pattern(
				pattern="trend",
				metric=m,
				dimension="(all)",
				segment="(all)",
				description=f"{m} trend slope={trend_slope:.2f}",
				strength=float(abs(trend_slope)),
				context={}
			))

	# Correlations between metrics
	if len(numeric_cols) >= 2 and date_col in local_df.columns:
		g = local_df.groupby(date_col)[numeric_cols].sum(min_count=1).sort_index()
		corr = g.corr().abs()
		corr.values[[np.arange(corr.shape[0])]*2] = 0.0
		pairs = []
		for i in range(len(corr.index)):
			for j in range(i+1, len(corr.columns)):
				pairs.append((corr.index[i], corr.columns[j], float(corr.iloc[i, j])))
		pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
		for a, b, r in pairs:
			if r > 0.6:
				patterns.append(Pattern(
					pattern="correlation",
					metric=f"{a}~{b}",
					dimension="(all)",
					segment="(all)",
					description=f"High correlation between {a} and {b} (r={r:.2f})",
					strength=r,
					context={}
				))

	# Recurring anomalous segments (segments with multiple anomalies across time)
	if cat_cols and date_col in local_df.columns and numeric_cols:
		for dim in cat_cols:
			counts: Dict[str, int] = {}
			for seg, seg_df in local_df.groupby(dim):
				g = seg_df.groupby(date_col)[numeric_cols].sum(min_count=1).sort_index()
				flagged = 0
				for m in numeric_cols:
					y = g[m].astype(float).fillna(0.0)
					if len(y) < 14:
						continue
					_, z = stl_residual_anomalies(y)
					outliers = z[np.abs(z) > 2.5]
					if not outliers.empty:
						flagged += 1
				counts[str(seg)] = counts.get(str(seg), 0) + flagged
			# top recurring
			top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
			for seg, cnt in top:
				if cnt > 0:
					patterns.append(Pattern(
						pattern="recurring_segment",
						metric="(various)",
						dimension=dim,
						segment=str(seg),
						description=f"Segment {dim}={seg} repeatedly shows anomalies across metrics",
						strength=float(cnt),
						context={}
					))

	# Rank patterns by strength
	patterns = sorted(patterns, key=lambda p: p.strength, reverse=True)
	return patterns


def analyze_data_autonomously(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	"""Autonomously analyze data to find insights and patterns."""
	insights = []
	patterns = []
	
	# Identify numeric columns for analysis
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
	
	# Analyze each numeric metric
	for metric in numeric_cols:
		series = pd.to_numeric(df[metric], errors='coerce').dropna()
		if len(series) < 10:
			continue
		
		# Detect anomalies
		anomalies = detect_anomalies(series)
		for anomaly in anomalies:
			confidence = anomaly.get('confidence', 0.8)
			description = anomaly.get('description', f"Anomaly detected in {metric}")
			
			# Add business context based on metric name
			business_context = get_business_context(metric, anomaly)
			full_description = f"{description}. {business_context}"
			
			insights.append({
				"metric": metric,
				"kind": "anomaly",
				"dimension": None,
				"segment": None,
				"description": full_description,
				"confidence": confidence,
				"details": anomaly
			})
		
		# Detect change points
		changes = change_points(series)
		for change in changes:
			confidence = change.get('confidence', 0.8)
			description = change.get('description', f"Change point detected in {metric}")
			
			# Add business context
			business_context = get_business_context(metric, change)
			full_description = f"{description}. {business_context}"
			
			insights.append({
				"metric": metric,
				"kind": "change_point",
				"dimension": None,
				"segment": None,
				"description": full_description,
				"confidence": confidence,
				"details": change
			})
		
		# Detect trends
		if len(series) >= 20:
			slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(series)), series)
			trend_strength = abs(r_value)
			if trend_strength > 0.3 and p_value < 0.05:
				trend_direction = "increasing" if slope > 0 else "decreasing"
				confidence = min(trend_strength, 1.0)
				
				# Calculate trend magnitude
				trend_magnitude = abs(slope) * len(series) / series.mean() * 100 if series.mean() != 0 else 0
				
				description = f"Strong {trend_direction} trend detected in {metric} (R²={trend_strength:.2f}, magnitude: {trend_magnitude:.1f}% change over period)"
				business_context = get_trend_business_context(metric, trend_direction, trend_magnitude)
				full_description = f"{description}. {business_context}"
				
				insights.append({
					"metric": metric,
					"kind": "trend",
					"dimension": None,
					"segment": None,
					"description": full_description,
					"confidence": confidence,
					"details": {"slope": slope, "r_squared": trend_strength, "p_value": p_value, "trend_magnitude": trend_magnitude}
				})
	
	# Detect patterns across metrics
	patterns.extend(detect_patterns(df))
	
	return insights, patterns


def get_business_context(metric: str, anomaly: Dict[str, Any]) -> str:
	"""Add business context to anomaly descriptions."""
	metric_lower = metric.lower()
	direction = anomaly.get('direction', 'change')
	deviation = abs(anomaly.get('deviation_percent', 0))
	
	if 'revenue' in metric_lower or 'sales' in metric_lower:
		if direction == 'spike':
			return f"This {deviation:.1f}% revenue increase could indicate successful marketing campaigns, seasonal demand, or pricing optimization."
		else:
			return f"This {deviation:.1f}% revenue decrease may signal market challenges, competitive pressure, or operational issues requiring immediate attention."
	
	elif 'conversion' in metric_lower:
		if direction == 'spike':
			return f"This {deviation:.1f}% conversion improvement suggests effective funnel optimization or high-quality traffic acquisition."
		else:
			return f"This {deviation:.1f}% conversion decline indicates potential funnel issues, poor traffic quality, or technical problems."
	
	elif 'cost' in metric_lower or 'cpc' in metric_lower:
		if direction == 'spike':
			return f"This {deviation:.1f}% cost increase may indicate market competition, seasonal pricing, or inefficient ad spend requiring budget optimization."
		else:
			return f"This {deviation:.1f}% cost decrease suggests improved efficiency, better targeting, or favorable market conditions."
	
	elif 'session' in metric_lower or 'traffic' in metric_lower:
		if direction == 'spike':
			return f"This {deviation:.1f}% traffic increase could result from successful campaigns, viral content, or improved SEO performance."
		else:
			return f"This {deviation:.1f}% traffic decline may indicate campaign fatigue, technical issues, or competitive pressure."
	
	else:
		return f"This {deviation:.1f}% {direction} in {metric} represents a significant deviation from normal patterns and should be investigated."


def get_trend_business_context(metric: str, direction: str, magnitude: float) -> str:
	"""Add business context to trend descriptions."""
	metric_lower = metric.lower()
	
	if 'revenue' in metric_lower or 'sales' in metric_lower:
		if direction == 'increasing':
			return f"Revenue growth of {magnitude:.1f}% suggests strong business performance and market demand."
		else:
			return f"Revenue decline of {magnitude:.1f}% indicates potential market challenges requiring strategic intervention."
	
	elif 'conversion' in metric_lower:
		if direction == 'increasing':
			return f"Conversion improvement of {magnitude:.1f}% shows effective optimization efforts and user experience enhancements."
		else:
			return f"Conversion decline of {magnitude:.1f}% suggests funnel issues that need immediate optimization."
	
	elif 'cost' in metric_lower or 'cpc' in metric_lower:
		if direction == 'increasing':
			return f"Cost increase of {magnitude:.1f}% may indicate rising competition or inefficient spend requiring optimization."
		else:
			return f"Cost decrease of {magnitude:.1f}% shows improved efficiency and better resource allocation."
	
	else:
		return f"This {direction} trend of {magnitude:.1f}% in {metric} represents a significant pattern that should be monitored and acted upon."


def detect_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
	"""Detect patterns across the dataset."""
	patterns = []
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	
	if len(numeric_cols) < 2:
		return patterns
	
	# Calculate correlations
	corr_matrix = df[numeric_cols].corr()
	
	# Find top correlated pairs
	correlations = []
	for i in range(len(corr_matrix.columns)):
		for j in range(i+1, len(corr_matrix.columns)):
			corr_val = corr_matrix.iloc[i, j]
			if abs(corr_val) > 0.5:  # Strong correlation threshold
				correlations.append({
					"metric1": corr_matrix.columns[i],
					"metric2": corr_matrix.columns[j],
					"correlation": corr_val,
					"confidence": abs(corr_val)
				})
	
	# Sort by absolute correlation
	correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
	
	# Add top correlations as patterns
	for corr in correlations[:3]:  # Top 3 correlations
		strength = "strong" if abs(corr["correlation"]) > 0.7 else "moderate"
		direction = "positive" if corr["correlation"] > 0 else "negative"
		description = f"{strength.title()} {direction} correlation between {corr['metric1']} and {corr['metric2']} (r={corr['correlation']:.2f})"
		
		# Add business interpretation
		business_interpretation = get_correlation_business_context(corr['metric1'], corr['metric2'], corr['correlation'])
		full_description = f"{description}. {business_interpretation}"
		
		patterns.append({
			"type": "correlation",
			"description": full_description,
			"confidence": corr["confidence"],
			"details": corr
		})
	
	# Detect seasonal patterns (simplified)
	for metric in numeric_cols[:3]:  # Check first 3 metrics
		series = pd.to_numeric(df[metric], errors='coerce').dropna()
		if len(series) >= 30:
			# Simple seasonality detection using autocorrelation
			autocorr = series.autocorr(lag=7)  # Weekly pattern
			if abs(autocorr) > 0.3:
				description = f"Weekly seasonal pattern detected in {metric} (autocorr={autocorr:.2f})"
				business_context = get_seasonality_business_context(metric, autocorr)
				full_description = f"{description}. {business_context}"
				
				patterns.append({
					"type": "seasonal",
					"description": full_description,
					"confidence": abs(autocorr),
					"details": {"metric": metric, "autocorr": autocorr, "period": "weekly"}
				})
	
	return patterns


def get_correlation_business_context(metric1: str, metric2: str, correlation: float) -> str:
	"""Add business context to correlation patterns."""
	metric1_lower = metric1.lower()
	metric2_lower = metric2.lower()
	
	if 'revenue' in metric1_lower and 'conversion' in metric2_lower or 'revenue' in metric2_lower and 'conversion' in metric1_lower:
		if correlation > 0:
			return "This suggests that conversion rate improvements directly impact revenue growth, validating the importance of funnel optimization."
		else:
			return "This inverse relationship may indicate pricing strategy effects or market dynamics affecting both metrics."
	
	elif 'cost' in metric1_lower and 'conversion' in metric2_lower or 'cost' in metric2_lower and 'conversion' in metric1_lower:
		if correlation > 0:
			return "Higher costs correlating with better conversions suggests effective investment in quality traffic or optimization."
		else:
			return "Lower costs with better conversions indicates improved efficiency and optimization success."
	
	else:
		return f"This correlation between {metric1} and {metric2} provides insights for cross-metric optimization strategies."


def get_seasonality_business_context(metric: str, autocorr: float) -> str:
	"""Add business context to seasonality patterns."""
	metric_lower = metric.lower()
	
	if 'revenue' in metric_lower or 'sales' in metric_lower:
		return "Weekly revenue patterns suggest consistent business cycles that can inform campaign scheduling and resource allocation."
	
	elif 'conversion' in metric_lower:
		return "Weekly conversion patterns indicate optimal timing for campaigns and user engagement strategies."
	
	elif 'traffic' in metric_lower or 'session' in metric_lower:
		return "Weekly traffic patterns help optimize campaign timing and content scheduling for maximum engagement."
	
	else:
		return f"Weekly patterns in {metric} provide opportunities for strategic timing optimization."


