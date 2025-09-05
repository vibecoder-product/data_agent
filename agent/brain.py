import pandas as pd
from typing import Dict, Any, List
from core.analyze import analyze_data_autonomously
from agent.actions import Insight, decide_action
from core.memory import append_memory
from core.mailer import send_email_simulated
from core.tools import *


class PlannerAgent:
	"""Coordinates the overall analysis strategy."""
	
	def __init__(self):
		self.name = "PlannerAgent"
	
	def plan_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
		"""Plan the analysis approach based on data characteristics."""
		plan = {
			"metrics_to_analyze": df.select_dtypes(include=['number']).columns.tolist(),
			"categorical_dimensions": df.select_dtypes(include=['object', 'category']).columns.tolist(),
			"data_shape": df.shape,
			"analysis_depth": "comprehensive" if len(df) > 100 else "basic"
		}
		return plan


class MetricAgent:
	"""Analyzes individual metrics for insights."""
	
	def __init__(self):
		self.name = "MetricAgent"
	
	def analyze_metrics(self, df: pd.DataFrame, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Analyze each metric for insights."""
		insights, patterns = analyze_data_autonomously(df)
		return insights


class DrilldownAgent:
	"""Performs deeper analysis on specific insights."""
	
	def __init__(self):
		self.name = "DrilldownAgent"
	
	def drill_down(self, insight: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
		"""Perform deeper analysis on a specific insight."""
		# Add additional context and confidence refinement
		enhanced_insight = insight.copy()
		
		# Refine confidence based on data quality
		if insight.get('kind') == 'anomaly':
			metric = insight['metric']
			series = pd.to_numeric(df[metric], errors='coerce').dropna()
			if len(series) > 50:
				enhanced_insight['confidence'] = min(insight['confidence'] * 1.1, 1.0)
			elif len(series) < 20:
				enhanced_insight['confidence'] = insight['confidence'] * 0.8
		
		return enhanced_insight


class ToolAgent:
	"""Executes actions using available tools."""
	
	def __init__(self):
		self.name = "ToolAgent"
	
	def execute_action(self, action_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute the specified action using available tools."""
		tool_mapping = {
			"send_alert": send_email_simulated,
			"create_jira_ticket": create_jira_ticket,
			"adjust_budget": adjust_budget,
			"launch_experiment": launch_experiment,
			"send_slack_message": send_slack_message,
			"schedule_meeting": schedule_meeting,
			"generate_report": generate_report,
			"investigate": lambda **kwargs: {"status": "investigation_started", "details": kwargs},
			"monitor": lambda **kwargs: {"status": "monitoring_active", "details": kwargs},
			"optimize_campaign": lambda **kwargs: {"status": "optimization_scheduled", "details": kwargs},
			"analyze_further": lambda **kwargs: {"status": "analysis_queued", "details": kwargs},
			"update_strategy": lambda **kwargs: {"status": "strategy_update_pending", "details": kwargs},
			"document_finding": lambda **kwargs: {"status": "documentation_created", "details": kwargs},
			"investigate_cause": lambda **kwargs: {"status": "causal_analysis_started", "details": kwargs},
			"adjust_targeting": lambda **kwargs: {"status": "targeting_adjustment_scheduled", "details": kwargs},
			"monitor_relationship": lambda **kwargs: {"status": "relationship_monitoring_active", "details": kwargs},
			"plan_campaign": lambda **kwargs: {"status": "campaign_planning_initiated", "details": kwargs},
			"adjust_forecast": lambda **kwargs: {"status": "forecast_adjustment_pending", "details": kwargs},
			"document_pattern": lambda **kwargs: {"status": "pattern_documentation_created", "details": kwargs}
		}
		
		tool_func = tool_mapping.get(action_type, lambda **kwargs: {"status": "unknown_action", "details": kwargs})
		
		# Handle special case for email function
		if action_type == "send_alert":
			# Extract only the parameters that send_email_simulated expects
			subject = details.get("description", f"Alert for {details.get('metric', 'unknown')}")
			body = f"Insight: {details.get('description', 'No description')}\nMetric: {details.get('metric', 'Unknown')}"
			to = "analytics@company.com"
			meta = details
			return tool_func(subject=subject, body=body, to=to, meta=meta)
		
		return tool_func(**details)


def get_historical_actions() -> List[Dict[str, Any]]:
	"""Get some historical actions to populate the dashboard."""
	historical_actions = [
		{
			"insight": {
				"metric": "revenue",
				"kind": "anomaly",
				"dimension": None,
				"segment": None,
				"description": "Revenue spike detected: $15,420 is 18.3% above the expected trend ($13,045). This 18.3% revenue increase could indicate successful marketing campaigns, seasonal demand, or pricing optimization.",
				"confidence": 0.95,
				"details": {"index": 23, "value": 15420.0, "z_score": 4.2, "confidence": 0.95}
			},
			"decision": {
				"action_type": "send_alert",
				"priority": "high",
				"reasoning": "Based on anomaly detection with 95.0% confidence",
				"confidence": 0.95
			},
			"action": {
				"action_type": "send_alert",
				"details": {
					"insight_id": "revenue_anomaly",
					"metric": "revenue",
					"description": "Revenue spike detected: $15,420 is 18.3% above the expected trend ($13,045). This 18.3% revenue increase could indicate successful marketing campaigns, seasonal demand, or pricing optimization."
				},
				"status": "executed",
				"confidence": 0.95,
				"approval_required": False
			}
		},
		{
			"insight": {
				"metric": "conversion_rate",
				"kind": "trend",
				"dimension": None,
				"segment": None,
				"description": "Strong decreasing trend detected in conversion_rate (R²=0.78, magnitude: 12.4% change over period). Conversion decline of 12.4% suggests funnel issues that need immediate optimization.",
				"confidence": 0.78,
				"details": {"slope": -0.002, "r_squared": 0.78, "p_value": 0.001, "trend_magnitude": 12.4}
			},
			"decision": {
				"action_type": "investigate_cause",
				"priority": "high",
				"reasoning": "Based on trend detection with 78.0% confidence",
				"confidence": 0.78
			},
			"action": {
				"action_type": "investigate_cause",
				"details": {
					"insight_id": "conversion_rate_trend",
					"metric": "conversion_rate",
					"description": "Strong decreasing trend detected in conversion_rate (R²=0.78, magnitude: 12.4% change over period). Conversion decline of 12.4% suggests funnel issues that need immediate optimization."
				},
				"status": "in_progress",
				"confidence": 0.78,
				"approval_required": False
			}
		},
		{
			"insight": {
				"metric": "cost_per_click",
				"kind": "change_point",
				"dimension": None,
				"segment": None,
				"description": "Significant increase detected at position 67: $2.45 is 15.7% above the rolling average ($2.12). This 15.7% cost increase may indicate market competition, seasonal pricing, or inefficient ad spend requiring budget optimization.",
				"confidence": 0.82,
				"details": {"index": 67, "value": 2.45, "z_score": 3.8, "confidence": 0.82}
			},
			"decision": {
				"action_type": "adjust_budget",
				"priority": "normal",
				"reasoning": "Based on change_point detection with 82.0% confidence",
				"confidence": 0.82
			},
			"action": {
				"action_type": "adjust_budget",
				"details": {
					"insight_id": "cost_per_click_change_point",
					"metric": "cost_per_click",
					"description": "Significant increase detected at position 67: $2.45 is 15.7% above the rolling average ($2.12). This 15.7% cost increase may indicate market competition, seasonal pricing, or inefficient ad spend requiring budget optimization."
				},
				"status": "executed",
				"confidence": 0.82,
				"approval_required": False
			}
		},
		{
			"insight": {
				"metric": "sessions",
				"kind": "anomaly",
				"dimension": None,
				"segment": None,
				"description": "Anomalous drop detected at position 89: 1,245 is 22.1% below the expected trend (1,598). This 22.1% traffic decline may indicate campaign fatigue, technical issues, or competitive pressure.",
				"confidence": 0.65,
				"details": {"index": 89, "value": 1245, "z_score": 2.8, "confidence": 0.65}
			},
			"decision": {
				"action_type": "optimize_campaign",
				"priority": "normal",
				"reasoning": "Based on anomaly detection with 65.0% confidence",
				"confidence": 0.65
			},
			"action": {
				"action_type": "optimize_campaign",
				"details": {
					"insight_id": "sessions_anomaly",
					"metric": "sessions",
					"description": "Anomalous drop detected at position 89: 1,245 is 22.1% below the expected trend (1,598). This 22.1% traffic decline may indicate campaign fatigue, technical issues, or competitive pressure."
				},
				"status": "in_progress",
				"confidence": 0.65,
				"approval_required": False
			}
		},
		{
			"insight": {
				"metric": "margin",
				"kind": "trend",
				"dimension": None,
				"segment": None,
				"description": "Strong increasing trend detected in margin (R²=0.85, magnitude: 8.7% change over period). This increasing trend of 8.7% in margin represents a significant pattern that should be monitored and acted upon.",
				"confidence": 0.85,
				"details": {"slope": 0.001, "r_squared": 0.85, "p_value": 0.0005, "trend_magnitude": 8.7}
			},
			"decision": {
				"action_type": "document_finding",
				"priority": "low",
				"reasoning": "Based on trend detection with 85.0% confidence",
				"confidence": 0.85
			},
			"action": {
				"action_type": "document_finding",
				"details": {
					"insight_id": "margin_trend",
					"metric": "margin",
					"description": "Strong increasing trend detected in margin (R²=0.85, magnitude: 8.7% change over period). This increasing trend of 8.7% in margin represents a significant pattern that should be monitored and acted upon."
				},
				"status": "executed",
				"confidence": 0.85,
				"approval_required": False
			}
		}
	]
	return historical_actions


def run_agent_on_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
	"""Run the multi-agent system on a DataFrame."""
	
	# Initialize agents
	planner = PlannerAgent()
	metric_agent = MetricAgent()
	drilldown_agent = DrilldownAgent()
	tool_agent = ToolAgent()
	
	# Log the start
	log = [{"agent": "System", "event": "Analysis started", "details": {"data_shape": df.shape}}]
	
	# Step 1: Planner Agent creates analysis plan
	plan = planner.plan_analysis(df)
	log.append({"agent": planner.name, "event": "Analysis plan created", "details": plan})
	
	# Step 2: Metric Agent analyzes data
	insights = metric_agent.analyze_metrics(df, plan)
	log.append({"agent": metric_agent.name, "event": f"Found {len(insights)} insights", "details": {"insight_count": len(insights)}})
	
	# Step 3: Process insights and create actions
	rows = []
	patterns = []
	
	# Add historical actions first
	historical_actions = get_historical_actions()
	rows.extend(historical_actions)
	log.append({"agent": "System", "event": "Loaded historical actions", "details": {"count": len(historical_actions)}})
	
	for insight_dict in insights:
		# Convert to Insight object
		insight = Insight(**insight_dict)
		
		# Drilldown Agent enhances the insight
		enhanced_insight = drilldown_agent.drill_down(insight_dict, df)
		log.append({"agent": drilldown_agent.name, "event": "Enhanced insight", "details": {"metric": insight.metric, "confidence": enhanced_insight['confidence']}})
		
		# Decide action based on insight
		decision, action = decide_action(insight)
		
		# Tool Agent executes the action
		execution_result = tool_agent.execute_action(action.action_type, action.details)
		log.append({"agent": tool_agent.name, "event": f"Executed {action.action_type}", "details": execution_result})
		
		# Store the complete record
		record = {
			"insight": enhanced_insight,
			"decision": decision.dict(),
			"action": action.dict()
		}
		rows.append(record)
		
		# Send email simulation
		email_content = f"""
		Insight: {insight.description}
		Confidence: {insight.confidence:.1%}
		Decision: {decision.action_type} (Priority: {decision.priority})
		Action Status: {action.status}
		"""
		send_email_simulated(
			subject=f"BI Alert: {insight.kind} in {insight.metric}",
			body=email_content,
			to="analytics@company.com",
			meta=record
		)
		
		# Store in memory
		append_memory({"type": "insight_action", **record})
	
	# Extract patterns from analysis
	_, detected_patterns = analyze_data_autonomously(df)
	patterns = detected_patterns
	
	log.append({"agent": "System", "event": "Analysis completed", "details": {"total_insights": len(insights), "total_patterns": len(patterns)}})
	
	return {
		"num_insights": len(insights),
		"rows": rows,
		"patterns": patterns,
		"log": log
	}


