from typing import Dict, Any, Optional, Tuple


def decide_action(insight: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
	"""Decide what action to take based on a dict-style insight."""

	kind = insight.get("kind", "anomaly")
	insight_conf = float(insight.get("confidence", 0.8))

	action_mappings = {
		"anomaly": [
			("send_alert", 0.9, "high"),
			("investigate", 0.8, "normal"),
			("monitor", 0.6, "low")
		],
		"trend": [
			("adjust_budget", 0.85, "high"),
			("optimize_campaign", 0.8, "normal"),
			("analyze_further", 0.7, "low")
		],
		"pattern": [
			("launch_experiment", 0.9, "high"),
			("update_strategy", 0.8, "normal"),
			("document_finding", 0.6, "low")
		],
		"correlation": [
			("investigate_cause", 0.85, "high"),
			("adjust_targeting", 0.8, "normal"),
			("monitor_relationship", 0.7, "low")
		],
		"seasonal": [
			("plan_campaign", 0.9, "high"),
			("adjust_forecast", 0.8, "normal"),
			("document_pattern", 0.6, "low")
		]
	}

	available_actions = action_mappings.get(kind, action_mappings["anomaly"])
	if insight_conf >= 0.9:
		action_type, base_conf, priority = available_actions[0]
	elif insight_conf >= 0.7:
		action_type, base_conf, priority = available_actions[1]
	else:
		action_type, base_conf, priority = available_actions[2]

	final_confidence = min(insight_conf * base_conf, 1.0)
	if final_confidence >= 0.9:
		status = "executed"; approval_required = False
	elif final_confidence >= 0.7:
		status = "in_progress"; approval_required = False
	else:
		status = "pending_approval"; approval_required = True

	decision: Dict[str, Any] = {
		"action_type": action_type,
		"priority": priority,
		"reasoning": f"Based on {kind} detection with {insight_conf:.1%} confidence",
		"confidence": final_confidence,
	}

	action: Dict[str, Any] = {
		"action_type": action_type,
		"details": {
			"insight_id": f"{insight.get('metric','metric')}_{kind}",
			"metric": insight.get("metric"),
			"description": insight.get("description"),
		},
		"status": status,
		"confidence": final_confidence,
		"approval_required": approval_required,
	}

	return decision, action


def approve_action(action: Dict[str, Any]) -> Dict[str, Any]:
	"""Approve a pending action (dict)."""
	if action.get("status") == "pending_approval":
		action["status"] = "approved"
		action["approval_required"] = False
	return action


def reject_action(action: Dict[str, Any]) -> Dict[str, Any]:
	"""Reject a pending action (dict)."""
	if action.get("status") == "pending_approval":
		action["status"] = "rejected"
		action["approval_required"] = False
	return action


