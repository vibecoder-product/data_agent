from typing import Dict, Any, Optional
from pydantic import BaseModel
import random


class Insight(BaseModel):
	metric: str
	kind: str  # anomaly, trend, pattern, etc.
	dimension: Optional[str] = None
	segment: Optional[str] = None
	description: str
	confidence: float = 0.8  # Default confidence score


class Decision(BaseModel):
	action_type: str
	priority: str = "normal"  # low, normal, high, critical
	reasoning: str
	confidence: float = 0.8  # Default confidence score


class Action(BaseModel):
	action_type: str
	details: Dict[str, Any]
	status: str = "pending_approval"  # pending_approval, in_progress, executed, failed, approved, rejected
	confidence: float = 0.8  # Default confidence score
	approval_required: bool = True


def decide_action(insight: Insight) -> tuple[Decision, Action]:
	"""Decide what action to take based on insight."""
	
	# Define action mappings with confidence scores
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
	
	# Get available actions for this insight type
	available_actions = action_mappings.get(insight.kind, action_mappings["anomaly"])
	
	# Select action based on insight confidence
	if insight.confidence >= 0.9:
		action_type, confidence, priority = available_actions[0]  # Highest confidence action
	elif insight.confidence >= 0.7:
		action_type, confidence, priority = available_actions[1]  # Medium confidence action
	else:
		action_type, confidence, priority = available_actions[2]  # Lower confidence action
	
	# Adjust confidence based on insight confidence
	final_confidence = min(insight.confidence * confidence, 1.0)
	
	# Determine status based on confidence
	if final_confidence >= 0.9:
		status = "executed"
		approval_required = False
	elif final_confidence >= 0.7:
		status = "in_progress"
		approval_required = False
	else:
		status = "pending_approval"
		approval_required = True
	
	decision = Decision(
		action_type=action_type,
		priority=priority,
		reasoning=f"Based on {insight.kind} detection with {insight.confidence:.1%} confidence",
		confidence=final_confidence
	)
	
	action = Action(
		action_type=action_type,
		details={
			"insight_id": f"{insight.metric}_{insight.kind}",
			"metric": insight.metric,
			"description": insight.description
		},
		status=status,
		confidence=final_confidence,
		approval_required=approval_required
	)
	
	return decision, action


def approve_action(action: Action) -> Action:
	"""Approve a pending action."""
	if action.status == "pending_approval":
		action.status = "approved"
		action.approval_required = False
	return action


def reject_action(action: Action) -> Action:
	"""Reject a pending action."""
	if action.status == "pending_approval":
		action.status = "rejected"
		action.approval_required = False
	return action


