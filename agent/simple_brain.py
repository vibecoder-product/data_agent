import pandas as pd
from typing import Dict, Any, List
from core.analyze import analyze_data_autonomously
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
            "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "analysis_depth": "comprehensive" if len(df) > 1000 else "standard"
        }
        return plan

class MetricAgent:
    """Performs autonomous data analysis."""
    
    def __init__(self):
        self.name = "MetricAgent"
    
    def analyze_metrics(self, df: pd.DataFrame, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze metrics autonomously."""
        insights, patterns = analyze_data_autonomously(df)
        # Store patterns for later use
        self.patterns = patterns
        return insights

class DrilldownAgent:
    """Enhances insights with additional context."""
    
    def __init__(self):
        self.name = "DrilldownAgent"
    
    def enhance_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance insights with additional analysis."""
        enhanced = []
        for insight in insights:
            # Add business context
            if insight.get('type') == 'anomaly':
                insight['business_impact'] = "Potential operational issue requiring investigation"
            elif insight.get('type') == 'trend':
                insight['business_impact'] = "Performance trend that may require strategic response"
            
            # Refine confidence based on additional factors
            base_confidence = insight.get('confidence', 0.8)
            if insight.get('severity') == 'high':
                base_confidence = min(0.95, base_confidence + 0.1)
            insight['confidence'] = base_confidence
            
            enhanced.append(insight)
        return enhanced

class ToolAgent:
    """Executes actions using various business tools."""
    
    def __init__(self):
        self.name = "ToolAgent"
        self.tool_mapping = {
            "create_jira_ticket": create_jira_ticket,
            "adjust_budget": adjust_budget,
            "launch_experiment": launch_experiment,
            "send_slack_message": send_slack_message,
            "schedule_meeting": schedule_meeting,
            "generate_report": generate_report,
            "send_alert": send_email_simulated
        }
    
    def execute_action(self, action_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified action."""
        if action_type in self.tool_mapping:
            tool_func = self.tool_mapping[action_type]
            try:
                if action_type == "send_alert":
                    # Special handling for email alerts
                    result = tool_func(
                        subject=details.get('subject', 'BI Alert'),
                        body=details.get('body', 'Automated business intelligence alert'),
                        to=details.get('to', 'admin@company.com'),
                        meta=details.get('meta', {})
                    )
                else:
                    result = tool_func(**details)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Unknown action type: {action_type}"}

def decide_action(insight: Dict[str, Any]) -> Dict[str, Any]:
    """Decide what action to take based on insight."""
    insight_type = insight.get('type', 'unknown')
    confidence = insight.get('confidence', 0.8)
    severity = insight.get('severity', 'medium')
    
    # Simple heuristic-based action mapping
    if insight_type == 'anomaly':
        if severity == 'high' and confidence > 0.8:
            action_type = "send_alert"
            priority = "high"
            status = "executed"
            approval_required = False
        elif severity == 'medium' and confidence > 0.6:
            action_type = "create_jira_ticket"
            priority = "medium"
            status = "pending_approval"
            approval_required = True
        else:
            action_type = "generate_report"
            priority = "low"
            status = "pending_approval"
            approval_required = True
    elif insight_type == 'trend':
        if confidence > 0.7:
            action_type = "schedule_meeting"
            priority = "medium"
            status = "pending_approval"
            approval_required = True
        else:
            action_type = "generate_report"
            priority = "low"
            status = "pending_approval"
            approval_required = True
    else:
        action_type = "generate_report"
        priority = "low"
        status = "pending_approval"
        approval_required = True
    
    return {
        "action_type": action_type,
        "priority": priority,
        "reasoning": f"Based on {insight_type} with {confidence:.1%} confidence",
        "confidence": confidence,
        "status": status,
        "approval_required": approval_required,
        "details": {
            "insight_id": insight.get('id', 'unknown'),
            "metric": insight.get('metric', 'unknown'),
            "description": insight.get('description', 'No description'),
            "subject": f"BI Alert: {insight.get('metric', 'Unknown Metric')}",
            "body": f"Automated alert: {insight.get('description', 'No description')}",
            "to": "admin@company.com"
        }
    }

def run_agent_on_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the complete autonomous analysis pipeline."""
    logs = []
    
    # Initialize agents
    planner = PlannerAgent()
    metric_agent = MetricAgent()
    drilldown_agent = DrilldownAgent()
    tool_agent = ToolAgent()
    
    logs.append(f"[{planner.name}] Starting analysis planning...")
    
    # Plan analysis
    plan = planner.plan_analysis(df)
    logs.append(f"[{planner.name}] Analysis plan created: {len(plan['metrics_to_analyze'])} metrics, {len(plan['categorical_dimensions'])} dimensions")
    
    # Analyze metrics
    logs.append(f"[{metric_agent.name}] Starting metric analysis...")
    insights = metric_agent.analyze_metrics(df, plan)
    logs.append(f"[{metric_agent.name}] Found {len(insights)} insights")
    
    # Enhance insights
    logs.append(f"[{drilldown_agent.name}] Enhancing insights...")
    enhanced_insights = drilldown_agent.enhance_insights(insights)
    logs.append(f"[{drilldown_agent.name}] Enhanced {len(enhanced_insights)} insights")
    
    # Generate decisions and actions
    rows = []
    for insight in enhanced_insights:
        decision = decide_action(insight)
        action = decision.copy()
        
        # Execute high-confidence actions immediately
        if not action.get('approval_required', True):
            logs.append(f"[{tool_agent.name}] Executing {action['action_type']}...")
            execution_result = tool_agent.execute_action(action['action_type'], action['details'])
            if execution_result['success']:
                action['status'] = 'executed'
                logs.append(f"[{tool_agent.name}] Successfully executed {action['action_type']}")
            else:
                action['status'] = 'failed'
                logs.append(f"[{tool_agent.name}] Failed to execute {action['action_type']}: {execution_result['error']}")
        
        # Store in memory
        record = {
            "insight": insight,
            "decision": decision,
            "action": action
        }
        append_memory({"type": "insight_action", **record})
        
        rows.append(record)
    
    # Get patterns from analysis
    patterns = getattr(metric_agent, 'patterns', [])
    
    logs.append(f"[System] Analysis complete: {len(rows)} insights processed, {len(patterns)} patterns found")
    
    return {
        "rows": rows,
        "patterns": patterns,
        "logs": logs
    }
