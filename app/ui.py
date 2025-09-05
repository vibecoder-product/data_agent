import io
import os
import sys
from typing import Optional
from pathlib import Path
import pandas as pd
import streamlit as st

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from core.generate import generate_dummy_data, generate_and_save_csv
    from agent.brain import run_agent_on_dataframe
    from core.mailer import read_outbox
    from core.connectors import get_connector
    from agent.actions import approve_action, reject_action
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


st.set_page_config(page_title="Autonomous BI Agent", layout="wide")
st.title("Autonomous Business Intelligence Agent")


@st.cache_data
def load_csv(uploaded_bytes: bytes) -> pd.DataFrame:
	return pd.read_csv(io.BytesIO(uploaded_bytes))


def get_confidence_color(confidence: float) -> str:
	"""Get color based on confidence score."""
	if confidence >= 0.9:
		return "🟢"  # Green for high confidence
	elif confidence >= 0.7:
		return "🟡"  # Yellow for medium confidence
	elif confidence >= 0.5:
		return "🟠"  # Orange for low confidence
	else:
		return "🔴"  # Red for very low confidence


def get_status_color(status: str) -> str:
	"""Get color based on action status."""
	status_colors = {
		"executed": "🟢",
		"in_progress": "🟡", 
		"pending_approval": "🟠",
		"approved": "🟢",
		"rejected": "🔴",
		"failed": "🔴"
	}
	return status_colors.get(status, "⚪")


with st.sidebar:
	st.header("Data Sources")
	
	# Data source selection
	data_source = st.selectbox(
		"Choose Data Source",
		["CSV Upload", "Generate Synthetic", "Salesforce", "BigQuery", "Snowflake", "MySQL"]
	)
	
	if data_source == "CSV Upload":
		st.subheader("Upload CSV")
		upload = st.file_uploader("Upload CSV", type=["csv"])
		
	elif data_source == "Generate Synthetic":
		st.subheader("Generate Synthetic Data")
		n_rows = st.slider("Rows", min_value=200, max_value=5000, value=500, step=100)
		seed = st.number_input("Random seed", value=42)
		gen_clicked = st.button("Generate CSV and Save")
		generated_path = None
		if gen_clicked:
			generated_path = generate_and_save_csv("data/sample.csv", num_rows=int(n_rows), random_seed=int(seed))
			st.success(f"Generated: {generated_path}")
	
	else:
		# Database connection section
		st.subheader(f"Connect to {data_source}")
		
		# Connection config based on source type
		if data_source == "Salesforce":
			username = st.text_input("Username", value="demo@example.com")
			password = st.text_input("Password", type="password", value="demo123")
			org_id = st.text_input("Organization ID", value="00D123456789")
			config = {"username": username, "password": password, "org_id": org_id}
			
		elif data_source == "BigQuery":
			project_id = st.text_input("Project ID", value="my-project-123")
			dataset = st.text_input("Dataset", value="analytics")
			service_account = st.text_area("Service Account JSON", value='{"type": "service_account"}')
			config = {"project_id": project_id, "dataset": dataset, "service_account": service_account}
			
		elif data_source == "Snowflake":
			account = st.text_input("Account", value="xy12345.us-east-1")
			username = st.text_input("Username", value="demo_user")
			password = st.text_input("Password", type="password", value="demo123")
			warehouse = st.text_input("Warehouse", value="COMPUTE_WH")
			config = {"account": account, "username": username, "password": password, "warehouse": warehouse}
			
		elif data_source == "MySQL":
			host = st.text_input("Host", value="localhost")
			port = st.number_input("Port", value=3306)
			username = st.text_input("Username", value="root")
			password = st.text_input("Password", type="password", value="password")
			database = st.text_input("Database", value="analytics")
			config = {"host": host, "port": port, "username": username, "password": password, "database": database}
		
		# Test connection
		if st.button("Test Connection"):
			connector = get_connector(data_source.lower(), config)
			if connector and connector.connect():
				st.success("✅ Connection successful!")
				st.session_state.connector = connector
				st.session_state.schema = connector.get_schema()
			else:
				st.error("❌ Connection failed!")
		
		# Show schema if connected
		if hasattr(st.session_state, 'schema') and st.session_state.schema:
			st.subheader("Available Objects")
			schema = st.session_state.schema
			
			if data_source == "Salesforce":
				for obj_name, obj_info in schema.get("objects", {}).items():
					with st.expander(obj_name):
						st.write("Fields:", ", ".join(obj_info["fields"]))
						st.code(obj_info["sample_query"], language="sql")
						if st.button(f"Load {obj_name}", key=f"load_{obj_name}"):
							df = st.session_state.connector.query(obj_info["sample_query"])
							st.session_state.current_df = df
							st.success(f"Loaded {len(df)} rows from {obj_name}")
			
			elif data_source in ["BigQuery", "Snowflake"]:
				if data_source == "BigQuery":
					datasets = schema.get("datasets", {})
				else:
					datasets = schema.get("databases", {})
				
				for dataset_name, dataset_info in datasets.items():
					with st.expander(dataset_name):
						if data_source == "BigQuery":
							tables = dataset_info.get("tables", {})
						else:
							tables = dataset_info.get("schemas", {}).get("PUBLIC", {}).get("tables", {})
						
						for table_name, table_info in tables.items():
							st.write(f"**{table_name}:**")
							st.write("Fields:", ", ".join(table_info["fields"]))
							st.code(table_info["sample_query"], language="sql")
							if st.button(f"Load {table_name}", key=f"load_{table_name}"):
								df = st.session_state.connector.query(table_info["sample_query"])
								st.session_state.current_df = df
								st.success(f"Loaded {len(df)} rows from {table_name}")
			
			elif data_source == "MySQL":
				for table_name, table_info in schema.get("tables", {}).items():
					with st.expander(table_name):
						st.write("Fields:", ", ".join(table_info["fields"]))
						st.code(table_info["sample_query"], language="sql")
						if st.button(f"Load {table_name}", key=f"load_{table_name}"):
							df = st.session_state.connector.query(table_info["sample_query"])
							st.session_state.current_df = df
							st.success(f"Loaded {len(df)} rows from {table_name}")
		
		# Custom query section
		if hasattr(st.session_state, 'connector'):
			st.subheader("Custom Query")
			custom_query = st.text_area("Enter your query:", height=100)
			if st.button("Execute Query"):
				if custom_query.strip():
					df = st.session_state.connector.query(custom_query)
					st.session_state.current_df = df
					st.success(f"Query executed! Loaded {len(df)} rows")

	st.header("Agent")
	run_clicked = st.button("Run Agent")


# Load data based on source
df: Optional[pd.DataFrame] = None

if data_source == "CSV Upload" and 'upload' in locals() and upload is not None:
	try:
		df = load_csv(upload.getvalue())
		st.success(f"Loaded uploaded CSV with shape {df.shape}")
	except Exception as e:
		st.error(f"Failed to parse CSV: {e}")

elif data_source == "Generate Synthetic" and 'generated_path' in locals() and generated_path and Path(generated_path).exists():
	df = pd.read_csv(generated_path)
	st.success(f"Loaded generated CSV with shape {df.shape}")

elif hasattr(st.session_state, 'current_df'):
	df = st.session_state.current_df
	st.success(f"Loaded data from {data_source} with shape {df.shape}")

else:
	st.info("Upload a CSV, generate synthetic data, or connect to a data source from the sidebar.")


if df is not None:
	st.subheader("Preview")
	st.dataframe(df.head(20), use_container_width=True)

	result = None
	if run_clicked:
		with st.spinner("Running agent... this may take a few seconds"):
			result = run_agent_on_dataframe(df)
			st.success(f"Agent completed. Insights: {result['num_insights']}")

	# Dashboard sections as horizontal tabs
	st.markdown("---")
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Patterns (24h)", len(result.get('patterns', [])) if result else 0)
	with col2:
		st.metric("Insights (24h)", result['num_insights'] if result else 0)
	with col3:
		st.metric("Decisions (24h)", (len(result['rows']) if result else 0))
	with col4:
		st.metric("Actions (24h)", (len(result['rows']) if result else 0))

	tabs = st.tabs(["Insights", "Patterns", "Decisions", "Actions", "Run Log"])

	with tabs[0]:
		if result and result['rows']:
			table_rows = []
			for r in result['rows']:
				ins = r['insight']
				dc = r.get('decision', {})
				ac = r.get('action', {})
				
				# Get confidence colors
				insight_confidence = ins.get('confidence', 0.8)
				decision_confidence = dc.get('confidence', 0.8)
				action_confidence = ac.get('confidence', 0.8)
				
				table_rows.append({
					"Metric": ins.get('metric'),
					"Kind": ins.get('kind'),
					"Dim": ins.get('dimension'),
					"Seg": ins.get('segment'),
					"Description": ins.get('description'),
					"Insight Conf": f"{get_confidence_color(insight_confidence)} {insight_confidence:.1%}",
					"Decision": f"{dc.get('action_type')} {get_confidence_color(decision_confidence)} {decision_confidence:.1%}",
					"Priority": dc.get('priority', 'normal'),
					"Action": f"{ac.get('action_type')} {get_confidence_color(action_confidence)} {action_confidence:.1%}",
					"Status": f"{get_status_color(ac.get('status', 'pending_approval'))} {ac.get('status', 'pending_approval')}",
				})
			st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
		else:
			st.info("No insights generated yet. Run the agent to see results.")

	with tabs[1]:
		if result and result.get('patterns'):
			pattern_rows = []
			for pattern in result['patterns']:
				confidence = pattern.get('confidence', 0.8)
				pattern_rows.append({
					"Type": pattern.get('type'),
					"Description": pattern.get('description'),
					"Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}",
					"Details": str(pattern.get('details', {}))
				})
			st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True, hide_index=True)
		else:
			st.info("No patterns detected yet. Run the agent to see results.")
			
	with tabs[2]:
		if result and result['rows']:
			decision_rows = []
			for r in result['rows']:
				dc = r.get('decision', {})
				confidence = dc.get('confidence', 0.8)
				decision_rows.append({
					"Action Type": dc.get('action_type'),
					"Priority": dc.get('priority'),
					"Reasoning": dc.get('reasoning'),
					"Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}"
				})
			st.dataframe(pd.DataFrame(decision_rows), use_container_width=True, hide_index=True)
		else:
			st.info("No decisions made yet. Run the agent to see results.")
			
	with tabs[3]:
		if result and result['rows']:
			st.subheader("Action Approval Dashboard")
			
			# Filter actions by status
			pending_actions = [r for r in result['rows'] if r.get('action', {}).get('status') == 'pending_approval']
			other_actions = [r for r in result['rows'] if r.get('action', {}).get('status') != 'pending_approval']
			
			# Show pending actions with approval buttons
			if pending_actions:
				st.write("**Actions Pending Approval:**")
				for i, r in enumerate(pending_actions):
					ac = r.get('action', {})
					ins = r.get('insight', {})
					
					col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
					with col1:
						st.write(f"**{ac.get('action_type', 'Unknown')}** - {ins.get('description', 'No description')}")
						st.write(f"Confidence: {get_confidence_color(ac.get('confidence', 0.8))} {ac.get('confidence', 0.8):.1%}")
					with col2:
						if st.button(f"✅ Approve", key=f"approve_{i}"):
							# Update the action status
							ac['status'] = 'approved'
							st.success("Action approved!")
							st.rerun()
					with col3:
						if st.button(f"❌ Reject", key=f"reject_{i}"):
							# Update the action status
							ac['status'] = 'rejected'
							st.error("Action rejected!")
							st.rerun()
					with col4:
						st.write(f"Status: {get_status_color(ac.get('status', 'pending_approval'))} {ac.get('status', 'pending_approval')}")
					st.divider()
			
			# Show all actions in a table
			st.write("**All Actions:**")
			action_rows = []
			for r in result['rows']:
				ac = r.get('action', {})
				confidence = ac.get('confidence', 0.8)
				status = ac.get('status', 'pending_approval')
				action_rows.append({
					"Action Type": ac.get('action_type'),
					"Status": f"{get_status_color(status)} {status}",
					"Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}",
					"Details": str(ac.get('details', {}))
				})
			st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
		else:
			st.info("No actions taken yet. Run the agent to see results.")
			
	with tabs[4]:
		if result:
			for e in result['log']:
				agent = e.get('agent', 'Agent')
				st.write(f"- [{agent}] {e['event']}: {e.get('details', {})}")
		else:
			st.info("Run the agent to see logs.")


