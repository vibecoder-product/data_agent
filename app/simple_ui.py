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
    from agent.simple_brain import run_agent_on_dataframe
    from core.mailer import read_outbox
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
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.6:
        return "🟡"
    else:
        return "🔴"

def get_status_color(status: str) -> str:
    """Get color based on status."""
    status_colors = {
        "executed": "✅",
        "in_progress": "🔄",
        "pending_approval": "⏳",
        "approved": "✅",
        "rejected": "❌",
        "failed": "🔴"
    }
    return status_colors.get(status, "⚪")

# Sidebar for data input
with st.sidebar:
    st.header("Data Input")
    
    data_source = st.selectbox(
        "Choose Data Source",
        ["CSV Upload", "Generate Synthetic"]
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

else:
    st.info("Upload a CSV or generate synthetic data from the sidebar.")

# Display data preview
if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Run agent
    if run_clicked:
        with st.spinner("Running autonomous analysis..."):
            try:
                result = run_agent_on_dataframe(df)
                st.session_state.agent_result = result
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error running agent: {e}")
                st.session_state.agent_result = None

# Display results
if hasattr(st.session_state, 'agent_result') and st.session_state.agent_result:
    result = st.session_state.agent_result
    
    # Create tabs for different views
    tabs = st.tabs(["Insights", "Patterns", "Decisions", "Actions"])
    
    with tabs[0]:
        if result and result['rows']:
            insight_rows = []
            for r in result['rows']:
                ins = r.get('insight', {})
                confidence = ins.get('confidence', 0.8)
                insight_rows.append({
                    "Metric": ins.get('metric'),
                    "Description": ins.get('description'),
                    "Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}"
                })
            st.dataframe(pd.DataFrame(insight_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No insights detected yet. Run the agent to see results.")
    
    with tabs[1]:
        if result and result.get('patterns'):
            pattern_rows = []
            for p in result['patterns']:
                confidence = p.get('confidence', 0.8)
                pattern_rows.append({
                    "Pattern": p.get('pattern'),
                    "Metric": p.get('metric'),
                    "Description": p.get('description'),
                    "Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}"
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
                            ac['status'] = 'approved'
                            st.success("Action approved!")
                            st.rerun()
                    with col3:
                        if st.button(f"❌ Reject", key=f"reject_{i}"):
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
                action_rows.append({
                    "Action Type": ac.get('action_type'),
                    "Status": f"{get_status_color(ac.get('status', 'pending_approval'))} {ac.get('status', 'pending_approval')}",
                    "Confidence": f"{get_confidence_color(confidence)} {confidence:.1%}",
                    "Details": str(ac.get('details', {}))
                })
            st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No actions taken yet. Run the agent to see results.")

# Show run logs
if hasattr(st.session_state, 'agent_result') and st.session_state.agent_result:
    st.subheader("Agent Run Logs")
    if result and result.get('logs'):
        for log in result['logs']:
            st.text(log)
    else:
        st.info("No logs available.")
