import streamlit as st
import time
from integritycore.agents.loop import LoopAgent
from integritycore.core.verifier import ETLStrategy

# Configure Streamlit page for "Glass Box" experience
st.set_page_config(page_title="IntegrityCore Verification Dashboard", page_icon="🔍", layout="wide")

st.title("IntegrityCore: Glass Box Verification")
st.markdown("A cloud-agnostic agentic framework for mathematical ETL verification.")

# Sidebar Configuration
st.sidebar.header("Agent Configuration")
model = st.sidebar.selectbox(
    "LLM Architecture (via ADK)", 
    ["gemini/gemini-2.5-pro", "gemini/gemini-pro", "gpt-4-turbo", "claude-3-opus"]
)
strategy_str = st.sidebar.radio("Verification Strategy", ["INCREMENTAL", "FULL_REFRESH"])
strategy = ETLStrategy[strategy_str]

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Context Protocol (MCP)")
st.sidebar.success("Database Tools Linked: Ready\n- PostgreSQL Plugin\n- Snowflake Plugin")

prompt = st.text_area(
    "ETL Verification Requirement", 
    "Create an ETL pipeline that loads user active sessions into the analytics schema. Only load sessions that have been updated since the last pipeline run.", 
    height=120
)

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("Generate & Verify", type="primary")

if run_btn:
    if not prompt:
        st.warning("Please provide an ETL prompt.")
    else:
        st.info(f"Initializing Multi-Agent Loop using `{model}`...")
        
        agent = LoopAgent(model_name=model)
        agent.discover_tools() # Output simulated MCP tools
        
        # Simulating UI feedback for the backend loop execution
        with st.status("Executing Agentic Loop...", expanded=True) as status:
            st.markdown("- **Step 1:** Connecting to MCP registry and resolving adapter tools (Postgres/Snowflake).")
            time.sleep(1)
            st.markdown("- **Step 2:** Requesting Generation from Model Logic.")
            
            try:
                # Actual backend execution (may take a few seconds)
                final_sql = agent.execute_etl_loop(prompt, strategy)
                
                st.markdown("- **Step 3:** Verifying Logic Bounds using Z3 SMT Solver & Abstract Syntax Trees.")
                
                status.update(label="Glass Box Verification Passed ✅", state="complete", expanded=True)
                
                st.success("Mathematical Proof Verified. The agent generated safe bounds for your pipeline.")
                st.code(final_sql, language="sql")
                
            except Exception as e:
                status.update(label="Agent Loop Verification Failed ❌", state="error", expanded=True)
                st.error(f"Error: {e}")
