import streamlit as st
import pandas as pd
import tools
import matplotlib.pyplot as plt
import io
from agent import agent

# Page configuration
st.set_page_config(page_title="AI Data Scientist", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Scientist")
st.markdown("Simply upload your dataset, and the agent will automatically perform a full analysis for you.")

# Initialize session state for chat history and auto-run status
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat / Reset"):
        st.session_state.messages = []
        st.session_state.last_uploaded = None
        st.rerun()
    st.markdown("---")
    st.write("### Agent Status")
    if st.session_state.last_uploaded:
        st.success("✅ Analysis Complete")
    else:
        st.info("🕒 Waiting for data...")

# 1. File Upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    
    # Auto-run Agent Analysis if it's a new file
    if st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        
        with st.chat_message("assistant"):
            with st.status("🚀 Running Full Autonomous Dashboard...", expanded=True) as status:
                try:
                    # Requesting full analysis
                    raw_response = agent.run("Perform a Full Comprehensive Analysis. Run every single tool from your mapping (Info, Stats, Missing, Correlation, Heatmap, Top Features, and Visualizations) based on the dataset.", df)
                    thought, actions = agent.parse_action(raw_response)
                    
                    if thought:
                        st.info(f"**STRATEGY:** {thought}")
                    
                    # Execute each step in the plan
                    for i, (tool_name, args_str) in enumerate(actions):
                        st.write(f"📊 Running {tool_name}...")
                        result = agent.execute_tool(tool_name, args_str, df)
                        
                        new_msg = {"role": "assistant", "content": f"**Full Analysis Step {i+1}:** Output from `{tool_name}`"}
                        
                        if hasattr(result, "savefig"):
                            st.pyplot(result)
                            buf = io.BytesIO()
                            result.savefig(buf, format="png", bbox_inches='tight')
                            buf.seek(0)
                            new_msg["visualization_bytes"] = buf.getvalue()
                            plt.close(result) 
                        elif isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                            new_msg["result"] = result
                        else:
                            st.markdown(f"**Result:** {result}")
                            new_msg["result"] = str(result)
                        
                        st.session_state.messages.append(new_msg)
                    
                    status.update(label="✅ All Tools Executed Successfully", state="complete", expanded=False)
                    st.rerun() 
                except Exception as e:
                    status.update(label="❌ Failure", state="error", expanded=True)
                    st.error(f"Auto-Analysis Error: {str(e)}")

    # Display Analysis Dashboard (History)
    st.divider()
    st.subheader("📋 Analysis Dashboard")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "visualization_bytes" in message:
                st.image(message["visualization_bytes"])
            if "result" in message:
                if isinstance(message["result"], pd.DataFrame):
                    st.dataframe(message["result"])
                else:
                    st.text(str(message["result"]))

    # Manual chat if user wants more
    if user_query := st.chat_input("I have more questions..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.status("🧠 Processing...", expanded=True) as status:
                try:
                    response = agent.run(user_query, df)
                    thought, actions = agent.parse_action(response)
                    for i, (tool_name, args_str) in enumerate(actions):
                        result = agent.execute_tool(tool_name, args_str, df)
                        new_msg = {"role": "assistant", "content": f"**Action:** `{tool_name}`"}
                        if hasattr(result, "savefig"):
                            st.pyplot(result)
                            buf = io.BytesIO()
                            result.savefig(buf, format="png", bbox_inches='tight')
                            buf.seek(0)
                            new_msg["visualization_bytes"] = buf.getvalue()
                            plt.close(result) 
                        elif isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                            new_msg["result"] = result
                        else:
                            st.markdown(result)
                            new_msg["result"] = str(result)
                        st.session_state.messages.append(new_msg)
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("👋 Upload a CSV file and I will automatically handle the rest!")
