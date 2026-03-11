import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from agent import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Page configuration
st.set_page_config(page_title="AI Data Scientist", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Scientist (Structured Workflow)")
st.markdown("""
**Workflow Implementation:**
1. **Prompt Breakdown** ➡️ Analyze query intent & ambiguity.
2. **Clarification** ➡️ ask user if columns/intent are unclear.
3. **AI Router** ➡️ Route to EDA/Analysis engine.
4. **Execution** ➡️ Run tools & generate visualizations.
""")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_messages" not in st.session_state:
    st.session_state.raw_messages = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat / Reset"):
        st.session_state.chat_history = []
        st.session_state.raw_messages = []
        st.session_state.last_uploaded = None
        st.rerun()
    st.markdown("---")
    st.write("### Agent Workflow Status")
    if st.session_state.last_uploaded:
        st.success(f"✅ Active: {st.session_state.last_uploaded}")
    else:
        st.info("🕒 Waiting for CSV...")

# Helper to process graph messages into streamlit-friendly chat history
def process_messages_to_history(messages, figures):
    processed = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            processed.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            content = msg.content if msg.content else ""
            p_msg = {"role": "assistant", "content": content}
            
            # Identify if this is a clarification request (HITL)
            if "QUESTION TO USER:" in content:
                p_msg["is_clarification"] = True
                p_msg["content"] = content.replace("QUESTION TO USER:", "").strip()
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_list = ", ".join([tc['name'] for tc in msg.tool_calls])
                p_msg["tool_calls"] = tool_list
            
            if p_msg["content"] or (hasattr(msg, 'tool_calls') and msg.tool_calls):
                processed.append(p_msg)
                
        elif isinstance(msg, ToolMessage):
            # Normal tool output
            p_msg = {"role": "assistant", "is_tool": True, "content": msg.content}
            # Check for visualizations
            fig_id = f"fig_{msg.tool_call_id}"
            if fig_id in figures:
                fig = figures[fig_id]
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                p_msg["viz"] = buf.getvalue()
                plt.close(fig)
            processed.append(p_msg)
    
    return processed

# 1. File Upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    
    # Auto-run for new files
    if st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state.raw_messages = []
        
        with st.status("🚀 Running Initial Multi-Stage Analysis...", expanded=True) as status:
            try:
                query = "Analyze this dataset and provide a comprehensive summary of key insights, including statistics and correlations."
                final_state = agent.run(query, df, messages_history=[])
                
                st.session_state.raw_messages = final_state['messages']
                st.session_state.chat_history = process_messages_to_history(final_state['messages'], final_state['figures'])
                
                status.update(label="✅ Run Complete", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display History Dashboard
    st.divider()
    st.subheader("📋 Pipeline Output")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("is_clarification"):
                st.warning(f"🤔 **Clarification Needed:** {msg['content']}")
            elif msg.get("is_tool"):
                with st.expander(f"📥 Tool: {msg.get('tool_name', 'Output')}"):
                    st.text(msg["content"])
            else:
                st.markdown(msg["content"])
                if "tool_calls" in msg:
                    st.caption(f"🛠️ Executed: {msg['tool_calls']}")
            
            if "viz" in msg:
                st.image(msg["viz"])

    # HITL / Follow-up Input
    if user_query := st.chat_input("Ask a question or reply to a clarification..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.status("🧠 Processing...", expanded=True) as status:
            try:
                # Continue from existing raw_messages history
                final_state = agent.run(user_query, df, messages_history=st.session_state.raw_messages)
                st.session_state.raw_messages = final_state['messages']
                st.session_state.chat_history = process_messages_to_history(final_state['messages'], final_state['figures'])
                
                status.update(label="✅ Done", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👋 Upload a CSV file and I will automatically break down the prompt and start the analysis!")
