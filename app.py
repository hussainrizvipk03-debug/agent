import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns
from agent import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Page configuration
st.set_page_config(page_title="AI Data Scientist", layout="wide", page_icon="🤖")

st.title("🤖 Proactive Autonomous Data Scientist")
st.markdown("""
**Autonomous Pipeline:**
1. **Whole EDA Diagnostics**: Automatic scan of Info, Stats (for all columns), Missing Values, and Correlation.
2. **Auto-Cleaning**: The agent fills or drops nulls autonomously.
3. **Smart Visualization**: The agent self-identifies the best columns for Uni/Bi/Multi-variant plots based on insights.
""")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_messages" not in st.session_state:
    st.session_state.raw_messages = []
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat / Reset Pipeline"):
        st.session_state.chat_history = []
        st.session_state.raw_messages = []
        st.session_state.last_dataset = None
        st.session_state.df = None
        if "mcqs" in st.session_state:
            del st.session_state["mcqs"]
        st.rerun()
    
    st.divider()
    st.header("📂 Select Dataset")
    
    # Dynamic Seaborn Dataset Selection
    seaborn_datasets = sns.get_dataset_names()
    selected_sns_dataset = st.selectbox("Choose a Seaborn Dataset", options=["None"] + seaborn_datasets)
    
    if selected_sns_dataset != "None":
        if st.button(f"Load '{selected_sns_dataset}' Dataset"):
            st.session_state.chat_history = []
            st.session_state.raw_messages = []
            st.session_state.last_dataset = f"sns_{selected_sns_dataset}"
            st.session_state.df = None # Reset df to trigger reload
            st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.header("📥 Export Data")
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(st.session_state.df)
        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Helper to process graph messages into streamlit-friendly chat history
def process_messages_to_history(messages, figures):
    processed = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            processed.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            content = msg.content if msg.content else ""
            p_msg = {"role": "assistant", "content": content}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                p_msg["tool_calls"] = ", ".join([tc['name'] for tc in msg.tool_calls])
            if p_msg["content"] or (hasattr(msg, 'tool_calls') and msg.tool_calls):
                processed.append(p_msg)
        elif isinstance(msg, ToolMessage):
            p_msg = {"role": "assistant", "is_tool": True, "content": msg.content}
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

# Data Loading Logic
if st.session_state.last_dataset and st.session_state.last_dataset.startswith("sns_") and st.session_state.df is None:
    dataset_name = st.session_state.last_dataset.replace("sns_", "")
    st.session_state.df = sns.load_dataset(dataset_name)
    st.success(f"✅ Seaborn '{dataset_name}' dataset loaded successfully!")
elif not st.session_state.last_dataset or not st.session_state.last_dataset.startswith("sns_"):
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        if st.session_state.last_dataset != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.last_dataset = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.raw_messages = []
            st.rerun()

# Core Execution Pipeline
if st.session_state.df is not None:
    if not st.session_state.chat_history:
        with st.status("🚀 Launching Autonomous Whole EDA...", expanded=True) as status:
            try:
                final_state = agent.run("", st.session_state.df.copy(), messages_history=[])
                st.session_state.df = final_state['df']
                st.session_state.raw_messages = final_state['messages']
                st.session_state.chat_history = process_messages_to_history(final_state['messages'], final_state['figures'])
                status.update(label="✅ Autonomous EDA & Visualization Complete", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

    # Display History
    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("is_tool"):
                with st.expander("📥 See Data Result"):
                    st.text(msg["content"])
            else:
                st.markdown(msg["content"])
                if "tool_calls" in msg:
                    st.caption(f"🛠️ Tool execution: {msg['tool_calls']}")
            if "viz" in msg:
                st.image(msg["viz"])

    if user_query := st.chat_input("Ask for custom plots or deeper insights..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.status("🧠 Analyzing request...", expanded=True) as status:
            try:
                final_state = agent.run(user_query, st.session_state.df.copy(), messages_history=st.session_state.raw_messages)
                st.session_state.df = final_state['df']
                st.session_state.raw_messages = final_state['messages']
                st.session_state.chat_history = process_messages_to_history(final_state['messages'], final_state['figures'])
                status.update(label="✅ Completed", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.info("👋 Use the sidebar to load any Seaborn dataset or upload your own CSV to begin.")
