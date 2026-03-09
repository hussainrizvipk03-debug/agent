import streamlit as st
import pandas as pd
import tools
import matplotlib.pyplot as plt
import io
import re
from llm import chain

# Page configuration
st.set_page_config(page_title="AI Data Scientist", layout="wide", page_icon="📊")

st.title("🤖 AI Data Scientist")
st.markdown("Upload your dataset and chat with me to analyze your data step-by-step.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for instructions and clear history
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.write("1. Upload a CSV file.")
    st.write("2. Ask questions about your data.")
    st.write("3. View visual outputs directly in chat.")

# 1. File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Cache the dataframe to avoid re-reading on every rerun
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    with st.expander("📊 Preview Data"):
        st.dataframe(df.head())

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "visualization_bytes" in message:
                st.image(message["visualization_bytes"])
            if "result" in message:
                if isinstance(message["result"], pd.DataFrame):
                    st.dataframe(message["result"])
                else:
                    st.text(message["result"])

    # Chat Input
    if user_query := st.chat_input("Ask about your data (e.g., 'Plot histogram of Age')..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Agent logic and visualization
        with st.chat_message("assistant"):
            with st.status("🛠️ AI Analysis Flow", expanded=True) as status:
                st.write("🔍 Analyzing request...")
                try:
                    full_response = chain.invoke({"user_query": user_query})
                    
                    # More robust regex for function extraction
                    match = re.search(r'(\w+)\s*\((.*)\)', full_response)
                    
                    if match:
                        tool_name = match.group(1).strip()
                        args_str = match.group(2).strip()
                        
                        st.write(f"⚙️ Selected Tool: `{tool_name}`")
                        
                        # Parse arguments
                        kwargs = {}
                        if args_str:
                            # Handle simple key=value or key='value'
                            parts = re.findall(r'(\w+)\s*=\s*(?:["\']([^"\']*)["\']|(\w+))', args_str)
                            for k, v1, v2 in parts:
                                kwargs[k] = v1 if v1 else v2
                        
                        st.write(f"🚀 Executing with args: `{kwargs}`")
                        
                        if hasattr(tools, tool_name):
                            method = getattr(tools, tool_name)
                            # Actual tool execution
                            result = method(df, **kwargs)
                            
                            st.write("✅ Data processed. Rendering output...")
                            status.update(label="Analysis Complete", state="complete", expanded=False)
                            
                            new_msg = {"role": "assistant", "content": f"Here is the analysis result for `{tool_name}`:"}
                            
                            # Determine result type and display
                            if hasattr(result, "savefig"):
                                # It's a matplotlib figure
                                st.pyplot(result)
                                
                                # Export to bytes for history persistence
                                buf = io.BytesIO()
                                result.savefig(buf, format="png", bbox_inches='tight')
                                buf.seek(0)
                                new_msg["visualization_bytes"] = buf.getvalue()
                                plt.close(result) 
                            elif isinstance(result, pd.DataFrame):
                                st.dataframe(result)
                                new_msg["result"] = result
                            elif isinstance(result, (pd.Series, dict)):
                                st.write(result)
                                new_msg["result"] = str(result)
                            else:
                                st.markdown(result)
                                new_msg["result"] = str(result)
                            
                            st.session_state.messages.append(new_msg)
                        else:
                            status.update(label="⚠️ Tool Error", state="error", expanded=True)
                            st.error(f"Function `{tool_name}` not found in tools.py")
                    else:
                        # Direct text answer
                        status.update(label="💡 Response", state="complete", expanded=False)
                        st.write(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                except Exception as e:
                    status.update(label="❌ Failure", state="error", expanded=True)
                    st.error(f"Execution Error: {str(e)}")
                    st.info("Check if column names are correct or if the tool supports this query.")
else:
    st.info("👋 Welcome! Please upload a CSV file to begin analysis.")
