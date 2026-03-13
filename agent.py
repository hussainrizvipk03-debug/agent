import operator
from typing import Annotated, Sequence, TypedDict, Union, Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from llm import get_llm
import tools
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- SCHEMA DEFINITIONS ---

class AutonomousPlan(TypedDict):
    is_ready: bool
    plan_description: str
    selected_columns: List[str]
    clarification_question: str

class PlanningSchema(BaseModel):
    is_ready: bool = Field(description="True if diagnostics are done and we can visualize.")
    plan_description: str = Field(description="Describe the plan for Uni, Bi, and MULTIVARIATE plots.")
    selected_columns: List[str] = Field(description="EXACT column names to be used.")
    clarification_question: str = Field(description="Question if stuck.")

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    df: pd.DataFrame
    figures: Annotated[Dict[str, Any], operator.ior]
    plan: AutonomousPlan
    phase: str 
    is_eda_finished: bool

# --- NODES ---

def node_eda_diagnostics(state: AgentState):
    """
    Phase 1: Forceful WHOLE EDA Diagnostics with Multi-turn Awareness.
    """
    llm = get_llm().bind_tools([
        tools.get_data_info,
        tools.get_data_stats,
        tools.check_missing_values,
        tools.drop_missing_values,
        tools.fill_missing_values,
        tools.label_encode_categorical_columns,
        tools.get_correlation_matrix
    ])
    
    # Identify completed tools to guide the LLM
    completed_tools = []
    for m in state['messages']:
        if hasattr(m, 'tool_calls') and m.tool_calls:
            for tc in m.tool_calls:
                completed_tools.append(tc['name'])

    prompt = f"""
    You are in the MANDATORY WHOLE EDA phase. You must perform ALL of these tools as efficiently as possible:
    1. get_data_info() - Report dimensions.
    2. get_data_stats() - Report full statistics for all columns.
    3. check_missing_values() - Detect nulls.
    4. IF NULLS ARE DETECTED: You MUST call drop_missing_values() or fill_missing_values() to fix them.
    5. CRITICAL: If you just fixed missing values, you MUST call check_missing_values() AGAIN to verify the count is now zero.
    6. IF CATEGORICAL COLUMNS (object/string) EXIST: You MUST call label_encode_categorical_columns() to prepare them for ML.
    7. get_correlation_matrix() - Finalize numerical relationships.

    Completed Tools So Far: {list(set(completed_tools))}
    
    IMPORTANT: To save time, call MULTIPLE tools in a SINGLE turn if they haven't been completed yet.
    Do not wait for one tool to finish before calling the next one if you know both are needed.
    
    YOUR GOAL: If any step is missing, if nulls are unfixed, or if categorical data is unencoded, call the necessary tools NOW.
    Only if the data is clean (verify with check_missing_values if needed), numeric-ready, and all diagnostic steps are complete, output a summary and say 'DIAGNOSTICS_COMPLETE'.
    """
    
    # SAFE TRIMMING: Ensure we don't include orphan ToolMessages.
    # We find the last 6 messages, but if the first one is a ToolMessage, we find its parent AIMessage.
    full_history = list(state['messages'])
    slice_index = max(0, len(full_history) - 6)
    
    # If the start of our slice is a ToolMessage, move back until we find an AIMessage or START
    while slice_index > 0 and isinstance(full_history[slice_index], ToolMessage):
        slice_index -= 1
        
    trimmed_history = full_history[slice_index:]
    messages = [SystemMessage(content=prompt)] + trimmed_history
    
    if not any(isinstance(m, HumanMessage) for m in state['messages']):
        messages.append(HumanMessage(content="Start Whole EDA Diagnostics."))
    
    response = llm.invoke(messages)
    
    # Check if we should mark EDA as finished
    finished = "DIAGNOSTICS_COMPLETE" in response.content
    
    return {
        "messages": [response],
        "phase": "diagnostic",
        "is_eda_finished": finished
    }

def node_chat_handler(state: AgentState):
    """
    Direct response node for user-specific queries once EDA is done.
    """
    llm = get_llm().bind_tools([
        tools.get_data_info,
        tools.get_data_stats,
        tools.check_missing_values,
        tools.drop_missing_values,
        tools.fill_missing_values,
        tools.label_encode_categorical_columns,
        tools.get_correlation_matrix,
        tools.plot_univariate_analysis,
        tools.plot_bivariate_analysis,
        tools.plot_multivariate_analysis,
        tools.plot_correlation_heatmap,
        tools.get_top_features
    ])
    
    # Just handle the current conversation thread
    messages = [SystemMessage(content="You are a helpful data assistant. Use tools if the user asks for specific analysis or plots.")] + list(state['messages'][-5:])
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "phase": "chat"
    }

def node_autonomous_refinement(state: AgentState):
    """
    Phase 2: Self-Analysis & Visualization Planning.
    Includes Univariate, Bivariate, AND Multivariate planning.
    """
    llm = get_llm().with_structured_output(PlanningSchema)
    
    cols = state['df'].columns.tolist()
    
    prompt = f"""
    WHOLE EDA is complete. The data has been cleaned.
    Available Columns: {cols}
    
    TASK:
    1. Autonomously plan a comprehensive visual suite.
    2. Include at least:
       - Univariate analysis for key features.
       - Bivariate analysis for the strongest correlations.
       - MULTIVARIATE analysis (pairplots) for the top 3-4 features.
    3. Set is_ready=True and provide the exact plan.
    """
    
    plan_obj = llm.invoke(prompt)
    
    return {
        "plan": plan_obj.model_dump(),
        "phase": "refinement"
    }

def node_visualization_engine(state: AgentState):
    """
    Phase 3: Autonomous Visualization Execution.
    """
    llm = get_llm().bind_tools([
        tools.plot_univariate_analysis,
        tools.plot_bivariate_analysis,
        tools.plot_multivariate_analysis,
        tools.plot_correlation_heatmap,
        tools.get_top_features
    ])
    
    plan_desc = state['plan']['plan_description']
    prompt = f"""
    Executing full visualization suite including Multivariate analysis. 
    Plan: {plan_desc}
    
    IMPORTANT: Call ALL planned tools in this SINGLE turn. Do not call them one by one.
    """
    
    # SAFE TRIMMING:
    full_history = list(state['messages'])
    slice_index = max(0, len(full_history) - 6)
    while slice_index > 0 and isinstance(full_history[slice_index], ToolMessage):
        slice_index -= 1
        
    trimmed_history = full_history[slice_index:]
    messages = [SystemMessage(content=prompt)] + trimmed_history
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "phase": "execution"
    }

def node_execute_tools(state: AgentState):
    """Bridge for @tool calls. Handles dataframe updates for cleaning."""
    last_message = state['messages'][-1]
    tool_messages = []
    new_figures = {}
    updated_df = state['df']
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args'].copy()
        
        if hasattr(tools, tool_name):
            args['df'] = updated_df
        
        try:
            tool_func = getattr(tools, tool_name)
            result = tool_func.invoke(args)
            
            if isinstance(result, plt.Figure):
                fig_id = f"fig_{tool_call['id']}"
                new_figures[fig_id] = result
                content = f"Visual Generated: {fig_id}"
            elif isinstance(result, pd.DataFrame):
                updated_df = result
                content = f"Cleaning complete. Dataframe updated. New shape: {updated_df.shape}"
            else:
                content = str(result)
                
            tool_messages.append(ToolMessage(tool_call_id=tool_call['id'], content=content))
        except Exception as e:
            tool_messages.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Error in {tool_name}: {str(e)}"))
            
    return {
        "messages": tool_messages, 
        "figures": new_figures,
        "df": updated_df
    }

# --- CONTROL FLOW ---

def route_initial(state: AgentState):
    # If a specific user query was provided in a follow-up, and EDA is done, go to chat
    if state.get('is_eda_finished') or len(state['messages']) > 1:
        if state.get('phase') != 'diagnostic':
            return "chat"
            
    # Default initial path for new datasets
    last_tool_output = [m.content for m in state['messages'] if isinstance(m, ToolMessage)]
    if not last_tool_output:
        return "diagnostics"
    return "refine"

def route_after_tools(state: AgentState):
    last_msg = state['messages'][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    
    if state['phase'] == 'diagnostic':
        if state.get('is_eda_finished'):
            return "refine"
        return "diagnostics"
        
    if state['phase'] == 'chat':
        return END

    return END

def route_refinement(state: AgentState):
    if state['plan'].get('is_ready'):
        return "visualize"
    return END

# --- GRAPH ---

workflow = StateGraph(AgentState)

workflow.add_node("diagnostics", node_eda_diagnostics)
workflow.add_node("refine", node_autonomous_refinement)
workflow.add_node("visualize", node_visualization_engine)
workflow.add_node("tools", node_execute_tools)
workflow.add_node("chat", node_chat_handler)

workflow.add_conditional_edges(START, route_initial, {
    "diagnostics": "diagnostics",
    "refine": "refine",
    "chat": "chat"
})

workflow.add_conditional_edges("diagnostics", route_after_tools, {
    "tools": "tools",
    "diagnostics": "diagnostics",
    "refine": "refine"
})

workflow.add_conditional_edges("tools", route_after_tools, {
    "tools": "tools",
    "diagnostics": "diagnostics",
    "refine": "refine",
    END: END
})

workflow.add_conditional_edges("refine", route_refinement, {
    "visualize": "visualize",
    END: END
})

workflow.add_conditional_edges("visualize", route_after_tools, {
    "tools": "tools",
    END: END
})

workflow.add_conditional_edges("chat", route_after_tools, {
    "tools": "tools",
    END: END
})

agent_graph = workflow.compile()

class DataAgent:
    def __init__(self):
        self.graph = agent_graph

    def run(self, user_query: str, df: pd.DataFrame, messages_history: list = None):
        messages = messages_history if messages_history else []
        if user_query and (not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != user_query):
            messages.append(HumanMessage(content=user_query))
            
        initial_state = {
            "messages": messages,
            "df": df,
            "figures": {},
            "plan": {"is_ready": False, "plan_description": "", "selected_columns": [], "clarification_question": ""},
            "phase": "",
            "is_eda_finished": False
        }
        
        return self.graph.invoke(initial_state)

agent = DataAgent()
