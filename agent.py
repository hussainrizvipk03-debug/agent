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

class BreakdownResult(BaseModel):
    intent: str = Field(description="The primary goal (e.g., 'analysis', 'cleaning', 'specific_question')")
    parameters: List[str] = Field(description="Column names or features identified")
    is_ambiguous: bool = Field(description="True if query is unclear or columns are ambiguous")
    missing_info: str = Field(description="What information is missing to proceed?")

# --- TOOLS ---

@tool
def breakdown_complex_prompt(query: str, columns: List[str]) -> str:
    """Analyzes and breaks down a complex data analysis request into manageable steps."""
    # This tool is used by the model to structure its thought process
    return f"Prompt broken down. Columns identified: {columns}"

@tool
def throw_ui_clarification(question: str) -> str:
    """Throws a clarification widget in the UI to ask the user for more information."""
    return f"QUESTION TO USER: {question}"

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    df: pd.DataFrame
    figures: Annotated[Dict[str, Any], operator.ior]
    breakdown_data: Dict[str, Any]
    next_node: str

# --- GRAPH NODES ---

def node_prompt_breakdown(state: AgentState):
    """Image Node: 'prompt breakdown'"""
    llm = get_llm().with_structured_output(BreakdownResult)
    query = state['messages'][-1].content
    columns = state['df'].columns.tolist()
    
    prompt = f"""
    Break down this user query: '{query}'
    Available columns: {columns}
    If there is any ambiguity (e.g. multiple matching columns), mark is_ambiguous as True and provide a question.
    """
    
    analysis = llm.invoke(prompt)
    
    # Decide next step
    if analysis.is_ambiguous:
        next_step = "clarification"
    else:
        next_step = "router"
        
    return {
        "breakdown_data": analysis.dict(),
        "next_node": next_step
    }

def node_clarification_questions(state: AgentState):
    """Image Node: 'clarification questions'"""
    # This node uses the breakdown data to prepare a UI question
    question = state['breakdown_data']['missing_info']
    return {
        "messages": [AIMessage(content=f"QUESTION TO USER: {question}")]
    }

def node_ai_router(state: AgentState):
    """Image Node: 'AI Router'"""
    # Simply routes to EDA for now, but structures for future RAG/VectorDB routing
    return {"next_node": "eda"}

def node_eda_engine(state: AgentState):
    """Autonomous EDA Engine using specialized tools."""
    llm = get_llm().bind_tools([
        tools.get_data_info, 
        tools.get_data_stats, 
        tools.check_missing_values, 
        tools.drop_missing_values, 
        tools.fill_missing_values, 
        tools.get_correlation_matrix,
        tools.plot_histogram, 
        tools.plot_correlation_heatmap, 
        tools.plot_scatter,
        tools.get_top_features
    ])
    
    messages = [SystemMessage(content=f"You are executing an EDA plan. Context: {state['breakdown_data']}")] + list(state['messages'])
    response = llm.invoke(messages)
    return {"messages": [response]}

def node_execute_tools(state: AgentState):
    """Executed when tools need to be called."""
    last_message = state['messages'][-1]
    tool_messages = []
    new_figures = {}
    
    # Safety check: ensure last message is an AIMessage with tool calls
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args'].copy()
        
        # Inject the dataframe if the tool is from our tools module
        if hasattr(tools, tool_name):
            args['df'] = state['df']
        
        try:
            # Safely get and invoke the tool
            tool_func = getattr(tools, tool_name)
            result = tool_func.invoke(args)
            
            # Handle visualization outputs
            if isinstance(result, plt.Figure):
                fig_id = f"fig_{tool_call['id']}"
                new_figures[fig_id] = result
                content = f"Plot generated: {fig_id}"
            elif isinstance(result, pd.DataFrame):
                content = result.to_string()
            else:
                content = str(result)
                
            tool_messages.append(ToolMessage(
                tool_call_id=tool_call['id'], 
                content=content
            ))
        except Exception as e:
            # Provide descriptive error messages for debugging
            error_content = f"Error executing {tool_name}: {str(e)}"
            tool_messages.append(ToolMessage(
                tool_call_id=tool_call['id'], 
                content=error_content
            ))
            
    return {"messages": tool_messages, "figures": new_figures}

# --- ROUTING LOGIC ---

def route_breakdown(state: AgentState):
    return state['next_node'] # 'clarification' or 'router'

def should_continue_eda(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

# Following the image nodes
workflow.add_node("prompt_breakdown", node_prompt_breakdown)
workflow.add_node("clarification_questions", node_clarification_questions)
workflow.add_node("ai_router", node_ai_router)
workflow.add_node("eda_engine", node_eda_engine)
workflow.add_node("tools", node_execute_tools)

workflow.add_edge(START, "prompt_breakdown")

workflow.add_conditional_edges("prompt_breakdown", route_breakdown, {
    "clarification": "clarification_questions",
    "router": "ai_router"
})

workflow.add_edge("clarification_questions", END) # HITL stop

workflow.add_edge("ai_router", "eda_engine")

workflow.add_conditional_edges("eda_engine", should_continue_eda, {
    "tools": "tools",
    END: END
})

workflow.add_edge("tools", "eda_engine")

# --- COMPILE ---

agent_graph = workflow.compile()

class DataAgent:
    def __init__(self):
        self.graph = agent_graph

    def run(self, user_query: str, df: pd.DataFrame, messages_history: list = None):
        messages = messages_history if messages_history else []
        # Ensure latest query is at the end
        if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != user_query:
            messages.append(HumanMessage(content=user_query))
            
        initial_state = {
            "messages": messages,
            "df": df,
            "figures": {},
            "breakdown_data": {},
            "next_node": ""
        }
        
        return self.graph.invoke(initial_state)

agent = DataAgent()
