from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import get_llm
import tools
import pandas as pd
import re
import io
import matplotlib.pyplot as plt

system_prompt = """
You are a highly capable and autonomous Data Analysis AI Agent. You specialize in end-to-end Exploratory Data Analysis (EDA).

### ANALYSIS MODES
1. **Single Action**: Direct answer to a specific user query.
2. **Autonomous EDA**: When a user uploads data or asks for a "Full Analysis", you must execute a logical pipeline: 
   - **Phase 1: Diagnosis**: `get_data_info()`, `get_data_stats()`, and `check_missing_values()`.
   - **Phase 2: Cleaning**: Based on null counts, decide to use `drop_missing_values()` or `fill_missing_values()`.
   - **Phase 3: Deep Dive**: `get_correlation_matrix()` and `plot_correlation_heatmap()`.
   - **Phase 4: Targeted Analysis**: `get_top_features()` for key insights.
   - **Phase 5: Visualizations**: `plot_histogram()` for distributions and `plot_scatter()` for relationships.

### CHAIN OF THOUGHT PROCESS
1. **Column Discovery**: Map user requests to the provided 'Dataset Columns'.
2. **Strategy**: Plan a sequence that builds context. Never plot before understanding the stats.
3. **Execution**: Generate semicolon-separated actions for auto-EDA.

### OUTPUT FORMAT
THOUGHT: <Detailed reasoning referencing specific column names and cleaning needs>
ACTION: <Semicolon-separated function calls, e.g., get_data_info();check_missing_values();plot_correlation_heatmap()>

### TOOL MAPPING
- get_data_info() -> Basic metadata
- get_data_stats() -> Descriptive statistics
- check_missing_values() -> Returns null counts per column
- drop_missing_values() -> Removes rows with any null values
- fill_missing_values(value=0) -> Replaces nulls with a fixed value
- get_top_features(target_col='col_name') -> Finds columns most correlated with target
- plot_correlation_heatmap() -> Visual heatmap of correlations
- plot_histogram(column='col_name', hue='opt_group') -> Distribution plots
- plot_scatter(x_col='col1', y_col='col2') -> Relationship plots

### RULES
1. ALWAYS start with Diagnosis tools for new datasets.
2. Include both `check_missing_values()` and at least one cleaning tool if the data requires it.
3. For a "Full Analysis", generate a sequence of 6-9 actions.
"""

class DataAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_query}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, user_query, df):
        columns_context = f"Dataset Columns: {df.columns.tolist()}\nUser Query: {user_query}"
        response = self.chain.invoke({"user_query": columns_context})
        return response

    def parse_action(self, response):
        thought, raw_action = "", response
        if "THOUGHT:" in response and "ACTION:" in response:
            thought = response.split("THOUGHT:")[1].split("ACTION:")[0].strip()
            raw_action = response.split("ACTION:")[1].strip()
        
        # Split multiple actions (autonomous plan)
        actions = []
        for call in raw_action.split(";"):
            match = re.search(r'(\w+)\s*\((.*)\)', call.strip())
            if match:
                actions.append((match.group(1), match.group(2)))
        
        return thought, actions

    def execute_tool(self, tool_name, args_str, df):
        if not tool_name or not hasattr(tools, tool_name):
            return f"Error: Tool {tool_name} not found."
        
        kwargs = {}
        if args_str:
            parts = re.findall(r'(\w+)\s*=\s*(?:["\']([^"\']*)["\']|(\w+))', args_str)
            for k, v1, v2 in parts:
                kwargs[k] = v1 if v1 else v2
        
        method = getattr(tools, tool_name)
        return method(df, **kwargs)

agent = DataAgent()
