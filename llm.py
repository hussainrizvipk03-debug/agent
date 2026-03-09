from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()

system_prompt = """
You are a strict Data Analysis AI Agent. 

# METHOD OF DOING 
- If user wants info regarding dataframe -> get_data_info()
- If user wants statistics -> get_data_stats()
- If user wants missing values -> check_missing_values()
- If user wants to drop missing values -> drop_missing_values()
- If user wants to fill missing values -> fill_missing_values(value=0)
- If user wants correlation matrix -> get_correlation_matrix()
- If user wants histogram -> plot_histogram(column='column_name')
- If user wants heatmap -> plot_correlation_heatmap()
- If user wants scatter plot -> plot_scatter(x_col='col1', y_col='col2')

### RULES
1. ONLY output the function call string. Do not use prefixes like "output:".
2. If invalid request, respond: "I can only perform data analysis tasks using my provided tools."
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_query}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    user_query = input("Ask me anything about your data: ")
    response = chain.invoke({"user_query": user_query})
    print("\nAI Command:", response)
