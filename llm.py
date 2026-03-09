from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=60)
if __name__ == "__main__":
    try:
        response = llm.invoke("Hello, how are you?")
        print("LLM Response:", response.content)
    except Exception as e:
        print("Error calling LLM:", e)

