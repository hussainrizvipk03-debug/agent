from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

# Only handles the initialization of the LLM
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

if __name__ == "__main__":
    print("LLM is ready.")
