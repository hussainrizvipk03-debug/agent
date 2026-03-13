from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

# Load environment variables from the .env file in the current directory
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

# Only handles the initialization of the LLM
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)

if __name__ == "__main__":
    print("LLM is ready.")
