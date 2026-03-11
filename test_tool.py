from typing import Annotated, Any
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolArg
import pandas as pd

@tool
def get_data_info(df: Annotated[Any, InjectedToolArg()]):
    """test"""
    pass

try:
    print("Schema properties:", get_data_info.args_schema.schema().get("properties", {}))
except Exception as e:
    print("Error:", e)
