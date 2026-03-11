from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolArg
from typing import Annotated, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@tool
def get_data_info(df: Annotated[Any, InjectedToolArg()]):
    """Returns basic metadata about the dataset, like number of rows and columns."""
    info_str = f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}"
    return info_str

@tool
def get_data_stats(df: Annotated[Any, InjectedToolArg()]):
    """Returns descriptive statistics for the numerical columns in the dataset."""
    return df.describe().to_string()

@tool
def check_missing_values(df: Annotated[Any, InjectedToolArg()]):
    """Returns the count of missing (null) values for each column."""
    return df.isnull().sum().to_string()

@tool
def drop_missing_values(df: Annotated[Any, InjectedToolArg()]):
    """Removes all rows that contain missing (null) values. Use this to clean the data."""
    return df.dropna()

@tool
def fill_missing_values(df: Annotated[Any, InjectedToolArg()], value: float = 0):
    """Replaces missing (null) values with a specified value (default is 0)."""
    return df.fillna(value)

@tool
def get_correlation_matrix(df: Annotated[Any, InjectedToolArg()]):
    """Calculates the correlation matrix for numerical columns."""
    return df.select_dtypes(include=['number']).corr().to_string()

@tool
def plot_histogram(df: Annotated[Any, InjectedToolArg()], column: str, hue: str = None):
    """Creates a histogram plot for a specific column. Optional 'hue' for grouping."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue=hue, kde=True, ax=ax)
    title = f'Histogram of {column}'
    if hue:
        title += f' grouped by {hue}'
    ax.set_title(title)
    return fig

@tool
def plot_correlation_heatmap(df: Annotated[Any, InjectedToolArg()]):
    """Generates a heatmap visualization of column correlations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

@tool
def plot_scatter(df: Annotated[Any, InjectedToolArg()], x_col: str, y_col: str):
    """Creates a scatter plot to visualize the relationship between two columns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
    return fig

@tool
def get_top_features(df: Annotated[Any, InjectedToolArg()], target_col: str):
    """Finds the top 5 features most correlated with a target column."""
    if target_col not in df.columns:
        return f"Error: Target column '{target_col}' not found."
    
    correlations = df.select_dtypes(include=['number']).corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.drop(target_col).head(5)
    return f"Top Features correlating with {target_col}:\n{top_features.to_string()}"

@tool
def get_numeric_columns(df: Annotated[Any, InjectedToolArg()]):
    """Returns a list of all numeric column names in the dataset."""
    return df.select_dtypes(include=['number']).columns.tolist()
