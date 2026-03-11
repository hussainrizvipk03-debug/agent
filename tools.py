from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolArg
from typing import Annotated, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

@tool
def get_data_info(df: Annotated[Any, InjectedToolArg()]):
    """Returns basic metadata about the dataset, like number of rows and columns."""
    info_str = f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}"
    return info_str

@tool
def get_data_stats(df: Annotated[Any, InjectedToolArg()]):
    """Returns descriptive statistics for all columns in the dataset."""
    return df.describe(include='all').to_string()

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
def plot_univariate_analysis(df: Annotated[Any, InjectedToolArg()], column: str):
    """Performs univariate analysis (histogram and boxplot) for a single column."""
    if column not in df.columns:
        return f"Error: Column '{column}' not found in dataset. Available: {df.columns.tolist()}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(data=df, x=column, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}')
    sns.boxplot(data=df, y=column, ax=ax2)
    ax2.set_title(f'Outliers in {column}')
    plt.tight_layout()
    return fig

@tool
def plot_bivariate_analysis(df: Annotated[Any, InjectedToolArg()], x_col: str, y_col: str, hue: str = None):
    """Performs bivariate analysis (scatter or boxplot depending on types)."""
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: One or both columns ('{x_col}', '{y_col}') not found. Available: {df.columns.tolist()}"
    fig, ax = plt.subplots(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        ax.set_title(f'Relationship: {x_col} vs {y_col}')
    else:
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        ax.set_title(f'Comparison: {x_col} by {y_col}')
    return fig

@tool
def plot_multivariate_analysis(df: Annotated[Any, InjectedToolArg()], columns: list):
    """Performs multivariate analysis using a pairplot for the specified columns."""
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return f"Error: No valid columns found in {columns}. Available: {df.columns.tolist()}"
    g = sns.pairplot(df[valid_cols].select_dtypes(include=['number']))
    return g.fig

@tool
def plot_correlation_heatmap(df: Annotated[Any, InjectedToolArg()]):
    """Generates a heatmap visualization of column correlations."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return "Error: No numeric columns found for heatmap."
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

@tool
def get_top_features(df: Annotated[Any, InjectedToolArg()], target_col: str):
    """Finds the top 5 features most correlated with a target column."""
    if target_col not in df.columns:
        return f"Error: Target column '{target_col}' not found. Available: {df.columns.tolist()}"
    correlations = df.select_dtypes(include=['number']).corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.drop(target_col).head(5)
    return f"Top Features correlating with {target_col}:\n{top_features.to_string()}"
