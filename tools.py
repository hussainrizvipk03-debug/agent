import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_data_info(df: pd.DataFrame):
    info_str = f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}"
    return info_str

def get_data_stats(df: pd.DataFrame):
    return df.describe()

def check_missing_values(df: pd.DataFrame):
    return df.isnull().sum()

def drop_missing_values(df: pd.DataFrame):
    return df.dropna()

def fill_missing_values(df: pd.DataFrame, value=0):
    return df.fillna(value)

def get_correlation_matrix(df: pd.DataFrame):
    return df.select_dtypes(include=['number']).corr()

def plot_histogram(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    return plt

def plot_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    return plt

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    return plt
