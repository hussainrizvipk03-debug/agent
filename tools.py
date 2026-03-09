import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_data_info(df):
    info_str = f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}"
    return info_str

def get_data_stats(df):
    return df.describe()

def check_missing_values(df):
    return df.isnull().sum()

def drop_missing_values(df):
    return df.dropna()

def fill_missing_values(df, value=0):
    return df.fillna(value)

def get_correlation_matrix(df):
    return df.select_dtypes(include=['number']).corr()

def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f'Histogram of {column}')
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

def plot_scatter(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
    return fig
