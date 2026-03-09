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

def plot_histogram(df, column, hue=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue=hue, kde=True, ax=ax)
    title = f'Histogram of {column}'
    if hue:
        title += f' grouped by {hue}'
    ax.set_title(title)
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

def get_top_features(df, target_col):
    if target_col not in df.columns:
        return f"Error: Target column '{target_col}' not found."
    
    # Calculate correlations with the target column
    correlations = df.select_dtypes(include=['number']).corr()[target_col].abs().sort_values(ascending=False)
    # Return the top features (excluding the target itself)
    top_features = correlations.drop(target_col).head(5)
    return f"Top Features correlating with {target_col}:\n{top_features.to_string()}"

def get_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()
