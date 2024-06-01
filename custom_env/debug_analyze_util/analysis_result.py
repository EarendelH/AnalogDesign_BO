import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os


def convert_to_float(value):
    """
    Convert string with magnitude suffix to float.

    Args:
        value (str): The value to be converted.

    Returns:
        float: Converted float value.
    """
    if isinstance(value, str):
        multipliers = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'M': 1e6}
        match = re.match(r"([-+]?\d*\.?\d+)([pnumkM]?)", value)
        if match:
            number, suffix = match.groups()
            return float(number) * multipliers.get(suffix, 1)
    return float(value)


def preprocess_data(data):
    """
    Preprocess the input data by converting magnitude strings to floats,
    merging specific columns, and standardizing the data.

    Args:
        data (pd.DataFrame): The input data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed and standardized data.
    """
    print("Starting data preprocessing...")
    # Drop missing values
    data = data.dropna()

    # Convert magnitude strings to floats
    print("Converting magnitude strings to floats...")
    for column in data.columns:
        data[column] = data[column].apply(convert_to_float)

    # Process columns starting with 'nf_' and their corresponding 'w_' columns
    print("Processing columns starting with 'nf_'...")
    nf_columns = [col for col in data.columns if col.startswith('nf_')]
    for nf_col in nf_columns:
        device_name = nf_col[3:]
        w_col = f'w_{device_name}_per_finger'
        if w_col in data.columns:
            new_col_name = f'w_{device_name}'
            data[new_col_name] = data[w_col] * data[nf_col]
            data.drop(columns=[w_col, nf_col], inplace=True)

    # Standardize the data
    print("Standardizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data preprocessing completed.")
    return pd.DataFrame(data_scaled, columns=data.columns)


def correlation_analysis(data, target):
    """
    Perform correlation analysis between input data and target.

    Args:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target data.

    Returns:
        pd.Series: Correlation values for each input feature.
    """
    correlations = data.corrwith(target)
    return correlations


def mutual_info_analysis(data, target):
    """
    Perform mutual information analysis between input data and target.

    Args:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target data.

    Returns:
        pd.Series: Mutual information values for each input feature.
    """
    mutual_info = {}
    for col in data.columns:
        mutual_info[col] = mutual_info_score(data[col], target)
    return pd.Series(mutual_info)


def feature_importance_linear(data, target):
    """
    Compute feature importance using linear regression.

    Args:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target data.

    Returns:
        pd.Series: Feature importance scores for each input feature.
    """
    model = LinearRegression()
    model.fit(data, target)
    importance = model.coef_
    return pd.Series(importance, index=data.columns)


def feature_importance_rf(data, target):
    """
    Compute feature importance using random forest.

    Args:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target data.

    Returns:
        pd.Series: Feature importance scores for each input feature.
    """
    model = RandomForestRegressor()
    model.fit(data, target)
    importance = model.feature_importances_
    return pd.Series(importance, index=data.columns)


def cluster_inputs(data, n_clusters):
    """
    Perform clustering on the input data.

    Args:
        data (pd.DataFrame): The input data.
        n_clusters (int): The number of clusters.

    Returns:
        np.ndarray: Cluster labels for each input sample.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    return clusters


# Read the Excel file
file_path = '/mnt/data/Demo.xlsx'
output_dir = os.path.dirname(file_path)
data = pd.read_excel(file_path)

# Ignore the first column, use columns 2 to 23 as output variables, and columns 24 onwards as input variables
outputs = data.iloc[:, 1:23]
inputs = data.iloc[:, 23:]

# Preprocess the input data
inputs_preprocessed = preprocess_data(inputs)

# Set the number of clusters
n_clusters = 3  # Adjust this value based on your requirements

# Process each output variable individually
total_outputs = len(outputs.columns)
for i, output_column in enumerate(outputs.columns):
    print(f"Processing output variable {output_column} ({i + 1}/{total_outputs})...")
    target = outputs[output_column]

    # Perform correlation analysis
    correlations = correlation_analysis(inputs_preprocessed, target)
    mutual_info = mutual_info_analysis(inputs_preprocessed, target)

    # Compute feature importance
    importance_linear = feature_importance_linear(inputs_preprocessed, target)
    importance_rf = feature_importance_rf(inputs_preprocessed, target)

    # Cross-validation
    linear_model = LinearRegression()
    linear_scores = cross_val_score(linear_model, inputs_preprocessed, target, cv=5)

    rf_model = RandomForestRegressor()
    rf_scores = cross_val_score(rf_model, inputs_preprocessed, target, cv=5)

    print(f"Output variable: {output_column}")
    print("Correlation analysis:\n", correlations)
    print("Mutual information analysis:\n", mutual_info)
    print("Linear regression feature importance:\n", importance_linear)
    print("Random forest feature importance:\n", importance_rf)
    print("Linear regression cross-validation scores:", linear_scores)
    print("Random forest cross-validation scores:", rf_scores)

    # Perform clustering
    clusters = cluster_inputs(inputs_preprocessed, n_clusters)

    # Save cluster plot
    sns.scatterplot(x=inputs_preprocessed.iloc[:, 0], y=inputs_preprocessed.iloc[:, 1], hue=clusters, palette="viridis")
    plt.title(f'Input Clusters for Output {output_column}')
    cluster_plot_path = os.path.join(output_dir, f'Input_Clusters_{output_column}.png')
    plt.savefig(cluster_plot_path)
    plt.clf()

    # Save linear regression feature importance plot
    importance_linear.sort_values().plot(kind='barh', title=f'Linear Regression Feature Importance for {output_column}')
    linear_importance_plot_path = os.path.join(output_dir, f'Linear_Importance_{output_column}.png')
    plt.savefig(linear_importance_plot_path)
    plt.clf()

    # Save random forest feature importance plot
    importance_rf.sort_values().plot(kind='barh', title=f'Random Forest Feature Importance for {output_column}')
    rf_importance_plot_path = os.path.join(output_dir, f'RF_Importance_{output_column}.png')
    plt.savefig(rf_importance_plot_path)
    plt.clf()

    print(f"Output variable {output_column} processing completed, plots saved.\n")

print("All output variables processed.")