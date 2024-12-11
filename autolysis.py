# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "scikit-learn",
#   "tabulate",
# ]
# ///

import sys
import os
import httpx
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import re
import pandas as pd
import seaborn as sns
import chardet
import matplotlib.pyplot as plt
import chardet
from dateutil import parser
import subprocess
import json


# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

# Function definitions
def detect_encoding(file_path):
    """
Detect the encoding of a CSV file.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def parse_date_with_regex(date_str):
    """
    Parse a date string using regex patterns to identify different date formats.
    """
    if not isinstance(date_str, str):  # Skip non-string values (e.g., NaN, float)
        return date_str  # Return the value as-is

    # Check if the string contains digits, as we expect date-like strings to contain numbers
    if not re.search(r'\d', date_str):
        return np.nan  # If no digits are found, it's not likely a date

    # Define regex patterns for common date formats
    patterns = [
        (r"\d{2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"),   # e.g., 15-Nov-2024
        (r"\d{2}-[A-Za-z]{3}-\d{2}", "%d-%b-%y"),   # e.g., 15-Nov-24
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),         # e.g., 2024-11-15
        (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),         # e.g., 11/15/2024
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),         # e.g., 15/11/2024
    ]

    # Check which regex pattern matches the date string
    for pattern, date_format in patterns:
        if re.match(pattern, date_str):
            try:
                return pd.to_datetime(date_str, format=date_format, errors='coerce')
            except Exception as e:
                print(f"Error parsing date: {date_str} with format {date_format}. Error: {e}")
                return np.nan

    # If no regex pattern matched, try dateutil parser as a fallback
    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=False)
    except Exception as e:
        print(f"Error parsing date with dateutil: {date_str}. Error: {e}")
        return np.nan

def is_date_column(column):
    """
    Determines whether a column likely contains dates based on column name or content.
    Checks if the column contains date-like strings and returns True if it's likely a date column.
    """
    # Check if the column name contains date-related terms
    if isinstance(column, str):
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            return True

    # Check the column's content for date-like patterns (e.g., strings with numbers)
    sample_values = column.dropna().head(10)  # Check the first 10 non-NaN values
    date_patterns = [r"\d{2}-[A-Za-z]{3}-\d{2}",r"\d{2}-[A-Za-z]{3}-\d{4}", r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"]

    for value in sample_values:
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
    return False

def read_csv(file_path):
    """
    Read a CSV file with automatic encoding detection and flexible date parsing using regex.
    """
    try:
        print("Detecting file encoding...")
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")

        # Load the CSV file with the detected encoding
        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')

        # Attempt to parse date columns using regex
        for column in df.columns:
            if df[column].dtype == object and is_date_column(df[column]):
                # Only apply date parsing to columns likely containing dates
                print(f"Parsing dates in column: {column}")
                df[column] = df[column].apply(parse_date_with_regex)

        return df

    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)


def perform_advanced_analysis(df):
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    for column in df.select_dtypes(include=[np.datetime64]).columns:
        df[column] = df[column].dt.strftime('%Y-%m-%d %H:%M:%S')
    outliers = detect_outliers(df)
    if outliers is not None:
        analysis["outliers"] = outliers.value_counts().to_dict()
    return analysis

def detect_outliers(df):
    """Detect outliers using Isolation Forest."""
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return None
    iso = IsolationForest(contamination=0.05, random_state=42)
    numeric_data["outliers"] = iso.fit_predict(numeric_data)
    return numeric_data["outliers"]

def regression_analysis(df):
    """Perform regression analysis on numeric columns."""
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2:
        return None
    x = numeric_data.iloc[:, :-1]  # Independent variables
    y = numeric_data.iloc[:, -1]  # Dependent variable
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    metrics = {
        "MSE": mean_squared_error(y, predictions),
        "R2": r2_score(y, predictions),
        "Coefficients": dict(zip(x.columns, model.coef_)),
    }
    return metrics

def clustering_analysis(df):
    """Perform clustering analysis on numeric columns."""
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    datetime_columns = df.select_dtypes(include=[np.datetime64])
    for col in datetime_columns:
        numeric_data[col] = (df[col] - df[col].min()).dt.days
    if numeric_data.empty:
        return None, None
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        numeric_data['Cluster'] = kmeans.fit_predict(numeric_data)
        return numeric_data['Cluster'], numeric_data.index
    except Exception as e:
        print(f"Error while Clustering: {e}")
        return None,None

def summarize_correlation(df):
    """Summarize key insights from the correlation matrix."""
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return "No numeric data available to compute correlations."

    corr_matrix = numeric_data.corr()

    # Get the highest correlation pairs
    correlations = corr_matrix.unstack().sort_values(ascending=False)

    # Filter out self-correlation (corr with itself)
    correlations = correlations[correlations < 1]

    # Get the top 5 most correlated variable pairs
    top_correlations = correlations.head(5)

    summary = "Top 5 most correlated variables:\n"
    for idx, corr_value in top_correlations.items():
        summary += f"{idx[0]} & {idx[1]}: {corr_value:.2f}\n"

    return summary

def summarize_pairplot(df):
    """Summarize the relationships between numeric variables."""
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return "No numeric data available to analyze in pairplot."

    # Count the number of variables
    num_vars = len(numeric_data.columns)

    summary = f"A pairplot has been created with {num_vars} numeric variables.\n"

    # Describe pairwise relationships (this can be extended to specifics based on domain knowledge)
    if num_vars > 1:
        summary += "The pairplot shows the pairwise relationships between the variables, helping to identify trends, correlations, and possible outliers.\n"
    else:
        summary += "Only one numeric variable is present, so no pairwise relationships could be visualized.\n"

    return summary

def summarize_clustering(df, clusters):
    """Summarize the results of clustering analysis."""
    if clusters is None or len(clusters) == 0:
        return "No clustering results available."

    # Add the cluster labels to the dataframe for analysis
    df['Cluster'] = clusters

    # Count the number of samples in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_values(ascending=False)

    summary = "Clustering results summary:\n"
    for cluster, count in cluster_counts.items():
        summary += f"Cluster {cluster}: {count} samples\n"

    return summary

def generate_summary(df, clusters):
    """Generate a full summary based on the analysis and visualizations."""
    correlation_summary = summarize_correlation(df)
    pairplot_summary = summarize_pairplot(df)
    clustering_summary = summarize_clustering(df, clusters)

    # Combine the summaries into a single narrative
    full_summary = (
        "### Data Analysis Summary\n\n"
        f"#### Correlation Insights:\n{correlation_summary}\n\n"
        f"#### Pairplot Insights:\n{pairplot_summary}\n\n"
        f"#### Clustering Insights:\n{clustering_summary}\n"
    )

    return full_summary


def visualize_advanced(df, output_folder):
    visualizations = []

    # Correlation Heatmap
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if not numeric_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        file_path = os.path.join(output_folder, "correlation_heatmap.png")
        plt.savefig(file_path)
        visualizations.append(file_path)
        plt.close()

    # Pairplot
    try:
        sns.pairplot(df.select_dtypes(include=[np.number]).dropna())
        file_path = os.path.join(output_folder, "pairplot.png")
        plt.savefig(file_path)
        visualizations.append(file_path)
    except Exception as e:
        print(f"Error creating pairplot: {e}")

    # Clustering Scatter Plot
    clusters, valid_indices = clustering_analysis(df)
    summary = generate_summary(df,clusters)
    if clusters is not None and len(valid_indices) > 1:
        df_with_clusters = numeric_data.loc[valid_indices].copy()
        df_with_clusters["Cluster"] = clusters.values
        plt.figure(figsize=(10, 8))
        for cluster in np.unique(clusters):
            subset = df_with_clusters[df_with_clusters["Cluster"] == cluster]
            # Ensure that the values are numeric and handle NaNs or infinite values
            subset.iloc[:, 0] = pd.to_numeric(subset.iloc[:, 0], errors='coerce')
            subset.iloc[:, 1] = pd.to_numeric(subset.iloc[:, 1], errors='coerce')

            subset = subset.dropna(subset=[subset.columns[0], subset.columns[1]])
            subset = subset[~subset.isin([np.inf, -np.inf]).any(axis=1)]

        # Now plot the data
            plt.scatter(subset.iloc[:, 0].astype(float), subset.iloc[:, 1].astype(float), label=f"Cluster {cluster}")
        plt.legend()
        file_path = os.path.join(output_folder, "clustering_scatter.png")
        plt.savefig(file_path)
        visualizations.append(file_path)

    return visualizations, summary

def query_llm(prompt):
    """
Queries the LLM for insights and returns the response.
    """
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",  # Supported chat model
            "messages": [
                {"role": "system", "content": "You are a helpful data analysis assistant. Provide insights, suggestions, and implications based on the given analysis and visualizations."},
                {"role": "user", "content": prompt},
            ],
        }
        payload_json = json.dumps(payload)
        curl_command = [
            "curl",
            "-X", "POST", url,
            "-H", f"Authorization: Bearer {AIPROXY_TOKEN}",
            "-H", "Content-Type: application/json",
            "-d", payload_json
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            return response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error in curl request: {result.stderr}")
    except Exception as e:
        print(f"Error querying AI Proxy: {e}")
        return "Error: Unable to generate narrative."

def create_story(analysis, visualizations_summary):
    """
Creates a narrative using LLM based on analysis and visualizations.
    """
    prompt = (
        f"### Data Summary:\nShape: {analysis['shape']}\n"
        f"Columns: {', '.join(list(analysis['columns'].keys()))}\n"
        f"Missing Values: {str(analysis['missing_values'])}\n\n"
        f"### Key Summary Statistics:\nTop 3 Columns:\n{pd.DataFrame(analysis['summary_statistics']).iloc[:, :3].to_string()}\n\n"
        f"### Visualizations:\nCorrelation heatmap, Pairplot, Clustering Scatter Plot.\n\n"
        "Based on the above, provide a detailed narrative including insights and potential actions."
    )

    return query_llm(prompt)


def save_results(analysis, visualizations, story, output_folder):
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Data Overview\n")
        f.write(f"**Shape**: {analysis['shape']}\n\n")
        f.write("## Summary Statistics\n")
        f.write(pd.DataFrame(analysis["summary_statistics"]).to_markdown())
        f.write("## Narrative\n")
        f.write(str(story))  # if story is a list of strings

def main():
    print("Starting script...")
    if len(sys.argv) != 2:
        print("Incorrect arguments. Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)


    file_path = sys.argv[1]
    print(f"Reading file: {file_path}")
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = dataset_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output folder created: {output_folder}")
    df = read_csv(file_path)
    print("Dataframe loaded.")
    analysis = perform_advanced_analysis(df)
    print("Analysis complete.")
    visualizations, summary = visualize_advanced(df, output_folder)
    print(f"Generated visualizations: {visualizations}")
    story = create_story(analysis, summary)
    print("Story created.")
    save_results(analysis,visualizations,story, output_folder)
    print("Results saved.")

if __name__ == "__main__":
    main()