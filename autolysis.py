import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np

# Ensure AIPROXY_TOKEN is set
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set")
    sys.exit(1)

AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/"

def load_data(filename):
    """
    Load CSV file with error handling and basic preprocessing
    """
    try:
        data = pd.read_csv(filename, low_memory=False)
        if data.empty:
            raise ValueError("The CSV file is empty")
        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(data):
    """
    Perform comprehensive data analysis
    """
    analysis = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    analysis['summary'] = data[numeric_cols].describe().to_dict()
    analysis['missing_values'] = data.isnull().sum().to_dict()
    if len(numeric_cols) > 1:
        try:
            analysis['correlation_matrix'] = data[numeric_cols].corr().to_dict()
        except Exception:
            analysis['correlation_matrix'] = "Correlation matrix could not be computed"
    outliers = {}
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_range = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        outliers[col] = len(data[(data[col] < outlier_range[0]) | (data[col] > outlier_range[1])])
    analysis['outliers'] = outliers
    return analysis

def ask_llm(prompt):
    """
    Call LLM with error handling and token management
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7
        }
        response = requests.post(f"{AIPROXY_URL}v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM API call failed: {e}")
        return "Unable to generate LLM analysis"

def create_visualizations(data, filename):
    """
    Create multiple types of visualizations
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(15, len(numeric_cols) * 3))
    for ax, col in zip(axes, numeric_cols):
        data[col].hist(bins=20, ax=ax)
        ax.set_title(col, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    plt.tight_layout()
    plt.savefig(f"{filename}_analysis_chart1.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        try:
            correlation_matrix = data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Heatmap")
        except Exception:
            plt.text(0.5, 0.5, "Correlation not possible", ha='center')
    plt.savefig(f"{filename}_analysis_chart2.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols].boxplot()
    plt.title("Box Plot of Numeric Columns")
    plt.xticks(rotation=45)
    plt.savefig(f"{filename}_analysis_chart3.png")
    plt.close()

def write_story(data, analysis, charts):
    """
    Generate a narrative using LLM
    """
    story_prompt = f"""Analyze this dataset with the following context:
    - Total rows: {len(data)}
    - Total columns: {len(data.columns)}
    - Column names: {', '.join(data.columns)}
    
    Analysis Summary:
    {analysis}
    
    Create a narrative that:
    1. Describes the dataset
    2. Highlights key insights from the analysis
    3. Suggests potential implications or further investigations
    4. Explains the significance of the visualizations
    
    Use the analysis and charts to craft an engaging, informative story."""
    
    story = ask_llm(story_prompt)
    
    with open("README.md", "w") as file:
        file.write("# Data Analysis Report\n\n")
        file.write(story)

def main(filename):
    """
    Main workflow
    """
    data = load_data(filename)
    analysis = analyze_data(data)
    create_visualizations(data, os.path.splitext(filename)[0])
    write_story(data, analysis, ['*_analysis_charts.png'])
    print("Analysis complete. Check README.md and generated charts.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    main(filename)
