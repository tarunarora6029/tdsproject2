# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "python-dotenv",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from dotenv import load_dotenv
import argparse

# Configure matplotlib backend
plt.switch_backend("Agg")

# Load environment variables
load_dotenv()

# Constants
API_URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        sys.exit(f"Error loading file: {e}")

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])
    return {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()
    }

def visualize_data(df, output_dir):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    for column in numeric_columns:
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
        plt.clf()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "")
    except Exception as e:
        print(f"Error during narrative generation: {e}")
        return "Narrative generation failed due to an error."

def save_narrative(narrative, output_dir):
    """Save the generated narrative to a README file."""
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(narrative)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default=None, help="Directory to save outputs.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set output directory
    output_dir = args.output_dir or os.path.splitext(os.path.basename(args.file_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Process data
    df = load_data(args.file_path)
    analysis = analyze_data(df)

    # Generate outputs
    visualize_data(df, output_dir)
    narrative = generate_narrative(analysis)
    save_narrative(narrative, output_dir)

if __name__ == "__main__":
    main()
