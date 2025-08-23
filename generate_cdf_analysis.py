#!/usr/bin/env python3
"""
Generate CDF graphs for each Model in XSUM dataset per metric
and process JSON files from Dialogset directory.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list of records")
            return []
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_metrics_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract all metrics from a single record, handling both flat and nested structures."""
    metrics = {}
    
    # Handle nested metrics structure
    if 'metrics' in record and isinstance(record['metrics'], dict):
        for key, value in record['metrics'].items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    
    # Handle flat structure - extract metric keys directly from record
    metric_suffixes = ['_before', '_after']
    metric_prefixes = ['EHI', 'EF1', 'PH', 'OF', 'LF', 'NH', 'EF', 'ROUGE1', 'ROUGE2', 'ROUGEL', 
                      'EHI_CAL', 'EHI_EXP', 'EHI_W']
    
    for key, value in record.items():
        if isinstance(value, (int, float)):
            # Check if it's a metric key
            if any(key.endswith(suffix) for suffix in metric_suffixes):
                metrics[key] = float(value)
            elif any(key.startswith(prefix) for prefix in metric_prefixes):
                metrics[key] = float(value)
    
    return metrics

def extract_all_metrics(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract all metrics from data and return as DataFrame."""
    all_metrics = []
    
    for i, record in enumerate(data):
        metrics = extract_metrics_from_record(record)
        metrics['record_id'] = record.get('id', i)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_metrics)
    return df

def get_model_name_from_filename(filename: str) -> str:
    """Extract model name from filename."""
    filename_lower = filename.lower()
    if 'flant5' in filename_lower or 'flan' in filename_lower:
        return 'FlanT5'
    elif 'mistral' in filename_lower:
        return 'Mistral'
    elif 'distilbart' in filename_lower or 'distil' in filename_lower:
        return 'DistilBART'
    else:
        return 'Unknown'

def plot_cdf(data: List[float], title: str, xlabel: str, output_path: str):
    """Generate CDF plot for given data."""
    if not data or len(data) == 0:
        print(f"No data to plot for {title}")
        return
    
    # Remove NaN values
    data = [x for x in data if not pd.isna(x)]
    if not data:
        print(f"No valid data to plot for {title}")
        return
    
    # Sort data for CDF
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, y, linewidth=2, marker='o', markersize=3)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CDF plot: {output_path}")

def generate_summary_table(datasets_metrics: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Generate summary statistics table."""
    summary_data = []
    
    for dataset_name, models_data in datasets_metrics.items():
        for model_name, df in models_data.items():
            if df.empty:
                continue
                
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'record_id']
            
            for metric in numeric_cols:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Metric': metric,
                        'Count': len(values),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Q25': values.quantile(0.25),
                        'Median': values.median(),
                        'Q75': values.quantile(0.75),
                        'Max': values.max()
                    })
    
    return pd.DataFrame(summary_data)

def process_dataset(dataset_path: str, dataset_name: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Process all JSON files in a dataset directory."""
    print(f"\nProcessing {dataset_name} dataset...")
    
    dataset_path = Path(dataset_path)
    models_data = {}
    
    # Find all JSON files
    json_files = list(dataset_path.glob('*.json'))
    
    for json_file in json_files:
        print(f"  Processing file: {json_file.name}")
        
        # Load data
        data = load_json_file(str(json_file))
        if not data:
            continue
        
        # Extract model name
        model_name = get_model_name_from_filename(json_file.name)
        
        # Extract metrics
        df = extract_all_metrics(data)
        if df.empty:
            print(f"    No metrics found in {json_file.name}")
            continue
        
        models_data[model_name] = df
        
        # Generate CDF plots for each metric
        output_model_dir = Path(output_dir) / dataset_name / model_name
        output_model_dir.mkdir(parents=True, exist_ok=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'record_id']
        
        for metric in numeric_cols:
            values = df[metric].dropna().tolist()
            if len(values) > 0:
                plot_title = f'{dataset_name} - {model_name} - {metric} (CDF)'
                output_path = output_model_dir / f'{metric}_cdf.png'
                plot_cdf(values, plot_title, metric, str(output_path))
    
    return models_data

def main():
    """Main function to process all datasets and generate outputs."""
    
    # Define paths
    base_path = Path('/home/runner/work/colab_scratchpad/colab_scratchpad')
    newset_path = base_path / 'Newset'
    output_dir = base_path / 'cdf_analysis_output'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Process datasets
    datasets_metrics = {}
    
    # Process XSUM dataset
    xsum_path = newset_path / 'XSUM'
    if xsum_path.exists():
        datasets_metrics['XSUM'] = process_dataset(str(xsum_path), 'XSUM', str(output_dir))
    
    # Process Dialogset dataset
    dialogset_path = newset_path / 'Dialogset'
    if dialogset_path.exists():
        datasets_metrics['Dialogset'] = process_dataset(str(dialogset_path), 'Dialogset', str(output_dir))
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_df = generate_summary_table(datasets_metrics)
    
    if not summary_df.empty:
        # Save summary table
        summary_path = output_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary table: {summary_path}")
        
        # Display summary table
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Save formatted summary table
        formatted_summary_path = output_dir / 'summary_statistics_formatted.txt'
        with open(formatted_summary_path, 'w') as f:
            f.write("Summary Statistics for All Datasets and Models\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary_df.to_string(index=False, float_format='%.4f'))
        print(f"Saved formatted summary: {formatted_summary_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()