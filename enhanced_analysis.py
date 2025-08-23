#!/usr/bin/env python3
"""
Enhanced CDF analysis with comparative visualizations and detailed tables
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

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_comparative_cdf_plots(datasets_metrics: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """Create comparative CDF plots across models for each metric."""
    print("Creating comparative CDF plots...")
    
    comp_output_dir = Path(output_dir) / 'comparative_plots'
    comp_output_dir.mkdir(exist_ok=True)
    
    # Get all unique metrics across all datasets and models
    all_metrics = set()
    for dataset_data in datasets_metrics.values():
        for model_df in dataset_data.values():
            if not model_df.empty:
                numeric_cols = model_df.select_dtypes(include=[np.number]).columns
                all_metrics.update([col for col in numeric_cols if col != 'record_id'])
    
    # Create comparative plots for each dataset
    for dataset_name, models_data in datasets_metrics.items():
        if not models_data:
            continue
            
        dataset_comp_dir = comp_output_dir / dataset_name
        dataset_comp_dir.mkdir(exist_ok=True)
        
        # Get metrics that are present in at least 2 models
        metric_counts = {}
        for model_name, df in models_data.items():
            if df.empty:
                continue
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for metric in numeric_cols:
                if metric != 'record_id':
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        common_metrics = [m for m, count in metric_counts.items() if count >= 2]
        
        for metric in common_metrics:
            plt.figure(figsize=(12, 8))
            
            for model_name, df in models_data.items():
                if df.empty or metric not in df.columns:
                    continue
                
                values = df[metric].dropna()
                if len(values) == 0:
                    continue
                
                # Calculate CDF
                sorted_data = np.sort(values)
                n = len(sorted_data)
                y = np.arange(1, n + 1) / n
                
                plt.plot(sorted_data, y, linewidth=2, marker='o', markersize=2, 
                        label=f'{model_name} (n={n})', alpha=0.8)
            
            plt.title(f'{dataset_name} Dataset - {metric} Comparison (CDF)', fontsize=16, fontweight='bold')
            plt.xlabel(metric, fontsize=14)
            plt.ylabel('Cumulative Probability', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = dataset_comp_dir / f'{metric}_comparison_cdf.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparative plot: {output_path}")

def create_metric_correlation_heatmap(datasets_metrics: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """Create correlation heatmaps for metrics within each model."""
    print("Creating correlation heatmaps...")
    
    corr_output_dir = Path(output_dir) / 'correlation_heatmaps'
    corr_output_dir.mkdir(exist_ok=True)
    
    for dataset_name, models_data in datasets_metrics.items():
        for model_name, df in models_data.items():
            if df.empty:
                continue
            
            # Get numeric columns excluding record_id
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'record_id']
            
            if len(numeric_cols) < 2:
                continue
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            plt.title(f'{dataset_name} - {model_name}\nMetric Correlations', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = corr_output_dir / f'{dataset_name}_{model_name}_correlations.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved correlation heatmap: {output_path}")

def create_before_after_comparison(datasets_metrics: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """Create before/after comparison plots."""
    print("Creating before/after comparison plots...")
    
    ba_output_dir = Path(output_dir) / 'before_after_comparisons'
    ba_output_dir.mkdir(exist_ok=True)
    
    for dataset_name, models_data in datasets_metrics.items():
        for model_name, df in models_data.items():
            if df.empty:
                continue
            
            # Find before/after metric pairs
            metric_pairs = []
            columns = df.columns.tolist()
            
            for col in columns:
                if col.endswith('_before'):
                    base_metric = col[:-7]  # Remove '_before'
                    after_col = base_metric + '_after'
                    if after_col in columns:
                        metric_pairs.append((col, after_col, base_metric))
            
            if not metric_pairs:
                continue
            
            # Create subplots for all before/after pairs
            n_metrics = len(metric_pairs)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, (before_col, after_col, base_metric) in enumerate(metric_pairs):
                ax = axes[i] if i < len(axes) else plt.subplot(n_rows, n_cols, i+1)
                
                before_vals = df[before_col].dropna()
                after_vals = df[after_col].dropna()
                
                if len(before_vals) > 0:
                    sorted_before = np.sort(before_vals)
                    y_before = np.arange(1, len(sorted_before) + 1) / len(sorted_before)
                    ax.plot(sorted_before, y_before, label=f'Before (n={len(before_vals)})', 
                           linewidth=2, alpha=0.8)
                
                if len(after_vals) > 0:
                    sorted_after = np.sort(after_vals)
                    y_after = np.arange(1, len(sorted_after) + 1) / len(sorted_after)
                    ax.plot(sorted_after, y_after, label=f'After (n={len(after_vals)})', 
                           linewidth=2, alpha=0.8)
                
                ax.set_title(f'{base_metric}', fontsize=12, fontweight='bold')
                ax.set_xlabel(base_metric, fontsize=10)
                ax.set_ylabel('Cumulative Probability', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'{dataset_name} - {model_name}\nBefore vs After Comparison', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = ba_output_dir / f'{dataset_name}_{model_name}_before_after.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved before/after comparison: {output_path}")

def create_detailed_summary_table(datasets_metrics: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """Create a more detailed summary table with additional statistics."""
    print("Creating detailed summary table...")
    
    detailed_summary = []
    
    for dataset_name, models_data in datasets_metrics.items():
        for model_name, df in models_data.items():
            if df.empty:
                continue
                
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'record_id']
            
            for metric in numeric_cols:
                values = df[metric].dropna()
                if len(values) == 0:
                    continue
                
                # Calculate additional statistics
                skewness = values.skew()
                kurtosis = values.kurtosis()
                cv = values.std() / values.mean() if values.mean() != 0 else np.inf
                
                # Determine metric category
                if '_before' in metric:
                    category = 'Before'
                    base_metric = metric[:-7]
                elif '_after' in metric:
                    category = 'After'
                    base_metric = metric[:-6]
                else:
                    category = 'Other'
                    base_metric = metric
                
                detailed_summary.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Metric': metric,
                    'Base_Metric': base_metric,
                    'Category': category,
                    'Count': len(values),
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'CV': cv,  # Coefficient of variation
                    'Skewness': skewness,
                    'Kurtosis': kurtosis,
                    'Min': values.min(),
                    'Q25': values.quantile(0.25),
                    'Median': values.median(),
                    'Q75': values.quantile(0.75),
                    'Max': values.max(),
                    'IQR': values.quantile(0.75) - values.quantile(0.25),
                    'Range': values.max() - values.min()
                })
    
    detailed_df = pd.DataFrame(detailed_summary)
    
    if not detailed_df.empty:
        # Save detailed summary
        detailed_path = Path(output_dir) / 'detailed_summary_statistics.csv'
        detailed_df.to_csv(detailed_path, index=False)
        
        # Create pivot tables for easier analysis
        for dataset in detailed_df['Dataset'].unique():
            dataset_data = detailed_df[detailed_df['Dataset'] == dataset]
            
            # Mean comparison table
            pivot_mean = dataset_data.pivot_table(
                values='Mean', 
                index=['Base_Metric', 'Category'], 
                columns='Model', 
                aggfunc='first'
            )
            
            mean_path = Path(output_dir) / f'{dataset}_mean_comparison.csv'
            pivot_mean.to_csv(mean_path)
            
            # Before/After improvement table
            before_after = dataset_data[dataset_data['Category'].isin(['Before', 'After'])]
            if not before_after.empty:
                improvement_data = []
                for model in before_after['Model'].unique():
                    model_data = before_after[before_after['Model'] == model]
                    for base_metric in model_data['Base_Metric'].unique():
                        metric_data = model_data[model_data['Base_Metric'] == base_metric]
                        before_row = metric_data[metric_data['Category'] == 'Before']
                        after_row = metric_data[metric_data['Category'] == 'After']
                        
                        if not before_row.empty and not after_row.empty:
                            before_mean = before_row['Mean'].iloc[0]
                            after_mean = after_row['Mean'].iloc[0]
                            
                            if before_mean != 0:
                                improvement = ((after_mean - before_mean) / before_mean) * 100
                            else:
                                improvement = np.inf if after_mean > 0 else 0
                            
                            improvement_data.append({
                                'Model': model,
                                'Metric': base_metric,
                                'Before_Mean': before_mean,
                                'After_Mean': after_mean,
                                'Improvement_%': improvement,
                                'Absolute_Change': after_mean - before_mean
                            })
                
                if improvement_data:
                    improvement_df = pd.DataFrame(improvement_data)
                    improvement_path = Path(output_dir) / f'{dataset}_improvement_analysis.csv'
                    improvement_df.to_csv(improvement_path, index=False)
        
        print(f"Saved detailed summary: {detailed_path}")

def main():
    """Enhanced main function with additional analysis."""
    # Import the original functions
    from generate_cdf_analysis import (
        load_json_file, extract_metrics_from_record, extract_all_metrics, 
        get_model_name_from_filename, plot_cdf, process_dataset
    )
    
    # Define paths
    base_path = Path('/home/runner/work/colab_scratchpad/colab_scratchpad')
    newset_path = base_path / 'Newset'
    output_dir = base_path / 'enhanced_cdf_analysis'
    
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
    
    # Create enhanced visualizations
    create_comparative_cdf_plots(datasets_metrics, str(output_dir))
    create_metric_correlation_heatmap(datasets_metrics, str(output_dir))
    create_before_after_comparison(datasets_metrics, str(output_dir))
    create_detailed_summary_table(datasets_metrics, str(output_dir))
    
    print(f"\nEnhanced analysis complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()