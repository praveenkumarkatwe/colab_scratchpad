#!/usr/bin/env python3
"""
Rouge Metrics Table Generator for XSUM Dataset Models

This script generates a comparison table of Rouge-1, Rouge-2, and Rouge-L metrics
for three models (FlanT5, Mistral, DistilBART) before and after fine-tuning.

Usage in Google Colab:
1. Upload this script to Colab
2. Upload the JSON files from Newset/XSUM/ directory
3. Run the script to generate the comparison table

Author: Generated for colab_scratchpad repository
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
import re
from collections import Counter

def compute_rouge_n(reference: str, candidate: str, n: int = 1) -> float:
    """
    Compute ROUGE-N score between reference and candidate text.
    
    Args:
        reference: Reference/target text
        candidate: Generated/candidate text
        n: N-gram size (1 for ROUGE-1, 2 for ROUGE-2)
    
    Returns:
        ROUGE-N F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    # Tokenize and convert to lowercase
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0
    
    # Generate n-grams
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
    cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
    
    if not ref_ngrams or not cand_ngrams:
        return 0.0
    
    # Count n-grams
    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)
    
    # Calculate overlap
    overlap = sum((ref_counter & cand_counter).values())
    
    # Calculate precision and recall
    precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0
    recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L score using Longest Common Subsequence.
    
    Args:
        reference: Reference/target text
        candidate: Generated/candidate text
    
    Returns:
        ROUGE-L F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Compute LCS length using dynamic programming
    def lcs_length(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(ref_tokens, cand_tokens)
    
    if lcs_len == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def extract_flant5_rouge_metrics(data: List[Dict]) -> Dict[str, float]:
    """
    Extract Rouge metrics from FlanT5 data which already contains computed Rouge scores.
    
    Args:
        data: List of records from FlanT5 JSON file
    
    Returns:
        Dictionary with average Rouge metrics
    """
    rouge1_before = []
    rouge1_after = []
    rouge2_before = []
    rouge2_after = []
    rougel_before = []
    rougel_after = []
    
    for record in data:
        rouge1_before.append(record.get('ROUGE1_before', 0.0))
        rouge1_after.append(record.get('ROUGE1_after', 0.0))
        rouge2_before.append(record.get('ROUGE2_before', 0.0))
        rouge2_after.append(record.get('ROUGE2_after', 0.0))
        rougel_before.append(record.get('ROUGEL_before', 0.0))
        rougel_after.append(record.get('ROUGEL_after', 0.0))
    
    return {
        'rouge1_before': np.mean(rouge1_before),
        'rouge1_after': np.mean(rouge1_after),
        'rouge2_before': np.mean(rouge2_before),
        'rouge2_after': np.mean(rouge2_after),
        'rougel_before': np.mean(rougel_before),
        'rougel_after': np.mean(rougel_after)
    }

def compute_rouge_metrics_from_text(data: List[Dict]) -> Dict[str, float]:
    """
    Compute Rouge metrics from text data for models that don't have precomputed Rouge scores.
    
    Args:
        data: List of records with reference and generated summaries
    
    Returns:
        Dictionary with average Rouge metrics
    """
    rouge1_before = []
    rouge1_after = []
    rouge2_before = []
    rouge2_after = []
    rougel_before = []
    rougel_after = []
    
    for record in data:
        # Get reference and generated summaries
        reference = record.get('reference_summary', record.get('reference', ''))
        gen_before = record.get('gen_before', record.get('generatedsummary_before', ''))
        gen_after = record.get('gen_after', record.get('generatedsummary_after', ''))
        
        if reference and gen_before:
            rouge1_before.append(compute_rouge_n(reference, gen_before, 1))
            rouge2_before.append(compute_rouge_n(reference, gen_before, 2))
            rougel_before.append(compute_rouge_l(reference, gen_before))
        
        if reference and gen_after:
            rouge1_after.append(compute_rouge_n(reference, gen_after, 1))
            rouge2_after.append(compute_rouge_n(reference, gen_after, 2))
            rougel_after.append(compute_rouge_l(reference, gen_after))
    
    return {
        'rouge1_before': np.mean(rouge1_before) if rouge1_before else 0.0,
        'rouge1_after': np.mean(rouge1_after) if rouge1_after else 0.0,
        'rouge2_before': np.mean(rouge2_before) if rouge2_before else 0.0,
        'rouge2_after': np.mean(rouge2_after) if rouge2_after else 0.0,
        'rougel_before': np.mean(rougel_before) if rougel_before else 0.0,
        'rougel_after': np.mean(rougel_after) if rougel_after else 0.0
    }

def load_and_process_files(file_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Load JSON files and compute Rouge metrics for all models.
    
    Args:
        file_paths: Dictionary mapping model names to file paths
    
    Returns:
        pandas DataFrame with Rouge metrics comparison table
    """
    results = {}
    
    for model_name, file_path in file_paths.items():
        print(f"Processing {model_name} data from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  Loaded {len(data)} records for {model_name}")
            
            # Check if this is FlanT5 with precomputed Rouge metrics
            if model_name.lower() == 'flant5' and 'ROUGE1_before' in data[0]:
                print(f"  Using precomputed Rouge metrics for {model_name}")
                metrics = extract_flant5_rouge_metrics(data)
            else:
                print(f"  Computing Rouge metrics from text for {model_name}")
                metrics = compute_rouge_metrics_from_text(data)
            
            results[model_name] = metrics
            print(f"  Completed processing {model_name}")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            # Fill with zeros if file can't be processed
            results[model_name] = {
                'rouge1_before': 0.0, 'rouge1_after': 0.0,
                'rouge2_before': 0.0, 'rouge2_after': 0.0,
                'rougel_before': 0.0, 'rougel_after': 0.0
            }
    
    # Create DataFrame
    df_data = []
    for model_name, metrics in results.items():
        df_data.append({
            'Model': model_name,
            'Rouge-1 Before': f"{metrics['rouge1_before']:.4f}",
            'Rouge-1 After': f"{metrics['rouge1_after']:.4f}",
            'Rouge-2 Before': f"{metrics['rouge2_before']:.4f}",
            'Rouge-2 After': f"{metrics['rouge2_after']:.4f}",
            'RougeL Before': f"{metrics['rougel_before']:.4f}",
            'RougeL After': f"{metrics['rougel_after']:.4f}"
        })
    
    return pd.DataFrame(df_data)

def main():
    """
    Main function to generate Rouge metrics comparison table.
    """
    print("🔍 Rouge Metrics Table Generator for XSUM Dataset Models")
    print("=" * 60)
    
    # Define file paths - adjust these paths based on your Colab environment
    file_paths = {
        'FlanT5': 'before_after_all_metrics_200XSUM_flant5.json',
        'Mistral': 'rescore_mistral_XSUM_before_after_JSON_file.json',
        'DistilBART': 'results_DISTILBART_XSUM_Dataset_200_rescore.json'
    }
    
    # Check if files exist and update paths if needed
    base_path = 'Newset/XSUM/'
    updated_paths = {}
    
    for model, filename in file_paths.items():
        # Try multiple possible locations
        possible_paths = [
            filename,  # Current directory
            os.path.join(base_path, filename),  # Newset/XSUM subdirectory
            os.path.join('files', base_path, filename)  # files/Newset/XSUM subdirectory
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                updated_paths[model] = path
                print(f"✅ Found {model} data at: {path}")
                found = True
                break
        
        if not found:
            print(f"❌ Could not find {model} data. Tried paths:")
            for path in possible_paths:
                print(f"   - {path}")
            updated_paths[model] = filename  # Use original path anyway
    
    print("\n🔄 Processing files...")
    print("-" * 30)
    
    # Generate comparison table
    df = load_and_process_files(updated_paths)
    
    print("\n📊 Rouge Metrics Comparison Table")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Also create a more formatted version
    print("\n📋 Formatted Table for Copy-Paste:")
    print("-" * 50)
    
    # Print headers
    print("| Model     | Rouge-1 Before | Rouge-1 After | Rouge-2 Before | Rouge-2 After | RougeL Before | RougeL After |")
    print("|-----------|----------------|---------------|----------------|---------------|---------------|--------------|")
    
    # Print data rows
    for _, row in df.iterrows():
        print(f"| {row['Model']:<9} | {row['Rouge-1 Before']:<14} | {row['Rouge-1 After']:<13} | {row['Rouge-2 Before']:<14} | {row['Rouge-2 After']:<13} | {row['RougeL Before']:<13} | {row['RougeL After']:<12} |")
    
    print("\n✅ Analysis complete!")
    
    # Save results to CSV for easy access
    csv_filename = 'rouge_metrics_comparison.csv'
    df.to_csv(csv_filename, index=False)
    print(f"💾 Results saved to: {csv_filename}")
    
    return df

if __name__ == "__main__":
    # Install required packages if running in Colab
    try:
        import pandas
        import numpy
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install pandas numpy")
        import pandas as pd
        import numpy as np
    
    # Run the main function
    df = main()