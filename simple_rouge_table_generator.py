# Rouge Metrics Table Generator for XSUM Dataset Models
# Simple script to generate comparison table - perfect for Google Colab

# Note: In Colab, run this first: !pip install pandas numpy --quiet

import json
import numpy as np
import pandas as pd
from collections import Counter
import os

def compute_rouge_n(reference, candidate, n=1):
    """Compute ROUGE-N F1 score"""
    if not reference or not candidate:
        return 0.0
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0
    
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
    cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
    
    if not ref_ngrams or not cand_ngrams:
        return 0.0
    
    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)
    overlap = sum((ref_counter & cand_counter).values())
    
    precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0
    recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def compute_rouge_l(reference, candidate):
    """Compute ROUGE-L F1 score using LCS"""
    if not reference or not candidate:
        return 0.0
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

# File paths (upload these files to your Colab environment)
file_paths = {
    'FlanT5': 'before_after_all_metrics_200XSUM_flant5.json',
    'Mistral': 'rescore_mistral_XSUM_before_after_JSON_file.json',
    'DistilBART': 'results_DISTILBART_XSUM_Dataset_200_rescore.json'
}

print("🔍 Rouge Metrics Table Generator")
print("=" * 50)

results = {}

for model_name, file_path in file_paths.items():
    print(f"\n📊 Processing {model_name}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("   Please upload this file to your Colab environment")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} records")
    
    # For FlanT5, use precomputed Rouge metrics if available
    if model_name == 'FlanT5' and 'ROUGE1_before' in data[0]:
        print("📋 Using precomputed Rouge metrics")
        rouge1_before = np.mean([r.get('ROUGE1_before', 0) for r in data])
        rouge1_after = np.mean([r.get('ROUGE1_after', 0) for r in data])
        rouge2_before = np.mean([r.get('ROUGE2_before', 0) for r in data])
        rouge2_after = np.mean([r.get('ROUGE2_after', 0) for r in data])
        rougel_before = np.mean([r.get('ROUGEL_before', 0) for r in data])
        rougel_after = np.mean([r.get('ROUGEL_after', 0) for r in data])
    else:
        print("🧮 Computing Rouge metrics from text")
        rouge1_before, rouge1_after = [], []
        rouge2_before, rouge2_after = [], []
        rougel_before, rougel_after = [], []
        
        for record in data:
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
        
        rouge1_before = np.mean(rouge1_before) if rouge1_before else 0.0
        rouge1_after = np.mean(rouge1_after) if rouge1_after else 0.0
        rouge2_before = np.mean(rouge2_before) if rouge2_before else 0.0
        rouge2_after = np.mean(rouge2_after) if rouge2_after else 0.0
        rougel_before = np.mean(rougel_before) if rougel_before else 0.0
        rougel_after = np.mean(rougel_after) if rougel_after else 0.0
    
    results[model_name] = {
        'rouge1_before': rouge1_before,
        'rouge1_after': rouge1_after,
        'rouge2_before': rouge2_before,
        'rouge2_after': rouge2_after,
        'rougel_before': rougel_before,
        'rougel_after': rougel_after
    }

# Create the table
print("\n📊 ROUGE METRICS COMPARISON TABLE")
print("=" * 80)

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

df = pd.DataFrame(df_data)
print(df.to_string(index=False))

print("\n📋 Copy-Paste Ready Table:")
print("| Model     | Rouge-1 Before | Rouge-1 After | Rouge-2 Before | Rouge-2 After | RougeL Before | RougeL After |")
print("|-----------|----------------|---------------|----------------|---------------|---------------|--------------|")
for _, row in df.iterrows():
    print(f"| {row['Model']:<9} | {row['Rouge-1 Before']:<14} | {row['Rouge-1 After']:<13} | {row['Rouge-2 Before']:<14} | {row['Rouge-2 After']:<13} | {row['RougeL Before']:<13} | {row['RougeL After']:<12} |")

# Save to CSV
df.to_csv('rouge_metrics_comparison.csv', index=False)
print(f"\n💾 Results saved to: rouge_metrics_comparison.csv")
print("🎉 Analysis Complete!")