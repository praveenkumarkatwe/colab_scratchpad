# CDF Analysis for XSUM and Dialogset Datasets

This analysis generates Cumulative Distribution Function (CDF) graphs for each model in the XSUM dataset per metric, and processes JSON files from the Dialogset directory to create comprehensive visualizations and statistical tables.

## Analysis Overview

### Datasets Processed
1. **XSUM Dataset** - 3 models:
   - FlanT5 (200 samples)
   - DistilBART (200 samples) 
   - Mistral (200 samples)

2. **Dialogset Dataset** - 3 models:
   - Mistral (200 samples)
   - DistilBART (187 samples)
   - FlanT5 (106 samples)

### Metrics Analyzed
The analysis covers multiple evaluation metrics in both "before" and "after" fine-tuning scenarios:

- **EHI** (Entity Hallucination Index)
- **EF1** (Entity F1 Score)
- **PH** (Person Hallucination)
- **OF** (Organization/Object Features)
- **LF** (Location Features)
- **NH** (Noun Hallucination)
- **EF** (Entity Features)
- **ROUGE1, ROUGE2, ROUGEL** (ROUGE scores)
- **EHI_CAL, EHI_EXP, EHI_W** (Extended EHI variants)

## Output Structure

### Individual CDF Plots
- `enhanced_cdf_analysis/{Dataset}/{Model}/{Metric}_cdf.png`
- Individual CDF plots for each metric per model per dataset

### Comparative Analysis
- `comparative_plots/{Dataset}/{Metric}_comparison_cdf.png`
- Side-by-side CDF comparisons across all models for each metric

### Correlation Analysis
- `correlation_heatmaps/{Dataset}_{Model}_correlations.png`
- Correlation heatmaps showing relationships between metrics within each model

### Before/After Comparisons
- `before_after_comparisons/{Dataset}_{Model}_before_after.png`
- Direct comparisons of before vs after fine-tuning performance

### Statistical Tables
- `detailed_summary_statistics.csv` - Comprehensive statistics for all metrics
- `{Dataset}_mean_comparison.csv` - Mean values comparison across models
- `{Dataset}_improvement_analysis.csv` - Before/after improvement analysis

## Key Findings

### XSUM Dataset Performance

**FlanT5 Model:**
- Shows significant performance drops after fine-tuning across most metrics
- ROUGE scores: 77-92% decrease (ROUGE1: -77.5%, ROUGE2: -92.0%)
- Hallucination metrics: Mixed results with some improvements in entity handling

**DistilBART Model:**
- Most stable performance with minimal changes after fine-tuning
- EHI scores: Minor 3.5% decrease
- Generally maintains baseline performance levels

**Mistral Model:**
- Moderate performance changes
- Some metrics show improvement, others show degradation
- EHI and related metrics show 1-16% changes

### Dialogset Dataset Performance

**Model Comparison:**
- FlanT5: Shows varying performance with some significant drops in ROUGE scores
- DistilBART: Consistent performance similar to XSUM results
- Mistral: Generally stable with minor fluctuations

## Usage

### Running the Analysis

1. **Basic Analysis:**
```bash
python generate_cdf_analysis.py
```

2. **Enhanced Analysis with Comparisons:**
```bash
python enhanced_analysis.py
```

### Requirements
- pandas
- matplotlib
- seaborn
- numpy

## Files Description

### Analysis Scripts
- `generate_cdf_analysis.py` - Main analysis script
- `enhanced_analysis.py` - Extended analysis with comparative visualizations

### Data Sources
- `Newset/XSUM/` - XSUM dataset JSON files
- `Newset/Dialogset/` - Dialogset JSON files

### Output Directories
- `cdf_analysis_output/` - Basic analysis results
- `enhanced_cdf_analysis/` - Enhanced analysis with comparisons

## Statistical Summary

The analysis reveals that:
1. **FlanT5** tends to show the most dramatic changes (often decreases) after fine-tuning
2. **DistilBART** maintains the most consistent performance
3. **Mistral** shows moderate and varied changes across different metrics
4. Hallucination-related metrics (EHI, EF1, PH) generally show improvement trends
5. ROUGE scores often decrease significantly after fine-tuning, suggesting a trade-off between factual accuracy and text similarity

## Interpretation Guidelines

- **Lower values** for hallucination metrics (EHI, PH, NH) indicate better performance
- **Higher values** for ROUGE scores indicate better text similarity
- **CDF curves** closer to the top-left indicate better performance for most metrics
- **Before/After comparisons** show the impact of fine-tuning on each model