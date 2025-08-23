# Rouge Metrics Table Generator

This repository contains tools to generate comparison tables of Rouge-1, Rouge-2, and Rouge-L metrics for three models (FlanT5, Mistral, DistilBART) before and after fine-tuning on the XSUM dataset.

## 📊 Generated Results

The analysis produces a comparison table with the following columns:
- **Rouge-1 Before**: Rouge-1 scores before fine-tuning
- **Rouge-1 After**: Rouge-1 scores after fine-tuning  
- **Rouge-2 Before**: Rouge-2 scores before fine-tuning
- **Rouge-2 After**: Rouge-2 scores after fine-tuning
- **RougeL Before**: Rouge-L scores before fine-tuning
- **RougeL After**: Rouge-L scores after fine-tuning

## 📁 Required Files

You need these three JSON files from `Newset/XSUM/` directory:

1. `before_after_all_metrics_200XSUM_flant5.json` (FlanT5 data)
2. `rescore_mistral_XSUM_before_after_JSON_file.json` (Mistral data)  
3. `results_DISTILBART_XSUM_Dataset_200_rescore.json` (DistilBART data)

## 🚀 Usage Options

### Option 1: Google Colab (Recommended)

1. **Upload the Jupyter notebook**: Upload `rouge_metrics_table_generator.ipynb` to Google Colab
2. **Upload JSON files**: Upload the three required JSON files to your Colab environment
3. **Run all cells**: Execute all cells in the notebook to generate the table

### Option 2: Simple Python Script for Colab

1. **Create a new cell in Colab** and copy-paste the content from `simple_rouge_table_generator.py`
2. **First run**: `!pip install pandas numpy --quiet`
3. **Upload JSON files** using the Files panel on the left
4. **Run the script** to generate the table

### Option 3: Local Environment

1. **Install dependencies**: `pip install pandas numpy`
2. **Place JSON files** in the same directory as the script
3. **Run**: `python rouge_metrics_table_generator.py`

## 📋 Sample Output

```
📊 ROUGE METRICS COMPARISON TABLE
================================================================================
     Model Rouge-1 Before Rouge-1 After Rouge-2 Before Rouge-2 After RougeL Before RougeL After
    FlanT5         0.3960        0.0889         0.2658        0.0213        0.3556       0.0693
   Mistral         0.1246        0.1253         0.0156        0.0170        0.0845       0.0875
DistilBART         0.1711        0.1686         0.0242        0.0214        0.1190       0.1179
```

## 🔧 How It Works

1. **FlanT5**: Uses precomputed Rouge metrics directly from the JSON file
2. **Mistral & DistilBART**: Computes Rouge metrics from text using:
   - Reference summaries (`reference_summary` field)
   - Generated summaries before fine-tuning (`gen_before` field)
   - Generated summaries after fine-tuning (`gen_after` field)

## 📊 Rouge Metrics Explained

- **Rouge-1**: Measures unigram (single word) overlap between generated and reference summaries
- **Rouge-2**: Measures bigram (two consecutive words) overlap
- **Rouge-L**: Measures longest common subsequence overlap

Higher scores indicate better overlap with reference summaries and generally better performance.

## 📁 Files Included

- `rouge_metrics_table_generator.py`: Complete Python script with detailed functions
- `rouge_metrics_table_generator.ipynb`: Jupyter notebook optimized for Google Colab
- `simple_rouge_table_generator.py`: Simplified script for easy copy-paste
- `rouge_metrics_comparison.csv`: Output CSV file with results

## 🎯 Key Features

- ✅ Automatic detection of precomputed Rouge metrics (FlanT5)
- ✅ On-the-fly Rouge computation for models without precomputed metrics
- ✅ Handles missing files gracefully
- ✅ Outputs both formatted table and CSV file
- ✅ Optimized for Google Colab environment
- ✅ Clear progress indicators and error handling

## 📝 Notes

- The scripts automatically detect whether Rouge metrics are precomputed or need to be calculated
- Results are saved to `rouge_metrics_comparison.csv` for further analysis
- All scripts include error handling for missing files or malformed data
- The table format matches the requirements specified in the problem statement