# FactCC and QAGS Score Calculator

This repository contains tools to calculate FactCC and QAGS scores for summarization evaluation.

## Overview

- **FactCC** - Measures factual consistency between source text and summary using a sequence classifier
- **QAGS** - Question Answering and Generation for Summarization - measures faithfulness by generating questions from the summary and comparing answers from source vs. summary

## Files

- `compute_factcc_qags_scores.py` - Main script to calculate scores
- `factcc_N_QAGS_Scorer.ipynb` - Original Jupyter notebook implementation
- JSON files (`*_FlanT5.json`, `*_Mistral.json`, `*_distilbart.json`) - Dataset files with summaries

## Usage

### Prerequisites

```bash
pip install transformers torch sentencepiece
```

### Basic Usage

```bash
# Process a single JSON file
python compute_factcc_qags_scores.py --input_path input.json --output_path output.json

# Process all JSON files in the repository
python compute_factcc_qags_scores.py --process_all

# Use CPU only (useful for testing without GPU)
python compute_factcc_qags_scores.py --input_path input.json --device cpu
```

### Advanced Options

```bash
python compute_factcc_qags_scores.py \
  --input_path data.json \
  --output_path scored_data.json \
  --factcc_model "tals/albert-base-v2-factcc" \
  --qg_model "valhalla/t5-small-qg-hl" \
  --qa_model "deepset/roberta-base-squad2" \
  --device auto \
  --max_source_tokens 448 \
  --max_summary_tokens 192 \
  --qags_max_questions 5
```

## Input JSON Format

The script supports flexible JSON input formats. It looks for these key patterns:

**Source text** (one of):
- `input`
- `text` 
- `source`

**Reference summary** (optional):
- `reference`
- `reference_summary`

**For before/after comparison**:
- Before: `generatedsummary_before`, `generated_before`, `summary_before`, `generated_summary`
- After: `generatedsummary_after`, `generated_after`, `summary_after`, `generated_finetuned`

**For single summary**:
- `summary`, `generated_summary`, `output`

### Example Input

```json
[
  {
    "id": "example_1",
    "input": "Source document text here...",
    "reference": "Reference summary",
    "generatedsummary_before": "Summary before fine-tuning",
    "generatedsummary_after": "Summary after fine-tuning"
  }
]
```

## Output

The script adds the following fields to each record:

**For before/after data**:
- `factcc_before` - FactCC score for before summary
- `factcc_after` - FactCC score for after summary  
- `qags_before` - QAGS score for before summary
- `qags_after` - QAGS score for after summary

**For single summary data**:
- `factcc_score` - FactCC score for the summary
- `qags_score` - QAGS score for the summary

## Score Interpretation

- **FactCC scores**: Range 0-1, higher is better (more factually consistent)
- **QAGS scores**: Range 0-1, higher is better (more faithful to source)

## Repository JSON Files

The repository contains several JSON files with summarization data:

1. `Dialogueset_FlanT5.json` - Dialogue summarization with FlanT5
2. `Dialogueset_Mistral.json` - Dialogue summarization with Mistral
3. `Dialogueset_distilbart.json` - Dialogue summarization with DistilBART
4. `XSUMDataset_FlanT5.json` - XSUM dataset with FlanT5
5. `XSUMDataset_Mistral.json` - XSUM dataset with Mistral
6. `XSUMDataset_distilbart.json` - XSUM dataset with DistilBART

Each file contains before/after summaries with various evaluation metrics already computed.

## Processing All Repository Files

To calculate FactCC and QAGS scores for all JSON files:

```bash
python compute_factcc_qags_scores.py --process_all
```

This will create `*_scored.json` files with the additional scores.

## Model Information

- **FactCC Model**: `tals/albert-base-v2-factcc` - ALBERT-based binary classifier for factual consistency
- **Question Generation**: `valhalla/t5-small-qg-hl` - T5 model fine-tuned for question generation with highlighting
- **Question Answering**: `deepset/roberta-base-squad2` - RoBERTa model fine-tuned on SQuAD 2.0

## Requirements

- Python 3.7+
- transformers
- torch 
- sentencepiece (for T5 models)
- Internet connection (for downloading models on first run)

## Troubleshooting

1. **Connection errors**: Ensure internet connection for downloading models
2. **Memory issues**: Use `--device cpu` for CPU-only processing
3. **Model download issues**: Models are cached after first download to `~/.cache/huggingface/`

## Citation

If you use this code, please cite the relevant papers:

- **FactCC**: Kryściński et al. "Evaluating the Factual Consistency of Abstractive Text Summarization"
- **QAGS**: Wang et al. "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries"