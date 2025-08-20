# FactCC Evaluation Script

This repository contains Python scripts to compute FactCC scores for evaluating whether `gen_after` shows improvements over `gen_before` when compared against the `input` text.

## Files

- `factcc_evaluation.py` - Main evaluation script
- `factcc_evaluation_colab.py` - Google Colab version with automatic dependency installation
- `Dialogueset_Mistral.json` - Input data file containing records with input, gen_before, and gen_after fields

## Features

1. **Data Processing**: Loads the first 10 records from `Dialogueset_Mistral.json`
2. **FactCC Evaluation**: Uses the FactCC model (manueldeprada/FactCC) to compute consistency scores
3. **Comparison Logic**: For each record, evaluates:
   - `input` vs `gen_before` (baseline score)
   - `input` vs `gen_after` (improved score)
4. **Results Analysis**: 
   - Displays individual scores for each record
   - Calculates improvement metrics (difference, percentage improvement)
   - Provides summary statistics showing overall improvement

## Usage

### Option 1: Local Environment

```bash
# Install dependencies
pip install transformers torch

# Run the script
python factcc_evaluation.py
```

### Option 2: Google Colab

1. Upload `factcc_evaluation_colab.py` and `Dialogueset_Mistral.json` to your Colab environment
2. Run the script - it will automatically install dependencies
3. View results with real FactCC scores

### Option 3: Offline/Demo Mode

If internet access is not available, the script will automatically fall back to a mock implementation that demonstrates the functionality with realistic-looking scores.

## Expected Output Format

```
Processing Record ID: 12396322
Input: [truncated input text]
Gen Before Score: 0.142
Gen After Score: 0.330
Improvement: +0.188

[... for 10 records ...]

Summary:
- Records with improvement: 8/10
- Average improvement: +0.079
- Best improvement: +0.223 (Record ID: 23606507)
- Worst change: -0.049 (Record ID: 21967739)
```

## Technical Details

- **Model**: Uses the HuggingFace model `manueldeprada/FactCC`
- **Input Format**: FactCC expects document-claim pairs, structured as: document=`input`, claim=`gen_before`/`gen_after`
- **Scoring**: Returns consistency probability scores between 0 and 1
- **Device Support**: Automatically uses GPU if available, falls back to CPU
- **Error Handling**: Gracefully handles missing data and model loading errors

## Requirements

- Python 3.7+
- transformers library
- torch library
- Internet access (for downloading the FactCC model)
- GPU recommended for faster processing

## Mock Implementation

When internet access is not available, the script uses a mock implementation that:
- Generates realistic scores based on text similarity and length ratios
- Simulates improvements for "after" versions
- Provides consistent results for demonstration purposes
- Clearly indicates when mock scores are being used

This allows you to test the script structure and see the expected output format even without model access.