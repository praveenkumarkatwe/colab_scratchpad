# Dataset Setup for EHI Hindi Summarization

This directory contains instructions and utilities for setting up Hindi summarization datasets for the EHI (Entity Hallucination Index) system.

## Supported Datasets

### 1. XL-Sum Hindi Dataset

The XL-Sum dataset is automatically downloaded from Hugging Face when using the `xlsum` dataset option.

**Usage:**
```bash
# Prepare XL-Sum Hindi dataset
ehi-prepare --dataset xlsum --lang hi --split all
```

**Dataset Details:**
- Source: [XL-Sum on Hugging Face](https://huggingface.co/datasets/csebuetnlp/xlsum)
- Language: Hindi (hindi)
- Splits: train, validation, test
- Content: News articles with abstractive summaries
- Size: ~40K training examples, 4K validation, 5K test

### 2. ILSUM Dataset

ILSUM (Indian Language Summarization) dataset requires manual setup.

**Setup Instructions:**

1. **Download ILSUM Dataset:**
   ```bash
   # Download from official source (placeholder - replace with actual URL)
   wget https://example.com/ilsum_hindi.zip
   unzip ilsum_hindi.zip -d data/raw/ilsum/
   ```

2. **Convert to Standard Format:**
   Place ILSUM files in `data/raw/ilsum/` with the following structure:
   ```
   data/raw/ilsum/
   ├── train.json
   ├── validation.json
   └── test.json
   ```

3. **Expected JSON Format:**
   ```json
   [
     {
       "article": "Source article text in Hindi...",
       "highlights": "Summary text in Hindi..."
     },
     ...
   ]
   ```

4. **Prepare Dataset:**
   ```bash
   ehi-prepare --dataset ilsum --lang hi --split all
   ```

### 3. Custom Datasets

You can use custom Hindi summarization datasets by placing them in the appropriate format.

**File Structure:**
```
data/raw/custom/
├── custom_train.json
├── custom_validation.json
└── custom_test.json
```

**JSON Format Options:**

**Option 1: List of Objects**
```json
[
  {
    "text": "Source document in Hindi...",
    "summary": "Target summary in Hindi..."
  },
  ...
]
```

**Option 2: Separate Arrays**
```json
{
  "text": ["Document 1...", "Document 2...", ...],
  "summary": ["Summary 1...", "Summary 2...", ...]
}
```

**Usage:**
```bash
ehi-prepare --dataset custom --lang hi --split all
```

## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. **Unicode Normalization:** Converts text to NFC form
2. **Tokenization:** Uses model-specific tokenizer (mT5/mBART)
3. **Length Filtering:** Filters examples based on length constraints
4. **Caching:** Saves processed data for faster loading

**Configuration:**
- `max_source_length`: Maximum source text length (default: 1024 tokens)
- `max_target_length`: Maximum summary length (default: 256 tokens)
- `num_proc`: Number of parallel processes (default: 4)

## Directory Structure

After processing, your data directory will look like:

```
data/
├── cache/                  # Raw dataset cache
│   ├── downloads/
│   └── extracted/
├── processed/              # Processed datasets
│   ├── xlsum_train_processed/
│   ├── xlsum_validation_processed/
│   └── xlsum_test_processed/
└── raw/                    # Raw datasets (for custom data)
    ├── ilsum/
    └── custom/
```

## Quality Guidelines

For best results with the EHI system, ensure your datasets have:

1. **High-Quality Hindi Text:**
   - Proper Devanagari script
   - Correct Unicode encoding
   - Minimal code-mixing

2. **Entity-Rich Content:**
   - Named entities (persons, organizations, locations)
   - Factual information
   - News articles work well

3. **Abstractive Summaries:**
   - Not just extractive concatenation
   - Proper grammatical structure
   - Relevant entity mentions

## Example Usage

```bash
# 1. Prepare XL-Sum dataset
ehi-prepare --dataset xlsum --lang hi --split train
ehi-prepare --dataset xlsum --lang hi --split validation
ehi-prepare --dataset xlsum --lang hi --split test

# 2. Train model on prepared data
ehi-train --config configs/ehi_rl_mt5_small.yaml

# 3. Evaluate on test set
ehi-evaluate --config configs/eval.yaml --checkpoint outputs/best_model

# 4. Generate summaries for new texts
echo "नई हिंदी खबर यहाँ लिखें..." | ehi-infer --checkpoint outputs/best_model
```

## Troubleshooting

**Common Issues:**

1. **Unicode Errors:**
   - Ensure files are saved with UTF-8 encoding
   - Check for proper Devanagari characters

2. **Memory Issues:**
   - Reduce `num_proc` parameter
   - Lower `max_source_length` and `max_target_length`

3. **Empty Datasets:**
   - Check file paths and formats
   - Verify JSON structure matches expected format

4. **Tokenizer Issues:**
   - Ensure model supports Hindi text
   - Check tokenizer compatibility

**Getting Help:**

- Check logs in `logs/` directory
- Use `--log-level DEBUG` for detailed output
- Verify dataset statistics after processing

## Dataset Statistics

After processing, you'll see statistics like:

```
Dataset Statistics:
  train:
    Examples: 35,000
    Avg source length: 456.2 tokens
    Avg target length: 64.8 tokens
  validation:
    Examples: 4,000
    Avg source length: 445.1 tokens
    Avg target length: 63.2 tokens
  test:
    Examples: 5,000
    Avg source length: 442.8 tokens
    Avg target length: 62.9 tokens
```

These statistics help ensure data quality and consistency across splits.