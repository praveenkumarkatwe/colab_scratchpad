# Summary Generation Evaluation Results

This document presents a comprehensive comparison of different language models (FlanT5, Mistral, DistilBART) on summarization tasks across different datasets (XSUM, Dialogue).

## Model Performance Comparison

### Key Metrics Explanation
- **EHI**: Entity Hallucination Index (lower is better)
- **EF1**: Entity F1 Score (higher is better)
- **PH**: Pronoun Hallucination (lower is better) 
- **OF**: Over-generation Factor (lower is better)
- **LF**: Length Factor
- **NH**: Negation Hallucination (lower is better)
- **EF**: Entity Factor
- **ROUGE**: ROUGE scores (higher is better)

### Results Summary Table

| Model | Dataset | Records | EHI_before | EHI_after | EF1_before | EF1_after | ROUGE1_before | ROUGE1_after | ROUGE2_before | ROUGE2_after |
|-------|---------|---------|------------|-----------|------------|-----------|---------------|--------------|---------------|--------------|
| DistilBART | Dialogue | 187 | 0.185 | 0.366 | 0.080 | 0.075 | N/A | N/A | N/A | N/A |
| DistilBART | XSUM | 200 | 0.309 | 0.299 | 0.104 | 0.100 | N/A | N/A | N/A | N/A |
| DistilBART | XSUM-Newset | 200 | 0.309 | 0.299 | 0.104 | 0.100 | N/A | N/A | N/A | N/A |
| FlanT5 | Dialogue | 187 | 0.294 | 0.279 | 0.092 | 0.001 | 0.234 | 0.005 | 0.079 | 0.000 |
| FlanT5 | XSUM | 200 | 0.196 | 0.005 | 0.076 | 0.060 | 0.130 | 0.023 | 0.023 | 0.001 |
| FlanT5 | XSUM-Newset | 200 | 0.307 | 0.126 | 0.311 | 0.072 | 0.396 | 0.089 | 0.266 | 0.021 |
| Mistral | Dialogue | 200 | 0.144 | 0.341 | 0.075 | 0.062 | N/A | N/A | N/A | N/A |
| Mistral | XSUM | 200 | 0.144 | 0.141 | 0.075 | 0.062 | N/A | N/A | N/A | N/A |
| Mistral | XSUM-Newset | 200 | 0.144 | 0.141 | 0.075 | 0.062 | N/A | N/A | N/A | N/A |

### Detailed Metrics by Model

#### DistilBART - Dialogue (187 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.185 | 0.366 | -0.181 |
| EF1 | 0.080 | 0.075 | -0.005 |
| PH | 0.080 | 0.075 | +0.005 |
| OF | 0.081 | 0.088 | -0.006 |
| LF | 0.780 | 0.859 | +0.080 |
| NH | 0.297 | 0.300 | -0.003 |
| EF | 0.023 | 0.020 | -0.003 |

#### DistilBART - XSUM (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.309 | 0.299 | +0.011 |
| EF1 | 0.104 | 0.100 | -0.004 |
| PH | 0.104 | 0.100 | +0.004 |
| OF | 0.181 | 0.194 | -0.013 |
| LF | 0.152 | 0.139 | -0.014 |
| NH | 0.101 | 0.100 | +0.001 |
| EF | 0.030 | 0.031 | +0.001 |

#### DistilBART - XSUM-Newset (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.309 | 0.299 | +0.011 |
| EF1 | 0.104 | 0.100 | -0.004 |
| PH | 0.104 | 0.100 | +0.004 |
| OF | 0.181 | 0.194 | -0.013 |
| LF | 0.152 | 0.139 | -0.014 |
| NH | 0.101 | 0.100 | +0.001 |
| EF | 0.030 | 0.031 | +0.001 |

#### FlanT5 - Dialogue (187 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.294 | 0.279 | +0.015 |
| EF1 | 0.092 | 0.001 | -0.091 |
| PH | 0.092 | 0.001 | +0.091 |
| OF | 0.247 | 0.002 | +0.246 |
| LF | 0.399 | 0.294 | -0.105 |
| NH | 0.285 | 0.984 | -0.698 |
| EF | 0.072 | 0.000 | -0.072 |

#### FlanT5 - XSUM (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.196 | 0.005 | +0.191 |
| EF1 | 0.076 | 0.060 | -0.016 |
| PH | 0.054 | 0.000 | +0.054 |
| OF | 0.080 | 0.000 | +0.080 |
| LF | 0.274 | 0.010 | -0.264 |
| NH | 0.340 | 0.020 | +0.320 |
| EF | 0.020 | 0.000 | -0.020 |

#### FlanT5 - XSUM-Newset (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.307 | 0.126 | +0.180 |
| EF1 | 0.311 | 0.072 | -0.239 |
| PH | 0.295 | 0.030 | +0.265 |
| OF | 0.070 | 0.240 | -0.170 |
| LF | 0.343 | 0.024 | -0.319 |
| NH | 0.118 | 0.021 | +0.098 |
| EF | 0.048 | 0.018 | -0.030 |

#### Mistral - Dialogue (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.144 | 0.341 | -0.197 |
| EF1 | 0.075 | 0.062 | -0.012 |
| PH | 0.075 | 0.062 | +0.012 |
| OF | 0.087 | 0.076 | +0.011 |
| LF | 0.149 | 0.157 | +0.008 |
| NH | 0.521 | 0.500 | +0.021 |
| EF | 0.019 | 0.017 | -0.002 |

#### Mistral - XSUM (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.144 | 0.141 | +0.003 |
| EF1 | 0.075 | 0.062 | -0.012 |
| PH | 0.075 | 0.062 | +0.012 |
| OF | 0.087 | 0.076 | +0.011 |
| LF | 0.149 | 0.157 | +0.008 |
| NH | 0.521 | 0.500 | +0.021 |
| EF | 0.019 | 0.017 | -0.002 |

#### Mistral - XSUM-Newset (200 records)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| EHI | 0.144 | 0.141 | +0.003 |
| EF1 | 0.075 | 0.062 | -0.012 |
| PH | 0.075 | 0.062 | +0.012 |
| OF | 0.087 | 0.076 | +0.011 |
| LF | 0.149 | 0.157 | +0.008 |
| NH | 0.521 | 0.500 | +0.021 |
| EF | 0.019 | 0.017 | -0.002 |

## Summary

This evaluation compares the performance of three language models across summarization tasks. The "before" and "after" columns likely represent performance before and after some optimization or fine-tuning process.

### Key Observations:
- All models were evaluated on both XSUM news articles and dialogue datasets
- Metrics track various aspects of summary quality including factual accuracy, entity handling, and traditional evaluation measures
- The results can help identify which models perform best for different types of text summarization tasks

### Model Performance Highlights:

**FlanT5** shows significant improvement in reducing hallucination metrics (EHI) after optimization, particularly on XSUM datasets, but with some degradation in other metrics like EF1.

**Mistral** demonstrates consistent performance with smaller changes between before/after states, showing stability across different datasets.

**DistilBART** shows mixed results with some improvement in EHI on XSUM but degradation on dialogue tasks.

### Data Sources:
- **XSUM Dataset**: News article summarization (200 records)
- **Dialogue Dataset**: Conversation summarization (187-200 records)  
- **Models**: FlanT5, Mistral, DistilBART

### Files Analyzed:
- `XSUMDataset_FlanT5.json` / `XSUMDataset_Mistral.json` / `XSUMDataset_distilbart.json`
- `Dialogueset_FlanT5.json` / `Dialogueset_Mistral.json` / `Dialogueset_distilbart.json` 
- `Newset/XSUM/before_after_all_metrics_200XSUM_flant5.json`
- `Newset/XSUM/results_DISTILBART_XSUM_Dataset_200_rescore.json`
- `Newset/XSUM/rescore_mistral_XSUM_before_after_JSON_file.json`