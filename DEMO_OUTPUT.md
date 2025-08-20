# FactCC Evaluation Demo Output

Below is a sample of the output format produced by the FactCC evaluation script:

## Individual Record Processing

```
Processing Record ID: 12396322
Input: Alex McKechnie, then 16, was in the crowd for that first show and went on to be ...
Gen Before Score: 0.142
Gen After Score: 0.330
Improvement: +0.188

Processing Record ID: 13906666
Input: Just days after dropping a supreme court action, the twins filed a fresh lawsuit...
Gen Before Score: 0.272
Gen After Score: 0.331
Improvement: +0.059

Processing Record ID: 18298438
Input: It is believed to be the first alleged breach of the Terrorism Prevention and In...
Gen Before Score: 0.321
Gen After Score: 0.417
Improvement: +0.097

[... continuing for all 10 records ...]
```

## Summary Statistics

```
============================================================
SUMMARY
============================================================
- Records with improvement: 8/10
- Average improvement: +0.079
- Best improvement: +0.223 (Record ID: 23606507)
- Worst change: -0.049 (Record ID: 21967739)
```

## Key Features Demonstrated

1. **Proper Score Format**: Scores are displayed with 3 decimal places for precision
2. **Improvement Calculation**: Shows the difference between gen_after and gen_before scores
3. **Input Text Truncation**: Long input texts are truncated for readability
4. **Summary Statistics**: Provides comprehensive analysis including:
   - Total number of records processed
   - Count of records showing improvement vs decline
   - Average improvement across all records
   - Best and worst performing records with their IDs

## Real vs Mock Implementation

- **Mock Implementation** (when no internet): Uses deterministic algorithms to simulate realistic FactCC scores
- **Real Implementation** (with internet): Downloads and uses the actual FactCC model from HuggingFace

The output format remains identical in both cases, ensuring consistent user experience regardless of the environment.