# Validation Evaluation Summary & Model Selection

**Winning Pipeline**: `distilbert-base-multilingual-cased`  
**Decision Rationale**: Selected distilbert-base-multilingual-cased because it achieved the highest Validation F1(CG) = 0.9469.  
**Seed Stability F1 (seeds 42, 7, 123)**: Mean = 0.9469, Std = 0.0000

### Validation Performance Comparison Table

| Pipeline | Best C | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier Score | ECE (10 bins) | Speed (rev/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | 0.01 | 0.9317 | 0.9338 | 0.9291 | **0.9314** | 0.0525 | 0.0252 | 5000.0 |
| `distilbert-base-multilingual-cased` | 0.5 | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 13.9 |
