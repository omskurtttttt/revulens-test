# Test Set Evaluation & Reliability Analysis

**Winning Pipeline**: `distilbert-base-multilingual-cased` (C=0.5)

**Baseline Pipeline**: `TF-IDF (1-2 grams) + LinearSVC` (C=0.01)

### Validation vs Test Performance Comparison

| Pipeline | Split | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier Score | ECE (10 bins) | Mean Conf (Correct) | Mean Conf (Incorrect) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | Validation | 0.9317 | 0.9338 | 0.9291 | **0.9314** | 0.0525 | 0.0252 | 92.22% | 70.20% |
| `TF-IDF + LinearSVC` | Test | 0.9265 | 0.9249 | 0.9281 | **0.9265** | 0.0527 | 0.0210 | 92.65% | 69.77% |
| `distilbert-base-multilingual-cased` | Validation | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 94.46% | 71.46% |
| `distilbert-base-multilingual-cased` | Test | 0.9416 | 0.9442 | 0.9386 | **0.9414** | 0.0439 | 0.0112 | 94.44% | 72.86% |

### Generalization Findings

- Selected pipeline `distilbert-base-multilingual-cased` achieved **F1(CG) = 0.9414** on the unseen test set (compared to **0.9469** on validation, delta: **-0.0055**).
- Baseline TF-IDF achieved **F1(CG) = 0.9265** on the test set.
- Test Brier score is **0.0439** and ECE is **0.0112**, confirming consistent probability calibration on unseen data.
- Reliability diagrams saved to `results\reliability_diagram.png`.
