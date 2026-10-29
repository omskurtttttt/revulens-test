# FiReCS Out-of-Domain False Alarm Analysis (Step 12)

### Presumed-Genuine Evaluation on Filipino / Taglish Reviews

- **Total FiReCS Reviews Analyzed**: 10,481 unique reviews
- **Selected Pipeline**: `distilbert-base-multilingual-cased`
- **Baseline Pipeline**: `TF-IDF + LinearSVC`

### False Alarm & Confidence Summary

| Metric | `TF-IDF + LinearSVC` | `distilbert-base-multilingual-cased` |
| :--- | :---: | :---: |
| **Predicted CG Count (False Alarms)** | 1,118 | **6,760** |
| **False Alarm Rate (%)** | 10.67% | **64.5%** |
| **Presumed Genuine Rate (%)** | 89.33% | **35.5%** |
| **Mean Confidence** | 85.39% | **88.31%** |
| **Median Confidence** | 90.65% | **95.85%** |
| **Confidence IQR (Q25 - Q75)** | 76.4% - 97.3% | **79.9% - 99.6%** |

### Confidence Buckets

| Confidence Bucket | `TF-IDF + LinearSVC` | `distilbert-base-multilingual-cased` |
| :--- | :---: | :---: |
| High Confidence (≥ 80%) | 7,333 (70.0%) | 7,851 (74.9%) |
| Moderate Confidence (60% - 80%) | 2,201 (21.0%) | 1,828 (17.4%) |
| Low / Borderline Confidence (50% - 60%) | 947 (9.0%) | 802 (7.7%) |

### Methodological Caveats (Required by GEMINI.md)

> [!IMPORTANT]
> FiReCS has NO ground-truth Genuine vs CG labels; it is treated strictly as presumed-genuine.

> [!IMPORTANT]
> Accuracy, precision, recall, and F1 CANNOT be computed on FiReCS.

> [!IMPORTANT]
> Both models were trained strictly on English reviews (Salminen et al.).

> [!IMPORTANT]
> An elevated false-alarm (CG) rate on Taglish/Filipino is expected due to cross-lingual domain shift and does not indicate model invalidity on English reviews.

