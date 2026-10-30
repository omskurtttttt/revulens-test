# Validation Evaluation Summary & Model Selection

**Winning Pipeline**: `distilbert-base-multilingual-cased`  
**Decision Rationale**: Selected via 1-point tie-break rule: an extended grid verified boundary behavior where `xlm-roberta-base` achieved optimal validation F1 of 0.9503 at C=0.01 (compared to 0.9469 for DistilBERT, delta +0.0034 < 0.01 threshold). `distilbert-base-multilingual-cased` was selected because it is more than 2× faster on CPU (60.66 ms vs 123.66 ms per review, ~61 ms vs ~124 ms; 16.5 vs 8.1 rev/s) and has approximately half the parameter footprint (verified at **134.7M vs. 278.0M parameters** on Hugging Face), comfortably satisfying the browser extension real-time latency budget (<100 ms).  

---

### Validation Performance Comparison Table (Extended Grid: [0.001, 0.003, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0])

| Pipeline | Best C | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier Score | ECE (10 bins) | Latency (ms) | Throughput (rev/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | 0.01 | 0.9317 | 0.9338 | 0.9291 | 0.9314 | 0.0525 | 0.0252 | 5.49 ms | 182.2 |
| `distilbert-base-multilingual-cased` | 0.50 | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 60.66 ms | 16.5 |
| `xlm-roberta-base` | 0.01 | 0.9502 | 0.9482 | 0.9524 | **0.9503** | 0.0383 | 0.0120 | 123.66 ms | 8.1 |
| `jcblaise/roberta-tagalog-base` | 0.001 | 0.9398 | 0.9400 | 0.9395 | 0.9398 | 0.0461 | 0.0152 | 137.98 ms | 7.2 |

*Note: In the extended grid, `jcblaise/roberta-tagalog-base` shifted from C=0.01 to C=0.001, improving F1 from 0.9375 to 0.9398 and Brier to 0.0461. `xlm-roberta-base` peaked at C=0.01 (F1=0.9508 uncalibrated / 0.9503 calibrated), outperforming C=0.003 (0.9500) and C=0.001 (0.9486).*

---

### Seed Stability Evaluation (Seeds 42, 7, 123)

Evaluated across three independent stratified splits of the development pool (36,351 reviews), strictly leaving the test holdout (4,040 reviews) untouched:

| Pipeline | Leak-Free Mean F1(CG) ± Std | Leak-Free Mean Brier ± Std | Fixed-C Mean F1(CG) ± Std (Mild Leakage) | Stability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `TF-IDF + LinearSVC` | **0.9351 ± 0.0010** | **0.0490 ± 0.0013** | 0.9330 ± 0.0013 | Highly stable baseline |
| `distilbert-base-multilingual-cased` | **0.9375 ± 0.0021** | **0.0464 ± 0.0013** | 0.9368 ± 0.0020 | Robust across partitions |
| `xlm-roberta-base` | **0.9460 ± 0.0035** | **0.0396 ± 0.0022** | 0.9465 ± 0.0041 | Strongest metric, consistent |
| `jcblaise/roberta-tagalog-base` | **0.9375 ± 0.0018** | **0.0486 ± 0.0018** | 0.9375 ± 0.0018 | Robust English transfer |

*Methodological Note: In the Leak-Free method, C is tuned strictly inside the training partition of each seed (via inner 15% validation split), eliminating optimistic leakage. Fixed-C scores are included for comparison.*
