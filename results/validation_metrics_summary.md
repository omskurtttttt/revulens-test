# Validation Evaluation Summary & Model Selection

**Winning Pipeline**: `distilbert-base-multilingual-cased`  
**Decision Rationale**: Selected via 1-point tie-break rule: `xlm-roberta-base` achieved marginally higher validation F1 (0.9491 vs 0.9469, delta +0.0022 < 0.01 threshold), but `distilbert-base-multilingual-cased` is more than 2× faster (60.66 ms vs 123.66 ms per review, 16.5 vs 8.1 rev/s) and has 50% smaller model footprint, comfortably satisfying the browser extension real-time latency budget (<100 ms).  

---

### Validation Performance Comparison Table

| Pipeline | Best C | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier Score | ECE (10 bins) | Latency (ms) | Throughput (rev/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | 0.01 | 0.9317 | 0.9338 | 0.9291 | 0.9314 | 0.0525 | 0.0252 | 5.49 ms | 182.2 |
| `distilbert-base-multilingual-cased` | 0.50 | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 60.66 ms | 16.5 |
| `xlm-roberta-base` | 0.01 | 0.9490 | 0.9463 | 0.9519 | **0.9491** | 0.0387 | 0.0134 | 123.66 ms | 8.1 |
| `jcblaise/roberta-tagalog-base` | 0.01 | 0.9374 | 0.9349 | 0.9400 | 0.9375 | 0.0469 | 0.0162 | 137.98 ms | 7.2 |

*Note: All speed and throughput metrics are sourced directly from the single-review CPU benchmark in `results/speed_benchmark.json`.*

---

### Seed Stability Evaluation (Seeds 42, 7, 123)

Evaluated across three independent stratified 80/10/10 splits of the development pool (36,351 reviews), strictly leaving the test holdout (4,040 reviews) untouched:

| Pipeline | Fixed Best C | Seed 42 F1 | Seed 7 F1 | Seed 123 F1 | Mean F1(CG) ± Std | Mean Brier ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | 0.01 | 0.9349 | 0.9319 | 0.9321 | 0.9330 ± 0.0013 | 0.0509 ± 0.0011 |
| `distilbert-base-multilingual-cased` | 0.50 | 0.9340 | 0.9379 | 0.9384 | 0.9368 ± 0.0020 | 0.0469 ± 0.0013 |
| `xlm-roberta-base` | 0.01 | 0.9416 | 0.9462 | 0.9516 | 0.9465 ± 0.0041 | 0.0394 ± 0.0024 |
| `jcblaise/roberta-tagalog-base` | 0.01 | 0.9351 | 0.9382 | 0.9394 | 0.9375 ± 0.0018 | 0.0488 ± 0.0015 |
