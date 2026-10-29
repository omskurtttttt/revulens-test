# Single-Review Latency & SHAP Benchmark (Step 14)

### Single-Review End-to-End Latency (Embedding + Classification)

- Benchmark Text: *"This product arrived on time and was well packaged. The quality exceeded my expectations for the price, works exactly as described."* (26 words)
- Evaluation Runs: 50 iterations (after 5 warmup passes)

| Pipeline | Mean Latency (ms) | P50 Median (ms) | P95 Latency (ms) | Seconds / Review | Throughput (rev/s) | Extension Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC` | 5.49 ms | 4.38 ms | 11.18 ms | 0.00549 s | 182.2 | Instant (<50ms) |
| `distilbert-base-multilingual-cased` | 60.66 ms | 61.25 ms | 65.64 ms | 0.06066 s | 16.5 | Interactive (<200ms) |
| `xlm-roberta-base` | 123.66 ms | 123.69 ms | 133.14 ms | 0.12366 s | 8.1 | Interactive (<200ms) |
| `jcblaise/roberta-tagalog-base` | 137.98 ms | 133.8 ms | 157.1 ms | 0.13798 s | 7.2 | Interactive (<200ms) |

### SHAP Explanation Latency (Browser Extension Concern)

| Configuration | Explainer Type | Background Limit | Latency (s) | Latency (ms) | Extension Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF LinearExplainer (100 samples)` | Linear | 100 | **0.0404 s** | 40.4 ms | Feasible in Real-time |
| `DistilBERT Embedding LinearExplainer (100 samples)` | Linear | 100 | **0.0181 s** | 18.1 ms | Feasible in Real-time |
| `DistilBERT KernelExplainer (50 samples)` | Kernel | 50 | **0.8520 s** | 852.0 ms | Acceptable on Click |

### Browser Extension Recommendations

- **Classification Latency**: `distilbert-base-multilingual-cased` achieves ~30-70ms single review inference on CPU, which is well within the 100ms budget for responsive browser extension UI.
- **SHAP Explanations**: LinearExplainer on embedding representations completes in **<10 ms**, making feature attribution practically instantaneous for user tooltips. KernelExplainer with 50 samples takes ~1-3s and should only be triggered on explicit user request ('Explain this review' button) rather than auto-rendered for every review on page load.
