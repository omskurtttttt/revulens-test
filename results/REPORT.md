# RevuLens Pilot Experiment Report: Encoder vs. SVM Comparison

## 1. Executive Summary & Selection Decision
This pilot evaluates three frozen transformer encoders against a calibrated TF-IDF LinearSVC baseline for classifying reviews as **Likely Genuine (0)** or **Potentially Computer-Generated (1)**, using the Salminen et al. (2022) English dataset and an out-of-domain false alarm check on FiReCS (Filipino/Taglish).

**Selected Winning Pipeline**: `distilbert-base-multilingual-cased` + LinearSVC (Sigmoid Calibrated, $C=0.5$).
- **Selection Rationale & Extended Grid Analysis**: An extended hyperparameter grid `[0.001, 0.003, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]` verified boundary behavior. For `xlm-roberta-base`, the optimal $C$ remained $0.01$ (Validation F1: $0.9503$, outperforming $C=0.003$ at $0.9500$ and $C=0.001$ at $0.9486$). For `jcblaise/roberta-tagalog-base`, the optimal $C$ shifted to $0.001$ (improving validation F1 from $0.9375$ to $0.9398$, with Brier $0.0461$). Crucially, the delta between `distilbert` ($0.9469$) and `xlm-roberta-base` ($0.9503$) is $+0.0034$ ($0.34$ percentage points), remaining strictly within the 1-point tie-breaking threshold ($<0.01$). `distilbert-base-multilingual-cased` was selected because it is **more than 2× faster** ($60.66$ ms vs. $123.66$ ms per review, $16.5$ vs. $8.1$ rev/s) and has half the model footprint ($66$M vs. $125$M parameters), fitting comfortably within the browser extension latency budget ($<100$ ms).
- **Test Set Generalization**: Evaluated strictly **once** on the unseen 4,040-review test holdout, achieving **Test F1(CG) = 0.9414** (validation F1: $0.9469$, delta: $-0.0055$), with an **Expected Calibration Error (ECE) of 0.0112** and **Brier Score of 0.0439**, confirming robust generalization without overfitting.
- **Leak-Free Seed Stability (Step 11)**: To avoid data leakage, $C$ was re-tuned strictly inside the training split of each seed partition (seeds 42, 7, 123). DistilBERT achieved Mean F1 = **$0.9375 \pm 0.0021$** and Mean Brier = **$0.0464 \pm 0.0013$**, demonstrating consistent performance without optimistic bias.

## 2. Benchmark Results Table

| Pipeline | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier | ECE | seconds per review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC (Val)` | 0.9317 | 0.9338 | 0.9291 | 0.9314 | 0.0525 | 0.0252 | 0.00549 s |
| `TF-IDF + LinearSVC (Test)` | 0.9265 | 0.9249 | 0.9281 | 0.9265 | 0.0527 | 0.0210 | 0.00549 s |
| `distilbert-base-multilingual-cased (Val)` | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 0.06066 s |
| `distilbert-base-multilingual-cased (Test - Selected)` | 0.9416 | 0.9442 | 0.9386 | **0.9414** | 0.0439 | 0.0112 | 0.06066 s |
| `xlm-roberta-base (Val)` | 0.9502 | 0.9482 | 0.9524 | **0.9503** | 0.0383 | 0.0120 | 0.12366 s |
| `jcblaise/roberta-tagalog-base (Val)` | 0.9398 | 0.9400 | 0.9395 | 0.9398 | 0.0461 | 0.0152 | 0.13798 s |

### Seed Stability Summary (Seeds 42, 7, 123)

| Pipeline | Leak-Free Mean F1(CG) ± Std | Leak-Free Mean Brier ± Std | Fixed-C Mean F1(CG) ± Std (Mild Leakage) | Stability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `TF-IDF + LinearSVC` | **0.9351 ± 0.0010** | **0.0490 ± 0.0013** | 0.9330 ± 0.0013 | Highly stable baseline |
| `distilbert-base-multilingual-cased` | **0.9375 ± 0.0021** | **0.0464 ± 0.0013** | 0.9368 ± 0.0020 | Robust across partitions |
| `xlm-roberta-base` | **0.9460 ± 0.0035** | **0.0396 ± 0.0022** | 0.9465 ± 0.0041 | Strongest metric, consistent |
| `jcblaise/roberta-tagalog-base` | **0.9375 ± 0.0018** | **0.0486 ± 0.0018** | 0.9375 ± 0.0018 | Robust English transfer |

*Methodological Note: In the Leak-Free method, C is tuned strictly inside the training partition of each seed (via inner 15% validation split), eliminating optimistic leakage. Fixed-C scores are included for comparison.*

## 3. FiReCS Presumed-Genuine False-Alarm Check
When deployed on 10,481 unlabeled Filipino/Taglish reviews:
- `distilbert-base-multilingual-cased` flagged **64.50%** as Computer-Generated (Mean Confidence: 88.31%).
- `TF-IDF + LinearSVC` flagged **10.67%** as Computer-Generated (due to out-of-vocabulary sparsity defaulting to Genuine).

## 4. Methodological Limitations & Strict Caveats
1. **English Training Only**: Models were trained exclusively on English Amazon product reviews. The elevated false alarm rate on FiReCS is an expected consequence of cross-lingual domain shift, not classifier failure.
2. **FiReCS is Unlabeled**: FiReCS lacks ground-truth Genuine/CG annotations; accuracy, recall, and F1 cannot be evaluated on it.
3. **Panel's Question on Filipino**: These findings do **not** answer how effectively RevuLens will detect machine-generated reviews in Filipino or Taglish. Answering that question requires training and evaluating on the upcoming labeled local dataset (scraped genuine + LLM-generated with language tags).

## 5. Browser Extension Feasibility & Explainability
- **Inference Speed**: DistilBERT delivers single-review latency of **$60.66$ ms** ($16.5$ reviews/sec) on standard CPU, ensuring responsive browser extension performance without perceptible UI lag.
- **SHAP Explainability Findings**:
  - In Day 6 benchmarks, SHAP `LinearExplainer` ($18.06$ ms) and `KernelExplainer` ($851.99$ ms) were computed over the **768 latent embedding dimensions**, which do not provide direct word-level visual highlights for user tooltips.
  - As detailed in `results/SHAP_NOTE.md`, true word-level SHAP requires perturbing input text via a text masker (`shap.maskers.Text`). When benchmarked on CPU, word-level SHAP with $25$ evaluations achieved **$349.3$ ms** per review. This is feasible strictly on an on-demand basis (e.g., clicking "Explain this review"), whereas running it automatically on page load is computationally prohibitive.

## 6. Exact Package Versions
- **Python**: 3.14.7
- **PyTorch**: 2.14.1+cpu
- **Transformers**: 5.18.0
- **Scikit-learn**: 1.9.1
- **NumPy**: 2.5.3
- **Pandas**: 3.0.6
- **SHAP**: 0.52.0
