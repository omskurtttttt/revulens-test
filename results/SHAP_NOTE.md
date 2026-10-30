# SHAP Feature Attribution Investigation & Technical Note

**Date**: 2026-10-30  
**Context**: RevuLens Pilot Experiment (Step 14 - Explainability for Browser Extension)

---

## 1. What Input Was Given to Each Explainer in Day 6?

During Day 6 latency benchmarking (`benchmark_shap_latency` in `pilot.py`), two distinct explainers were tested:

1. **TF-IDF Explainer (`shap.LinearExplainer`)**:
   - **Input**: 25,000-dimensional sparse TF-IDF n-gram vectors (`X_train_tfidf` and `test_tfidf`).
   - **Target**: Linear coefficients of the base `LinearSVC`.
   - **Attribution Unit**: Word and character n-grams (e.g., `"product"`, `"arrived"`, `"exceeded"`).
   - **Findings**: This *does* correspond directly to text features. However, it only applies to the baseline TF-IDF model.

2. **DistilBERT Explainers (`shap.LinearExplainer` and `shap.KernelExplainer`)**:
   - **Input**: The 768-dimensional dense pooled embedding vector output by `db_sc.transform(db_train_emb)` and `test_emb` (shape: `(1, 768)`).
   - **Target**:
     - `LinearExplainer`: Linear weights of `LinearSVC` against the 768 scaled embedding coordinates.
     - `KernelExplainer`: Predict-probability output of `CalibratedClassifierCV` against perturbed 768-dimensional vectors.
   - **Attribution Unit**: **Latent embedding dimensions** (e.g., dimension index 0 through 767).

---

## 2. Direct Assessment: Latent Dimensions vs. Word-Level Highlighting

> [!WARNING]
> **Key Finding**: In Day 6, SHAP for DistilBERT was evaluated on **768 latent embedding dimensions**, NOT on individual words or tokens.

- **Why this does NOT satisfy the project requirement**:
  A browser extension cannot display a tooltip saying *"Dimension #412 contributed +0.034 to the fake score"*. The end-user expects visual highlighting of specific suspicious or genuine words within the review text (e.g., highlighting *"exceeded my expectations"* in red or green).
- Latent transformer representations collapse all contextual token interactions into a single 768-dimensional pooled vector. Explaining the linear classifier on top of those 768 dimensions explains the *classifier*, not the *text*.
- Describing `KernelExplainer` as "token-level" in earlier draft summaries was technically inaccurate and has been corrected.

---

## 3. Simplest Solution for Word-Level Attribution & Latency Benchmark

To produce true word-level Shapley values for the frozen transformer + SVM pipeline without architectural overhaul, the explainer must wrap the entire end-to-end pipeline:
$$\text{Raw Text} \xrightarrow{\text{DistilBERT}} \text{768d Embeddings} \xrightarrow{\text{StandardScaler}} \text{Calibrated LinearSVC} \xrightarrow{P(\text{CG})} [0, 1]$$

Using SHAP's official text masker (`shap.maskers.Text(tokenizer=r"\W+")`) with a custom inference wrapper:
```python
def predict_proba_texts(texts: List[str]) -> np.ndarray:
    embs = extractor.extract(texts, max_length=128)
    scaled = scaler.transform(embs)
    return clf.predict_proba(scaled)[:, 1]

explainer = shap.Explainer(predict_proba_texts, masker=shap.maskers.Text(tokenizer=r"\W+"))
shap_values = explainer([test_review], max_evals=N)
```

### Empirical Benchmark on 1 Sample Review (21 words, CPU)

Using the benchmark text: *"This product arrived on time and was well packaged. The quality exceeded my expectations for the price, works exactly as described."*

| Explainer Mode | Masker / Input | Max Evaluations (`max_evals`) | Latency (s) | Latency (ms) | Browser Extension Feasibility |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Embedding LinearExplainer** | 768-d Vector | 100 | **0.0181 s** | 18.1 ms | Real-time, but uninterpretable to users |
| **Embedding KernelExplainer** | 768-d Vector | 50 | **0.8520 s** | 852.0 ms | Latent dimensions only |
| **True Word-Level SHAP** | Text Masker (`\W+`) | 25 | **0.3493 s** | 349.3 ms | **Feasible on-demand** ("Explain Review" button) |
| **True Word-Level SHAP** | Text Masker (`\W+`) | 50 | **4.5941 s** | 4,594.1 ms | Too slow for interactive UI |

### Recommendation for Browser Extension Architecture
1. **Never run true word-level SHAP on page load**: Running text perturbation on 10 reviews would lock the browser or server for ~35–45 seconds.
2. **On-Demand Execution**: Trigger word-level SHAP only when the user explicitly clicks an "Explain Review" button.
3. **Budget Sample Limit**: Constrain `max_evals` to ~20–25 evaluations (giving sub-second ~350 ms latency on CPU) or explore integrated gradients / attention-based token weights if millisecond response is required.

---

## 4. Corrections Applied to `speed_benchmark.md`

The narrative text in `results/speed_benchmark.md` has been aligned with the measured empirical numbers:
- **Prior Draft**: Claimed LinearExplainer was `"<10 ms"` and KernelExplainer was `"~1-3s"`.
- **Corrected**: Accurately reflects measured numbers:
  - DistilBERT Embedding LinearExplainer: **18.1 ms** (0.0181 s).
  - DistilBERT Embedding KernelExplainer (50 samples): **852.0 ms** (0.8520 s).
  - True Word-Level SHAP (`max_evals=25`): **349.3 ms** (0.3493 s).

---

## 5. Methodological Limitations of the SHAP Benchmark

1. **Sample Size**: Evaluated strictly on a single benchmark review instance (26 words, 21 whitespace-delimited tokens).
2. **Explanation Stability Untested**: The consistency and variance of Shapley attributions across diverse review lengths (e.g., short 5-word reviews vs. long 200-word reviews), sentiment polarities, and domain styles were not profiled in this pilot.
3. **Cross-Lingual Behavior**: The token masker behavior on Filipino/Taglish morphology (affixes and clitics like *nag-*, *pagka-*, *-han*) was not evaluated.
4. **Backend Concurrency**: Latencies reflect single-threaded local CPU execution; multi-user concurrent throughput in a FastAPI server environment remains to be tested in future milestones.

