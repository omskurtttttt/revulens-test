"""RevuLens Pilot Experiment: Encoder + SVM Comparison & TF-IDF Baseline.

Pipeline:
1. Normalization (lowercase, strip URLs/HTML, normalize punctuation, preserve stop words)
2. Stratified 80/10/10 Split (fixed seed 42)
3. Baseline: TF-IDF (word 1-2 grams) + StandardScaler (train only) + LinearSVC + Sigmoid Calibration
4. Metrics: Accuracy, Precision(CG), Recall(CG), F1(CG), Brier score, ECE (10 bins), Mean Confidence
"""

import os
import re
import html
import json
import random
import time
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

# Configuration & Paths
SALMINEN_DATA_PATH = r"C:\Users\jessica\Downloads\thesis dataset\fake reviews dataset.csv"
FIRECS_TRAIN_PATH = r"C:\Users\jessica\Downloads\thesis dataset\FiReCS_train_set.csv"
FIRECS_TEST_PATH = r"C:\Users\jessica\Downloads\thesis dataset\FiReCS_test_set.csv"
RANDOM_SEED = 42
RESULTS_DIR = "results"
DATA_CACHE_DIR = "data"

CANDIDATE_ENCODERS = [
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base",
    "jcblaise/roberta-tagalog-base"
]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_text(text: Any) -> str:
    """Normalize input review text.
    
    Rules:
    - Lowercase
    - Remove HTML tags and entities
    - Remove URLs
    - Normalize repeated punctuation
    - Preserve stop words
    """
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (&amp;, etc.)
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Lowercase
    text = text.lower()
    # Normalize repeated punctuation (e.g. !!!! -> !, ??? -> ?)
    text = re.sub(r"([!?,.:;])\1+", r"\1", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """Load dataset, map labels, drop duplicates, and normalize.
    
    Supports:
    - Salminen format: ['category', 'rating', 'label', 'text_']
    - Future local format: ['text', 'label', 'language']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    if verbose:
        print(f"Loaded raw file: {filepath} with shape {df.shape}")
        print(f"Columns present: {df.columns.tolist()}")

    # Determine text column
    if "text_" in df.columns:
        text_col = "text_"
    elif "text" in df.columns:
        text_col = "text"
    elif "review" in df.columns:
        text_col = "review"
    else:
        raise ValueError(f"No valid text column found in columns: {df.columns.tolist()}")

    # Normalize text
    df["clean_text"] = df[text_col].apply(normalize_text)

    # Determine label mapping
    if "label" in df.columns:
        # Standardize labels
        # Salminen: OR -> 0 (Genuine), CG -> 1 (Computer-Generated)
        label_mapping = {
            "OR": 0,
            "or": 0,
            "Original": 0,
            "genuine": 0,
            "Genuine": 0,
            0: 0,
            0.0: 0,
            "0": 0,
            "CG": 1,
            "cg": 1,
            "Computer-Generated": 1,
            "fake": 1,
            "Fake": 1,
            1: 1,
            1.0: 1,
            "1": 1,
        }
        df["mapped_label"] = df["label"].map(label_mapping)
        if df["mapped_label"].isnull().any():
            invalid = df[df["mapped_label"].isnull()]["label"].unique()
            print(f"Warning: Found unmapped labels: {invalid}. Dropping these rows.")
            df = df.dropna(subset=["mapped_label"])
        df["label"] = df["mapped_label"].astype(int)
    else:
        df["label"] = -1

    # Language tag
    if "language" not in df.columns:
        df["language"] = "English"

    # Duplicates and empty texts inspection
    initial_len = len(df)
    empty_mask = df["clean_text"].str.strip() == ""
    empty_count = empty_mask.sum()
    if empty_count > 0:
        if verbose:
            print(f"Dropping {empty_count} empty reviews.")
        df = df[~empty_mask]

    dup_count = df.duplicated(subset=["clean_text"]).sum()
    if dup_count > 0:
        if verbose:
            print(f"Dropping {dup_count} duplicate reviews.")
        df = df.drop_duplicates(subset=["clean_text"], keep="first")

    # Text length stats
    df["char_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()

    if verbose:
        print("\n=== Dataset Statistics ===")
        print(f"Final valid rows: {len(df)} (Dropped total: {initial_len - len(df)})")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        print("Text character length stats:")
        print(df["char_length"].describe().to_dict())
        print("Text word count stats:")
        print(df["word_count"].describe().to_dict())

    df["text"] = df["clean_text"]
    return df[["text", "label", "language"]].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    save_path: Optional[str] = None
) -> Dict[str, List[int]]:
    """Stratified 80/10/10 train/validation/test split.
    
    Split is performed BEFORE any fitting and indices are saved to disk.
    """
    if save_path and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Loaded existing splits from {save_path}: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits

    indices = np.arange(len(df))
    labels = df["label"].values

    # 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels
    )

    # 10% validation, 10% test from 20% temp
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Saved stratified split indices to: {save_path}")

    return splits


class ECEEvaluator:
    """Expected Calibration Error (ECE) calculation using equal-width bins."""

    @staticmethod
    def compute(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (prob_pos >= bin_lower) & (prob_pos <= bin_upper)
            else:
                in_bin = (prob_pos >= bin_lower) & (prob_pos < bin_upper)

            count_in_bin = np.sum(in_bin)
            if count_in_bin > 0:
                acc_in_bin = np.mean(y_true[in_bin])
                conf_in_bin = np.mean(prob_pos[in_bin])
                ece += (count_in_bin / n_samples) * np.abs(acc_in_bin - conf_in_bin)

        return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    prob_cg: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute all evaluation metrics required by GEMINI.md:
    
    - Accuracy
    - Precision(CG)
    - Recall(CG)
    - F1(CG)
    - Brier score
    - ECE (10 bins)
    - Mean confidence of correct vs incorrect predictions
    """
    y_pred = (prob_cg >= threshold).astype(int)
    # Confidence defined as max(P(CG), 1 - P(CG))
    confidences = np.maximum(prob_cg, 1.0 - prob_cg)

    acc = accuracy_score(y_true, y_pred)
    prec_cg = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_cg = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_cg = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    brier = brier_score_loss(y_true, prob_cg)
    ece = ECEEvaluator.compute(y_true, prob_cg, n_bins=10)

    correct_mask = (y_pred == y_true)
    incorrect_mask = (y_pred != y_true)

    mean_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
    mean_conf_incorrect = float(np.mean(confidences[incorrect_mask])) if np.any(incorrect_mask) else 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_cg": float(prec_cg),
        "recall_cg": float(rec_cg),
        "f1_cg": float(f1_cg),
        "brier_score": float(brier),
        "ece": float(ece),
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_incorrect": mean_conf_incorrect,
        "confusion_matrix": cm.tolist()
    }


def mean_pooling(token_embeddings: Any, attention_mask: Any) -> Any:
    """Mean pooling over token embeddings taking attention mask into account."""
    import torch
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class FeatureExtractor:
    """Frozen transformer encoder with attention-mask mean pooling."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 64
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading {model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        import torch

        all_embeddings = []
        n_samples = len(texts)
        for i in range(0, n_samples, self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**encoded)
                token_embeddings = outputs[0]
                pooled = mean_pooling(token_embeddings, encoded["attention_mask"])
                all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        if not all_embeddings:
            return np.empty((0, 768), dtype=np.float32)
        return np.vstack(all_embeddings)


def get_cache_filename(model_name: str, split_name: str, max_length: int = 128) -> str:
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{safe_name}_{split_name}_len{max_length}.npy"


def get_or_compute_embeddings(
    model_name: str,
    split_name: str,
    texts: List[str],
    cache_dir: str = DATA_CACHE_DIR,
    max_length: int = 128,
    batch_size: int = 64,
    extractor: Optional[FeatureExtractor] = None
) -> np.ndarray:
    """Load cached .npy embeddings from disk or compute and cache them."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, get_cache_filename(model_name, split_name, max_length))
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Computing embeddings for {model_name} on '{split_name}' ({len(texts)} texts, max_len={max_length})...")
    if extractor is None:
        extractor = FeatureExtractor(model_name, batch_size=batch_size)

    t0 = time.time()
    embeddings = extractor.extract(texts, max_length=max_length)
    elapsed = time.time() - t0
    np.save(cache_path, embeddings)
    print(f"Computed & cached {len(texts)} embeddings in {elapsed:.2f}s ({len(texts)/max(elapsed, 0.001):.1f} reviews/s) -> {cache_path}")
    return embeddings


def run_smoke_test(
    sample_texts: List[str],
    models: Optional[List[str]] = None,
    sample_size: int = 2000,
    max_length: int = 128,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Step 4 requirement: Smoke test on 2,000-review sample to check speed and memory.
    
    Reports time per encoder, RAM usage, and embedding shape.
    """
    if models is None:
        models = CANDIDATE_ENCODERS

    eval_texts = sample_texts[:sample_size]
    print(f"\n==========================================")
    print(f"=== SMOKE TEST: {len(eval_texts)} Reviews ===")
    print(f"==========================================")

    results = {}
    for model_name in models:
        print(f"\n--- Smoke Testing: {model_name} ---")
        tracemalloc.start()
        t0 = time.time()

        extractor = FeatureExtractor(model_name, batch_size=batch_size)
        t_load = time.time() - t0

        t_infer_start = time.time()
        embs = extractor.extract(eval_texts, max_length=max_length)
        infer_time = time.time() - t_infer_start
        total_time = time.time() - t0

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        speed = len(eval_texts) / max(infer_time, 0.001)
        peak_mb = peak_mem / (1024 * 1024)

        report = {
            "model_name": model_name,
            "sample_size": len(eval_texts),
            "max_length": max_length,
            "load_time_sec": round(t_load, 2),
            "infer_time_sec": round(infer_time, 2),
            "total_time_sec": round(total_time, 2),
            "reviews_per_sec": round(speed, 2),
            "peak_memory_mb": round(peak_mb, 2),
            "embedding_shape": list(embs.shape),
            "has_nan": bool(np.isnan(embs).any())
        }
        results[model_name] = report
        print(f"Result for {model_name}:")
        print(f"  Load Time:      {report['load_time_sec']}s")
        print(f"  Inference Time: {report['infer_time_sec']}s ({report['reviews_per_sec']} reviews/sec)")
        print(f"  Peak Memory:    {report['peak_memory_mb']} MB")
        print(f"  Shape:          {report['embedding_shape']}, Has NaNs: {report['has_nan']}")

    # Save smoke test report in JSON and Markdown
    os.makedirs(RESULTS_DIR, exist_ok=True)
    smoke_file = os.path.join(RESULTS_DIR, "smoke_test_results.json")
    with open(smoke_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    smoke_md = os.path.join(RESULTS_DIR, "smoke_test_results.md")
    with open(smoke_md, "w", encoding="utf-8") as f:
        f.write("# Smoke Test Results (2,000-Review Sample)\n\n")
        f.write("| Model | Load Time (s) | Inference Time (s) | Speed (reviews/s) | Peak Memory (MB) | Embedding Shape |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for m, rep in results.items():
            f.write(f"| `{m}` | {rep['load_time_sec']} | {rep['infer_time_sec']} | {rep['reviews_per_sec']} | {rep['peak_memory_mb']} | {rep['embedding_shape']} |\n")

    print(f"\nSmoke test complete. Results saved to: {smoke_file} and {smoke_md}")

    return results


def train_baseline_tfidf(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    c_grid: Optional[List[float]] = None
) -> Tuple[Any, Any, Any, Dict[str, Any], np.ndarray]:
    """Train TF-IDF + StandardScaler + LinearSVC + Sigmoid Calibration baseline.
    
    Rules:
    - TF-IDF fit on TRAIN only.
    - StandardScaler(with_mean=False) fit on TRAIN only.
    - C tuned over grid on VALIDATION set.
    - Sigmoid calibration via CalibratedClassifierCV(method="sigmoid", cv=5) fit on TRAIN only.
    """
    if c_grid is None:
        c_grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    print("\n--- Training TF-IDF Baseline ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)

    # StandardScaler fit on train only (with_mean=False preserves sparse matrix efficiency)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_val_scaled = scaler.transform(X_val_tfidf)

    best_c = c_grid[0]
    best_val_f1 = -1.0

    print("Tuning LinearSVC C on validation set...")
    for c in c_grid:
        svc = LinearSVC(C=c, random_state=RANDOM_SEED, max_iter=2000, dual=False)
        svc.fit(X_train_scaled, train_labels)
        val_preds = svc.predict(X_val_scaled)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
        print(f"  C={c:6.2f} -> Validation F1(CG) = {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c

    print(f"Optimal C selected: {best_c} (Validation F1(CG) = {best_val_f1:.4f})")

    # Fit CalibratedClassifierCV on TRAIN only using best_c
    base_svc = LinearSVC(C=best_c, random_state=RANDOM_SEED, max_iter=2000, dual=False)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=5
    )
    print("Fitting CalibratedClassifierCV(method='sigmoid', cv=5) on TRAIN set...")
    calibrated_clf.fit(X_train_scaled, train_labels)

    # Predict probabilities on Validation
    val_probas = calibrated_clf.predict_proba(X_val_scaled)[:, 1]
    val_metrics = compute_metrics(val_labels, val_probas)

    print("\n--- Baseline Validation Results ---")
    print(f"Accuracy:              {val_metrics['accuracy']:.4f}")
    print(f"Precision (CG):        {val_metrics['precision_cg']:.4f}")
    print(f"Recall (CG):           {val_metrics['recall_cg']:.4f}")
    print(f"F1 (CG):               {val_metrics['f1_cg']:.4f}")
    print(f"Brier Score:           {val_metrics['brier_score']:.4f}")
    print(f"ECE (10 bins):         {val_metrics['ece']:.4f}")
    print(f"Mean Conf (Correct):   {val_metrics['mean_conf_correct'] * 100:.2f}%")
    print(f"Mean Conf (Incorrect): {val_metrics['mean_conf_incorrect'] * 100:.2f}%")
    print(f"Confusion Matrix:      {val_metrics['confusion_matrix']}")

    return calibrated_clf, scaler, vectorizer, val_metrics, val_probas


def run_day_1(dataset_path: str = SALMINEN_DATA_PATH) -> Dict[str, Any]:
    """Execute Day 1 workflow:
    1. Set seed
    2. Load and inspect dataset
    3. Stratified 80/10/10 split
    4. Train and calibrate TF-IDF baseline
    5. Save Day 1 outputs
    """
    set_seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(">>> Day 1: Loading & Preprocessing Salminen Dataset...")
    df = load_dataset(dataset_path, verbose=True)

    print("\n>>> Performing Stratified 80/10/10 Split...")
    splits_path = os.path.join(DATA_CACHE_DIR, "split_indices.json")
    splits = split_dataset(df, seed=RANDOM_SEED, save_path=splits_path)

    train_df = df.iloc[splits["train"]].reset_index(drop=True)
    val_df = df.iloc[splits["val"]].reset_index(drop=True)
    test_df = df.iloc[splits["test"]].reset_index(drop=True)

    print(f"Train set: {len(train_df)} rows | Label balance: {train_df['label'].value_counts().to_dict()}")
    print(f"Val set:   {len(val_df)} rows | Label balance: {val_df['label'].value_counts().to_dict()}")
    print(f"Test set:  {len(test_df)} rows | Label balance: {test_df['label'].value_counts().to_dict()}")

    clf, scaler, vectorizer, val_metrics, val_probas = train_baseline_tfidf(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].values,
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["label"].values
    )

    # Save baseline validation metrics to results/
    baseline_results = {
        "pipeline": "TF-IDF (1-2 grams) + LinearSVC (Sigmoid Calibrated)",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "metrics": val_metrics
    }
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\n[DONE] Baseline pipeline executed successfully. Results saved to {RESULTS_DIR}/baseline_metrics.json")
    return baseline_results


def run_day_2(
    dataset_path: str = SALMINEN_DATA_PATH,
    max_length: int = 128,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Execute Day 2 workflow:
    1. Set seed
    2. Load preprocessed dataset and load splits
    3. Run smoke test on 2,000-review sample for all 3 encoders
    4. Extract and cache embeddings for Train, Val, Test for each encoder
    """
    set_seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(">>> Day 2: Loading dataset and stratified splits...")
    df = load_dataset(dataset_path, verbose=False)
    splits_path = os.path.join(DATA_CACHE_DIR, "split_indices.json")
    splits = split_dataset(df, seed=RANDOM_SEED, save_path=splits_path)

    train_df = df.iloc[splits["train"]].reset_index(drop=True)
    val_df = df.iloc[splits["val"]].reset_index(drop=True)
    test_df = df.iloc[splits["test"]].reset_index(drop=True)

    # 1. Smoke test on 2,000-review sample (Step 4 of GEMINI.md)
    print("\n>>> Step 4: Running smoke test on 2,000 reviews...")
    smoke_results = run_smoke_test(
        sample_texts=train_df["text"].tolist(),
        models=CANDIDATE_ENCODERS,
        sample_size=2000,
        max_length=max_length,
        batch_size=batch_size
    )

    # 2. Extract and cache embeddings for each encoder (Step 5 of GEMINI.md)
    print("\n>>> Step 5: Extracting and caching embeddings for all splits...")
    cached_info = {}
    for model_name in CANDIDATE_ENCODERS:
        print(f"\n--- Processing Encoder: {model_name} ---")
        extractor = FeatureExtractor(model_name, batch_size=batch_size)

        # Validation split
        val_emb = get_or_compute_embeddings(
            model_name=model_name,
            split_name="val",
            texts=val_df["text"].tolist(),
            cache_dir=DATA_CACHE_DIR,
            max_length=max_length,
            batch_size=batch_size,
            extractor=extractor
        )

        # Test split
        test_emb = get_or_compute_embeddings(
            model_name=model_name,
            split_name="test",
            texts=test_df["text"].tolist(),
            cache_dir=DATA_CACHE_DIR,
            max_length=max_length,
            batch_size=batch_size,
            extractor=extractor
        )

        # Train split
        train_emb = get_or_compute_embeddings(
            model_name=model_name,
            split_name="train",
            texts=train_df["text"].tolist(),
            cache_dir=DATA_CACHE_DIR,
            max_length=max_length,
            batch_size=batch_size,
            extractor=extractor
        )

        cached_info[model_name] = {
            "train_shape": list(train_emb.shape),
            "val_shape": list(val_emb.shape),
            "test_shape": list(test_emb.shape),
        }

    feature_extraction_results = {
        "smoke_test": smoke_results,
        "cached_embeddings": cached_info
    }
    with open(os.path.join(RESULTS_DIR, "feature_extraction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(feature_extraction_results, f, indent=2)

    print(f"\n[DONE] Feature extraction workflow executed successfully. Summary saved to {RESULTS_DIR}/feature_extraction_summary.json")
    return feature_extraction_results


run_baseline = run_day_1
run_feature_extraction = run_day_2


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("baseline", "day1"):
        run_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] in ("features", "day2"):
        run_feature_extraction()
    else:
        # Default to running feature extraction when invoked directly
        run_feature_extraction()
