import os
import pytest
import pandas as pd
import numpy as np

from pilot import (
    normalize_text,
    load_dataset,
    split_dataset,
    compute_metrics,
    ECEEvaluator,
    mean_pooling,
    FeatureExtractor,
    get_or_compute_embeddings,
    train_linear_svc_pipeline,
    select_best_pipeline,
    evaluate_seed_stability,
    plot_confusion_matrices,
)

def test_normalize_text():
    # HTML removal
    raw_html = "<div>Awesome product! <br/>Works well.</div>"
    assert "<" not in normalize_text(raw_html)
    assert ">" not in normalize_text(raw_html)

    # URL removal
    raw_url = "Check this out https://example.com/product and www.test.com for details!"
    norm_url = normalize_text(raw_url)
    assert "https://" not in norm_url
    assert "www." not in norm_url
    assert "example.com" not in norm_url

    # Lowercasing
    raw_case = "VERY GOOD QUALITY"
    assert normalize_text(raw_case) == "very good quality"

    # Preserves stop words
    raw_stops = "this is a very good product and it works for me"
    norm_stops = normalize_text(raw_stops)
    for word in ["this", "is", "a", "and", "it", "for", "me"]:
        assert word in norm_stops

    # Punctuation normalization
    raw_punct = "Really???? Yes!!!!"
    norm_punct = normalize_text(raw_punct)
    assert "??" not in norm_punct
    assert "!!" not in norm_punct
    assert "really? yes!" == norm_punct

def test_load_dataset_salminen_format(tmp_path):
    # Mock Salminen dataset
    csv_file = tmp_path / "mock_salminen.csv"
    data = {
        "category": ["Electronics", "Books", "Home", "Home"],
        "rating": [5.0, 4.0, 1.0, 1.0],
        "label": ["OR", "CG", "OR", "OR"],  # contains duplicate text
        "text_": [
            "Original review 1",
            "Computer generated review 2",
            "Original review 3",
            "Original review 3"  # duplicate
        ]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_dataset(str(csv_file))
    assert "text" in df.columns
    assert "label" in df.columns
    assert "language" in df.columns
    # Duplicate removed: 4 rows -> 3 rows
    assert len(df) == 3
    # Labels mapped to 0 and 1
    assert set(df["label"].unique()) == {0, 1}
    assert df.loc[df["text"] == "original review 1", "label"].iloc[0] == 0
    assert df.loc[df["text"] == "computer generated review 2", "label"].iloc[0] == 1

def test_load_dataset_future_format(tmp_path):
    # Mock Future format with text, label, language
    csv_file = tmp_path / "mock_future.csv"
    data = {
        "text": ["Magandang produkto", "Mabilis dumating pero sira"],
        "label": [0, 1],
        "language": ["Filipino", "Taglish"]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_dataset(str(csv_file))
    assert len(df) == 2
    assert "language" in df.columns
    assert set(df["language"].unique()) == {"Filipino", "Taglish"}

def test_split_dataset(tmp_path):
    n = 100
    df = pd.DataFrame({
        "text": [f"Review {i}" for i in range(n)],
        "label": [0] * 50 + [1] * 50,
        "language": ["English"] * n
    })
    
    indices_path = tmp_path / "splits.json"
    splits = split_dataset(df, seed=42, save_path=str(indices_path))

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # 80/10/10 split
    assert len(train_idx) == 80
    assert len(val_idx) == 10
    assert len(test_idx) == 10

    # No overlap
    assert len(set(train_idx).intersection(val_idx)) == 0
    assert len(set(train_idx).intersection(test_idx)) == 0
    assert len(set(val_idx).intersection(test_idx)) == 0

    # Stratified balance in train (40 label 0, 40 label 1)
    train_labels = df.iloc[train_idx]["label"]
    assert (train_labels == 0).sum() == 40
    assert (train_labels == 1).sum() == 40

    # File was saved
    assert os.path.exists(indices_path)

def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    # Perfect calibrated predictions
    prob_cg = np.array([0.05, 0.10, 0.90, 0.95])
    
    metrics = compute_metrics(y_true, prob_cg)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_cg"] == 1.0
    assert metrics["recall_cg"] == 1.0
    assert metrics["f1_cg"] == 1.0
    assert metrics["brier_score"] < 0.05
    assert metrics["ece"] < 0.15
    assert metrics["mean_conf_correct"] > 0.90
    assert np.isnan(metrics["mean_conf_incorrect"]) or metrics["mean_conf_incorrect"] == 0.0

def test_mean_pooling_function():
    import torch
    # 2 samples, seq_len 3, dim 2
    token_emb = torch.tensor([
        [[2.0, 4.0], [4.0, 8.0], [0.0, 0.0]], # 2 valid tokens -> mean should be [3.0, 6.0]
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]  # 3 valid tokens -> mean should be [3.0, 3.0]
    ])
    mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1]
    ])
    pooled = mean_pooling(token_emb, mask)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 6.0]))
    assert torch.allclose(pooled[1], torch.tensor([3.0, 3.0]))

def test_feature_extractor_extraction():
    # Use distilbert since its weights are already cached
    extractor = FeatureExtractor("distilbert-base-multilingual-cased", batch_size=2)
    samples = ["Great item!", "Medyo sira yung box."]
    emb = extractor.extract(samples, max_length=32)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 768)
    assert emb.dtype == np.float32
    assert not np.isnan(emb).any()

def test_get_or_compute_embeddings(tmp_path):
    samples = ["First test review.", "Second test review."]
    cache_dir = str(tmp_path / "cache")
    
    # Compute and save
    emb1 = get_or_compute_embeddings(
        model_name="distilbert-base-multilingual-cased",
        split_name="test_split",
        texts=samples,
        cache_dir=cache_dir,
        max_length=32,
        batch_size=2
    )
    assert emb1.shape == (2, 768)
    
    # Reload from cache
    emb2 = get_or_compute_embeddings(
        model_name="distilbert-base-multilingual-cased",
        split_name="test_split",
        texts=samples,
        cache_dir=cache_dir,
        max_length=32,
        batch_size=2
    )
    assert np.array_equal(emb1, emb2)


def test_train_linear_svc_pipeline():
    np.random.seed(42)
    # 100 samples, dim 16
    X_train = np.random.randn(100, 16)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(30, 16)
    y_val = (X_val[:, 0] > 0).astype(int)

    clf, scaler, best_c, metrics, probs = train_linear_svc_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        c_grid=[0.01, 0.1, 1.0],
        seed=42
    )
    assert best_c in [0.01, 0.1, 1.0]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_cg"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert len(probs) == 30
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_select_best_pipeline_highest_f1():
    results = {
        "ModelA": {"metrics": {"f1_cg": 0.95}},
        "ModelB": {"metrics": {"f1_cg": 0.90}},
    }
    speeds = {"ModelA": 10.0, "ModelB": 20.0}
    winner, rationale = select_best_pipeline(results, speeds)
    assert winner == "ModelA"
    assert "highest Validation F1(CG)" in rationale


def test_select_best_pipeline_speed_tiebreak():
    # ModelB is within 0.01 of ModelA (0.945 vs 0.950), but ModelB is much faster
    results = {
        "ModelA": {"metrics": {"f1_cg": 0.950}},
        "ModelB": {"metrics": {"f1_cg": 0.945}},
    }
    speeds = {"ModelA": 7.0, "ModelB": 15.0}
    winner, rationale = select_best_pipeline(results, speeds)
    assert winner == "ModelB"
    assert "faster/smaller" in rationale


def test_evaluate_seed_stability():
    np.random.seed(42)
    X_train = np.random.randn(80, 16)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(20, 16)
    y_val = (X_val[:, 0] > 0).astype(int)

    stability = evaluate_seed_stability(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        best_c=1.0,
        seeds=[42, 7, 123]
    )
    assert "mean_f1" in stability
    assert "std_f1" in stability
    assert len(stability["f1_scores"]) == 3
    assert stability["std_f1"] >= 0.0


def test_plot_confusion_matrices(tmp_path):
    mock_results = {
        "Baseline": {"metrics": {"f1_cg": 0.93, "accuracy": 0.93, "confusion_matrix": [[1888, 133], [143, 1875]]}},
        "DistilBERT": {"metrics": {"f1_cg": 0.94, "accuracy": 0.94, "confusion_matrix": [[1900, 121], [130, 1888]]}},
    }
    out_file = str(tmp_path / "test_cm.png")
    saved_path = plot_confusion_matrices(mock_results, output_path=out_file)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1000
