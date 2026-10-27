import os
import pytest
import pandas as pd
import numpy as np

# We will import from pilot once implemented
from pilot import normalize_text, load_dataset, split_dataset, compute_metrics, ECEEvaluator

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
