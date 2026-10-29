import os
import json
import pytest
import pandas as pd
import numpy as np

from pilot import (
    RESULTS_DIR,
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
    evaluate_test_set,
    evaluate_baseline_test_set,
    plot_reliability_diagram,
    load_firecs_dataset,
    compute_confidence_distribution,
    evaluate_firecs_pipeline,
    benchmark_single_review_latency,
    benchmark_shap_latency,
    generate_consolidated_results,
    run_day_7,
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
    assert "mean_brier" in stability
    assert "std_brier" in stability
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


def test_evaluate_test_set():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    np.random.seed(42)
    X_train = np.random.randn(100, 8)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.random.randn(20, 8)
    y_test = (X_test[:, 0] > 0).astype(int)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    base_svc = LinearSVC(C=1.0, random_state=42, dual=False)
    clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    clf.fit(X_train_scaled, y_train)

    metrics, probas = evaluate_test_set(clf, scaler, X_test, y_test)
    assert "accuracy" in metrics
    assert "precision_cg" in metrics
    assert "recall_cg" in metrics
    assert "f1_cg" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert len(probas) == len(y_test)
    assert np.all((probas >= 0.0) & (probas <= 1.0))


def test_evaluate_baseline_test_set():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    train_texts = ["great product love it", "terrible waste of money", "excellent quality", "bad broken item"] * 10
    y_train = np.array([0, 1, 0, 1] * 10)
    test_texts = ["great quality love it", "horrible broken bad item"]
    y_test = np.array([0, 1])

    vectorizer = TfidfVectorizer().fit(train_texts)
    X_tr_tfidf = vectorizer.transform(train_texts)
    scaler = StandardScaler(with_mean=False).fit(X_tr_tfidf)
    X_tr_scaled = scaler.transform(X_tr_tfidf)

    base_svc = LinearSVC(C=1.0, random_state=42, dual=False)
    clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    clf.fit(X_tr_scaled, y_train)

    metrics, probas = evaluate_baseline_test_set(clf, scaler, vectorizer, test_texts, y_test)
    assert "accuracy" in metrics
    assert "f1_cg" in metrics
    assert len(probas) == len(y_test)
    assert np.all((probas >= 0.0) & (probas <= 1.0))


def test_plot_reliability_diagram(tmp_path):
    mock_data = {
        "Validation": {
            "y_true": np.array([0, 0, 1, 1, 0, 1, 0, 1]),
            "y_prob": np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.3, 0.7]),
            "label": "TestPipeline (Val)"
        },
        "Test": {
            "y_true": np.array([0, 0, 1, 1, 0, 1, 0, 1]),
            "y_prob": np.array([0.12, 0.18, 0.82, 0.88, 0.14, 0.86, 0.28, 0.72]),
            "label": "TestPipeline (Test)"
        }
    }
    out_file = str(tmp_path / "test_reliability.png")
    saved_path = plot_reliability_diagram(mock_data, output_path=out_file, n_bins=5)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1000


def test_load_firecs_dataset(tmp_path):
    train_file = tmp_path / "mock_firecs_train.csv"
    test_file = tmp_path / "mock_firecs_test.csv"

    pd.DataFrame({
        "review": ["Salamat sa item napakaganda <br/> www.shop.com", "Mabilis dumating"],
        "label": [1.0, 0.0]  # Sentiment labels, not genuine/CG!
    }).to_csv(train_file, index=False)

    pd.DataFrame({
        "review": ["Mabilis dumating", "Sira agad pagkabukas"],  # Contains 1 duplicate across splits
        "label": [0.0, 1.0]
    }).to_csv(test_file, index=False)

    df = load_firecs_dataset(str(train_file), str(test_file), verbose=False)
    # 4 rows raw -> 1 duplicate -> 3 unique rows
    assert len(df) == 3
    assert "clean_text" in df.columns
    assert "word_count" in df.columns
    assert "char_length" in df.columns
    assert "language" in df.columns
    assert df["language"].iloc[0] == "Filipino/Taglish"
    # Ensure text is normalized
    assert "<br/>" not in df["clean_text"].iloc[0]
    assert "www.shop.com" not in df["clean_text"].iloc[0]
    # Ensure raw sentiment label is not mistaken as target label
    assert "label" not in df.columns


def test_compute_confidence_distribution():
    # 4 predictions: 2 genuine (0.1, 0.4), 2 CG (0.7, 0.9)
    probas = np.array([0.1, 0.4, 0.7, 0.9])
    res = compute_confidence_distribution(probas, threshold=0.50)

    assert res["total_reviews"] == 4
    assert res["predicted_cg_count"] == 2
    assert res["predicted_genuine_count"] == 2
    assert res["false_alarm_rate_percent"] == 50.0
    assert res["presumed_genuine_rate_percent"] == 50.0
    # Confidences: [0.9, 0.6, 0.7, 0.9] -> mean = 3.1 / 4 = 0.775
    assert np.isclose(res["mean_confidence"], 0.775, atol=1e-3)
    assert np.isclose(res["median_confidence"], 0.80, atol=1e-3)
    assert res["confidence_buckets"]["high_gte_80_percent"] == 2
    assert res["confidence_buckets"]["moderate_60_to_80_percent"] == 2
    assert res["confidence_buckets"]["low_50_to_60_percent"] == 0


def test_evaluate_firecs_pipeline():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    np.random.seed(42)
    X_train = np.random.randn(50, 4)
    y_train = (X_train[:, 0] > 0).astype(int)

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    clf = CalibratedClassifierCV(LinearSVC(dual=False), method="sigmoid", cv=2).fit(X_train_sc, y_train)

    X_firecs = np.random.randn(10, 4)
    metrics, probas, confs = evaluate_firecs_pipeline(clf, scaler, X_firecs)

    assert metrics["total_reviews"] == 10
    assert len(probas) == 10
    assert len(confs) == 10
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert np.all((confs >= 0.50) & (confs <= 1.0))


def test_benchmark_single_review_latency_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    corpus = ["great product works well", "terrible broken item", "fast shipping loved it", "waste of money bad"]
    labels = np.array([0, 1, 0, 1])

    vec = TfidfVectorizer().fit(corpus)
    sc = StandardScaler(with_mean=False).fit(vec.transform(corpus))
    clf = CalibratedClassifierCV(LinearSVC(dual=False), method="sigmoid", cv=2).fit(sc.transform(vec.transform(corpus)), labels)

    pipeline_obj = {"vectorizer": vec, "scaler": sc, "clf": clf}
    res = benchmark_single_review_latency("tfidf", pipeline_obj, test_review="great product works well", n_warmup=2, n_runs=5)

    assert res["n_runs"] == 5
    assert res["mean_ms"] > 0
    assert res["median_ms"] > 0
    assert res["seconds_per_review"] > 0
    assert res["reviews_per_sec"] > 0


def test_benchmark_shap_latency():
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    np.random.seed(42)
    X = np.random.randn(25, 4)
    y = (X[:, 0] > 0).astype(int)
    clf = CalibratedClassifierCV(LinearSVC(dual=False), method="sigmoid", cv=2).fit(X, y)

    res = benchmark_shap_latency(clf, X, X[0], max_samples=10, explainer_mode="linear")
    assert res["shap_installed"] is True
    assert res["seconds_per_review"] is not None
    assert res["seconds_per_review"] >= 0.0
    assert res["background_sample_limit"] == 10


def test_generate_consolidated_results(tmp_path):
    df_results, report_path = generate_consolidated_results(str(tmp_path))

    required_cols = ["Pipeline", "Accuracy", "Precision(CG)", "Recall(CG)", "F1(CG)", "Brier", "ECE", "seconds per review"]
    assert list(df_results.columns) == required_cols
    assert len(df_results) == 6

    # Verify CSV was written
    csv_file = tmp_path / "results_table.csv"
    assert os.path.exists(csv_file)
    csv_df = pd.read_csv(csv_file)
    assert list(csv_df.columns) == required_cols

    # Verify Markdown table was written
    md_file = tmp_path / "results_table.md"
    assert os.path.exists(md_file)
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "| Pipeline | Accuracy | Precision(CG) |" in md_text
        assert "distilbert-base-multilingual-cased" in md_text

    # Verify REPORT.md was written with all required elements
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
        assert "Executive Summary & Selection Decision" in report_text
        assert "FiReCS Presumed-Genuine False-Alarm Check" in report_text
        assert "Methodological Limitations & Strict Caveats" in report_text
        assert "Exact Package Versions" in report_text
        assert "**Python**: 3.14.7" in report_text
        assert "**PyTorch**: 2.14.1+cpu" in report_text
        assert "**Transformers**: 5.18.0" in report_text
        assert "**Scikit-learn**: 1.9.1" in report_text


def test_shap_note_and_speed_harmonization():
    """Verify that SHAP note exists and speed recommendations match empirical benchmarks."""
    shap_note_path = os.path.join(RESULTS_DIR, "SHAP_NOTE.md")
    if os.path.exists(shap_note_path):
        with open(shap_note_path, "r", encoding="utf-8") as f:
            note_text = f.read()
            assert "768-dimensional dense pooled embedding vector" in note_text
            assert "Latent embedding dimensions" in note_text
            assert "shap.maskers.Text" in note_text
            assert "18.1 ms" in note_text
            assert "852.0 ms" in note_text

    speed_md_path = os.path.join(RESULTS_DIR, "speed_benchmark.md")
    assert os.path.exists(speed_md_path)
    with open(speed_md_path, "r", encoding="utf-8") as f:
        speed_text = f.read()
        assert "18.1 ms" in speed_text
        assert "852.0 ms" in speed_text
        assert "<10 ms" not in speed_text
        assert "~1-3s" not in speed_text

    val_json_path = os.path.join(RESULTS_DIR, "validation_metrics_summary.json")
    assert os.path.exists(val_json_path)
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        assert len(val_data["pipelines"]) == 4
        assert val_data["pipelines"]["TF-IDF + LinearSVC"]["reviews_per_sec"] == 182.2
        assert "tie-break" in val_data["selection_rationale"].lower()




