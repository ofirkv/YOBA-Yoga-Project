#utils/calibrate_classifier.py
import csv
import json
from pathlib import Path
import numpy as np

EPS = 1e-9
INVERT_FEATURES = {"center_of_mass_y": True, "horizontal_vs_vertical_ratio": True}
POS_LABEL = "raised"
NEG_LABEL = "lowered"

BASE_DIR = Path(__file__).parent.parent

def load_features(path):
    path = Path(path)
    names, data = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[1:]
        for row in reader:
            if not row: 
                continue
            names.append(row[0])
            data.append([float(x) for x in row[1:]])
    return names, np.array(data, dtype=np.float32), feature_names

def load_labels(path):
    path = Path(path)
    labels = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            labels[row[0]] = row[1]
    return labels


def align_data(names, X, labels_map):
    X_aligned, y_aligned, kept_names = [], [], []
    for n, row in zip(names, X):
        if n not in labels_map:
            continue
        y_aligned.append(1 if labels_map[n] == POS_LABEL else 0)
        X_aligned.append(row)
        kept_names.append(n)
    return np.array(X_aligned, dtype=np.float32), np.array(y_aligned, dtype=np.int32), kept_names


def compute_thresholds(X, y, feature_names):
    thresholds = {}
    stats = {}
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        pos_vals = col[y == 1]
        neg_vals = col[y == 0]
        mean_pos = float(np.mean(pos_vals)) if pos_vals.size else float(np.nan)
        mean_neg = float(np.mean(neg_vals)) if neg_vals.size else float(np.nan)
        threshold = 0.5 * (mean_pos + mean_neg) if not (np.isnan(mean_pos) or np.isnan(mean_neg)) else float(np.mean(col))
        thresholds[fname] = threshold
        stats[fname] = {
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "std": float(np.std(col) if col.size else np.nan)
        }
    return thresholds, stats


def prepare_normalized_X(X, thresholds, stats, feature_names, add_bias=False):
    N, D = X.shape
    Z = np.zeros_like(X, dtype=np.float32)
    for j, fname in enumerate(feature_names):
        t = thresholds.get(fname, 0.0)
        s = stats.get(fname, {}).get("std", EPS) or EPS
        if INVERT_FEATURES.get(fname, False):
            Z[:, j] = (t - X[:, j]) / (s + EPS)
        else:
            Z[:, j] = (X[:, j] - t) / (s + EPS)
    if add_bias:
        Z = np.concatenate([Z, np.ones((N, 1), dtype=np.float32)], axis=1)
    return Z


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_gradient_descent(Z, y, lr=0.1, epochs=1000, weight_decay=1e-4, seed=0, verbose=False):
    np.random.seed(seed)
    N, D = Z.shape
    w = np.random.randn(D).astype(np.float32) * 0.01
    for epoch in range(1, epochs + 1):
        logits = Z.dot(w)
        probs = sigmoid(logits)
        loss = -np.mean(y * np.log(probs + EPS) + (1 - y) * np.log(1 - probs + EPS)) + 0.5 * weight_decay * (w @ w)
        grad = Z.T.dot(probs - y) / N + weight_decay * w
        w -= lr * grad
        if verbose and (epoch == 1 or epoch % (epochs // 5 or 1) == 0 or epoch == epochs):
            acc = 1 - np.abs(np.round(probs) - y).sum() / len(y)
            print(f"[train] epoch {epoch}/{epochs} loss={loss:.6f} acc={acc:.4f}")
    return w


def compute_metrics(scores, y_true, threshold=0.5):
    preds = (scores >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    accuracy = (tp + tn) / (len(y_true) + EPS)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def find_best_threshold(scores, y_true):
    unique_scores = np.unique(np.sort(scores))
    if unique_scores.size <= 1:
        return 0.5, compute_metrics(scores, y_true, 0.5)
    candidates = [(unique_scores[i] + unique_scores[i + 1]) / 2 for i in range(len(unique_scores) - 1)]
    best_thr = 0.5
    best_metrics = None
    best_f1 = -1.0
    for c in candidates:
        m = compute_metrics(scores, y_true, c)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = c
            best_metrics = m
    return float(best_thr), best_metrics


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main(features=None,
         labels=None,
         out_dir=None,
         lr=0.1,
         epochs=200,
         weight_decay=1e-4,
         seed=0,
         add_bias=False,
         verbose=False):

    # Resolve paths relative to BASE_DIR
    features = BASE_DIR / "data/features_processed.csv" if features is None else Path(features)
    labels   = BASE_DIR / "data/labels_processed.csv" if labels is None else Path(labels)
    out_dir  = BASE_DIR / "utils" if out_dir is None else Path(out_dir)

    names, Xraw, feature_names = load_features(features)
    labels_map = load_labels(labels)
    X, y, aligned_names = align_data(names, Xraw, labels_map)
    print(f"[main] aligned {len(aligned_names)} examples")

    thresholds, stats = compute_thresholds(X, y, feature_names)
    Z = prepare_normalized_X(X, thresholds, stats, feature_names, add_bias=add_bias)
    weights = train_gradient_descent(Z, y, lr=lr, epochs=epochs, weight_decay=weight_decay, seed=seed, verbose=verbose)

    scores = sigmoid(Z.dot(weights))
    best_thr, best_metrics = find_best_threshold(scores, y)

    weights_map = {fn: float(weights[i]) for i, fn in enumerate(feature_names)}
    if add_bias:
        weights_map["bias"] = float(weights[-1])

    save_json(out_dir / "feature_thresholds.json", thresholds)
    save_json(out_dir / "feature_weights.json", weights_map)

    report = {
        "n_examples": int(X.shape[0]),
        "best_score_threshold": float(best_thr),
        "best_threshold_metrics": best_metrics,
        "train_metrics_at_0.5": compute_metrics(scores, y, 0.5),
        "hyperparams": {"lr": lr, "epochs": epochs, "weight_decay": weight_decay, "bias": add_bias}
    }
    save_json(out_dir / "calibration_report.json", report)
    print("[main] done. Saved outputs to", str(out_dir))


if __name__ == "__main__":
    main(verbose=True)