# Mini-Project/model_free/classifier.py
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import math
import json
import csv
from pathlib import Path

EPS = 1e-9

# Calibrated thresholds (from your features calibration step)
DEFAULT_THRESHOLDS = {
    "sum_total_global": 901.4325866699219,
    "ratio_top_bottom": 0.48347145318984985,
    "center_of_mass_y": 0.5125085115432739,
    "avg_percent_above_threshold": 0.1775716170668602,
    "horizontal_vs_vertical_ratio": 1.1499664783477783,
    "max_overall": 1.0,
    "mean_of_means": 0.07335877045989037,
    "std_of_means": 0.009664916899055243,
}


def _get_feature_value(feature_vector, feature_names, name, default = 0.0):
    """
    Helper: get value by feature name from vector; returns default if not found.
    """
    try:
        idx = feature_names.index(name)
    except ValueError:
        return float(default)
    # ensure numpy float
    return float(np.array(feature_vector, dtype=np.float32)[idx])


def _sigmoid(x):
    """Numerically-stable sigmoid."""
    # clamp to avoid overflow
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def score_with_rule(feature_vector, feature_names, weights=None, verbose=False):
    """
    Compute a continuous score in [0,1] indicating support for 'raised',
    using multiple features instead of just 2.

    Parameters:
        feature_vector: 1D array-like
        feature_names: ordered list of names matching vector
        weights: dict mapping feature_name -> weight (optional)
        verbose: if True prints intermediate values

    Returns:
        score float in [0,1]
    """
    # Default weights for 6 most informative features
    default_weights = {
        "sum_total_global": 0.1165,
        "ratio_top_bottom": 0.1883,
        "center_of_mass_y": 0.3015,
        "avg_percent_above_threshold": 0.0,
        "horizontal_vs_vertical_ratio": 0.0931,
        "max_overall": 0.0,
        "mean_of_means": 0.1165,
        "std_of_means": 0.1166
    }

    if weights is None:
        weights = default_weights

    # Extract values for all used features
    feature_signals = {}
    for f in weights.keys():
        val = _get_feature_value(feature_vector, feature_names, f)
        # compute normalized signal relative to default threshold if exists
        if f in DEFAULT_THRESHOLDS:
            threshold = float(DEFAULT_THRESHOLDS[f])
            # signal: positive = supports 'raised'
            if f in ["center_of_mass_y"]:  # smaller = raised
                signal = (threshold - val) / (threshold + EPS)
            else:  # higher = raised
                signal = (val - threshold) / (threshold + EPS)
        else:
            signal = val  # fallback
        # clip extreme values
        feature_signals[f] = max(min(signal, 5.0), -5.0)

    # linear combination
    linear = sum(feature_signals[f] * weights[f] for f in weights)

    # scale before sigmoid for sharper separation
    scale = 3.0
    score = _sigmoid(scale * linear)

    if verbose:
        print("[score_with_rule multi-feature] signals:", feature_signals)
        print("[score_with_rule multi-feature] linear:", linear, "score:", score)

    return float(score)


def classify_pose_rule(feature_vector, feature_names, thresholds = None, weights = None, score_threshold = 0.5, verbose = False):
    """
    Classify a single feature_vector using a rule-based score.

    Parameters:
        feature_vector: 1D array-like
        feature_names: ordered list of feature names
        thresholds: dict of calibrated thresholds (optional)
        weights: weights passed to score_with_rule (optional)
        score_threshold: cutoff on score (>= => 'raised')
        verbose: prints debug info if True

    Returns:
        label: 'raised' or 'lowered'
    """
    score = score_with_rule(feature_vector, feature_names, weights=weights, verbose=verbose)

    label = "raised" if score >= score_threshold else "lowered"
    if verbose:
        print(f"[classify_pose_rule] score={score:.4f} => label={label}")
    return label


def predict_batch(feature_matrix, feature_names, thresholds = None, weights = None, score_threshold = 0.568, verbose = False):
    """
    Predict labels for a batch of feature vectors.

    Returns:
        (labels_list, scores_list)
    """
    X = np.array(feature_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("feature_matrix must be 2D (N, D)")

    labels = []
    scores = []
    for i in range(X.shape[0]):
        vec = X[i]
        score = score_with_rule(vec, feature_names, weights=weights, verbose=(verbose and i < 3))
        label = "raised" if score >= score_threshold else "lowered"
        labels.append(label)
        scores.append(float(score))
    return labels, scores


def evaluate(predictions, ground_truth, positive_label = "raised"):
    """
    Compute evaluation metrics and confusion matrix.

    Returns dict:
      {
        'accuracy': ...,
        'precision': ...,
        'recall': ...,
        'f1': ...,
        'confusion': {'TP':.., 'FP':.., 'FN':.., 'TN':..},
        'n': int
      }
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")

    preds = np.array(predictions, dtype=object)
    truths = np.array(ground_truth, dtype=object)

    pos = truths == positive_label
    neg = ~pos

    tp = int(((preds == positive_label) & pos).sum())
    fp = int(((preds == positive_label) & ~pos).sum())
    fn = int(((preds != positive_label) & pos).sum())
    tn = int(((preds != positive_label) & ~pos).sum())

    n = len(preds)
    accuracy = float((tp + tn) / (n + EPS))
    precision = float(tp / (tp + fp + EPS))
    recall = float(tp / (tp + fn + EPS))
    f1 = float(2 * precision * recall / (precision + recall + EPS))

    result = {
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    }
    return result


def save_report_csv(csv_path, metrics, extra_info = None, overwrite = False):
    """
    Save metrics (and optional extra_info) to a CSV file.
    CSV will contain simple key,value rows and also a confusion matrix table.
    """
    csv_p = Path(csv_path)
    if csv_p.exists() and not overwrite:
        raise FileExistsError(f"File exists: {csv_p}")

    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n", metrics.get("n", "")])
        writer.writerow(["accuracy", metrics.get("accuracy", "")])
        writer.writerow(["precision", metrics.get("precision", "")])
        writer.writerow(["recall", metrics.get("recall", "")])
        writer.writerow(["f1", metrics.get("f1", "")])
        # confusion matrix
        conf = metrics.get("confusion", {})
        writer.writerow([])
        writer.writerow(["confusion_matrix", "value"])
        writer.writerow(["TP", conf.get("TP", 0)])
        writer.writerow(["FP", conf.get("FP", 0)])
        writer.writerow(["FN", conf.get("FN", 0)])
        writer.writerow(["TN", conf.get("TN", 0)])
        # extra info (optional)
        if extra_info:
            writer.writerow([])
            writer.writerow(["extra_info", "value"])
            for k, v in extra_info.items():
                writer.writerow([k, json.dumps(v)])

def save_report_json(json_path, metrics, extra_info = None, overwrite = False):
    """
    Save metrics and optional extra_info to a JSON file.
    """
    p = Path(json_path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {"metrics": metrics}
    if extra_info:
        payload["extra_info"] = extra_info

    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)