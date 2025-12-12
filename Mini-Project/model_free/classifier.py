# Mini-Project/model_free/classifier.py
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import math
import json
import csv
from pathlib import Path

EPS = 1e-9

DEFAULT_THRESHOLDS = {
  "sum_total_global": 9430.701171875,
  "ratio_top_bottom": 0.8906041383743286,
  "center_of_mass_y": 0.5348625779151917,
  "avg_percent_above_threshold": 0.24429841339588165,
  "horizontal_vs_vertical_ratio": 1.203217089176178,
  "max_overall": 1.0,
  "mean_of_means": 0.08222918957471848,
  "std_of_means": 0.06356267631053925,
  "mean_signed_vertical": 0.04727532900869846,
  "mean_signed_horizontal": 0.0425900686532259
}

DEFAULT_WEIGHTS = {
  "sum_total_global": 0.05778217315673828,
  "ratio_top_bottom": -0.4423084557056427,
  "center_of_mass_y": -0.7852282524108887,
  "avg_percent_above_threshold": -0.1377377063035965,
  "horizontal_vs_vertical_ratio": -0.03750080615282059,
  "max_overall": -0.009753220714628696,
  "mean_of_means": 0.049660809338092804,
  "std_of_means": -0.029878200963139534,
  "mean_signed_vertical": 0.28921234607696533,
  "mean_signed_horizontal": 0.1721014529466629
}

SCORE_THRESHOLD = 0.4259198307991028

GOOD_FEATURES = [
  "sum_total_global",
  "ratio_top_bottom",
  "center_of_mass_y",
  "avg_percent_above_threshold",
  "horizontal_vs_vertical_ratio",
  "max_overall",
  "mean_of_means",
  "std_of_means",
  "mean_signed_vertical",
  "mean_signed_horizontal"
]

# Features where smaller value implies 'raised' (invert signal)
INVERT_FEATURES = {
    "center_of_mass_y": True,
    "horizontal_vs_vertical_ratio": True, 
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


def _normalized_signal(val, thr, invert = False, std = None):
    """
    Normalize a single feature value into a stable signal.
    - If std is provided: use (val - thr) / std
    - Else: use (val - thr) / (abs(thr) + EPS)
    - If invert=True: smaller val => stronger support for 'raised'
    Returns clipped float.
    """
    if std is not None and std > 0:
        base = std
    else:
        base = abs(thr) + EPS

    if invert:
        s = (thr - val) / base
    else:
        s = (val - thr) / base

    return float(max(min(s, 10.0), -10.0))


def compute_feature_stats(feature_matrix, labels, feature_names, positive_label = "raised"):
    """
    Compute per-feature statistics: mean_pos, mean_neg, std_all.
    Returns dict mapping feature_name -> {"mean_pos":..., "mean_neg":..., "std":...}
    """
    X = np.array(feature_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("feature_matrix must be 2D (N, D)")

    labels_arr = np.array(labels, dtype=object)
    pos_mask = labels_arr == positive_label
    neg_mask = ~pos_mask

    stats = {}
    D = X.shape[1]
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        mean_pos = float(col[pos_mask].mean()) if pos_mask.any() else float(np.nan)
        mean_neg = float(col[neg_mask].mean()) if neg_mask.any() else float(np.nan)
        std_all = float(col.std()) if col.size > 0 else float(np.nan)
        stats[fname] = {"mean_pos": mean_pos, "mean_neg": mean_neg, "std": std_all}
    return stats


def compute_weights_mean_diff(feature_matrix, labels, feature_names, positive_label = "raised", features_subset = None):
    """
    Compute simple weights = |mean_pos - mean_neg| for each feature.
    Returns dict feature_name -> weight (normalized to sum 1).
    If features_subset provided, only compute weights for those features.
    """
    stats = compute_feature_stats(feature_matrix, labels, feature_names, positive_label=positive_label)
    diffs = {}
    for fname in (features_subset if features_subset is not None else feature_names):
        s = stats.get(fname, {})
        mp = s.get("mean_pos", np.nan)
        mn = s.get("mean_neg", np.nan)
        if np.isnan(mp) or np.isnan(mn):
            diffs[fname] = 0.0
        else:
            diffs[fname] = abs(mp - mn)

    total = float(sum(diffs.values())) + EPS
    weights = {k: float(v / total) for k, v in diffs.items()}
    return weights


def _normalized_default_weights(feature_subset):
    w = {f: float(DEFAULT_WEIGHTS.get(f, 0.0)) for f in feature_subset}
    total = sum(w.values())
    if total <= 0:
        return {f: 1.0 / len(feature_subset) for f in feature_subset}
    return {f: (w[f] / total) for f in feature_subset}

def score_with_rule(feature_vector, feature_names, feature_subset = None, scale = 3.0, verbose = False):
    """
    Compute a continuous score in [0,1] indicating support for 'raised'.
    - feature_vector: 1D array
    - feature_names: list of names matching vector
    - weights: dict mapping feature_name -> weight (if None, uses equal weights on feature_subset or GOOD_FEATURES)
    - thresholds: dict of per-feature thresholds (DEFAULT_THRESHOLDS used if None)
    - stds: optional dict of per-feature std to use in normalization
    - feature_subset: list of features to include (defaults to GOOD_FEATURES)
    - scale: multiplier before sigmoid to control sharpness
    """
    if feature_subset is None:
        feature_subset = GOOD_FEATURES.copy()

    weights = _normalized_default_weights(feature_subset)
    thresholds = DEFAULT_THRESHOLDS

    signals = {}
    for f in feature_subset:
        val = _get_feature_value(feature_vector, feature_names, f)
        thr = float(thresholds.get(f, 0.0))
        invert = INVERT_FEATURES.get(f, False)
        sig = _normalized_signal(val, thr, invert=invert, std=None)
        signals[f] = sig

    linear = 0.0
    for f in feature_subset:
        w = float(weights.get(f, 0.0))
        linear += signals[f] * w

    score = _sigmoid(scale * linear)

    if verbose:
        print("[score_with_rule] subset:", feature_subset)
        print("[score_with_rule] signals:", signals)
        print("[score_with_rule] linear:", linear, "score:", score)

    return float(score)


def find_best_score_threshold(scores, labels, positive_label = "raised"):
    """
    Given array of continuous scores and ground-truth labels, sweep thresholds
    and find threshold that maximizes F1. Returns (best_threshold, metrics_dict).
    Metrics dict contains accuracy/precision/recall/f1 for the best threshold.
    """
    s = np.array(scores, dtype=np.float32)
    y = np.array(labels, dtype=object)
    uniq = np.unique(np.sort(s))
    if uniq.size == 1:
        thr_candidates = [float(uniq[0])]
    else:
        thr_candidates = [(uniq[i] + uniq[i+1]) / 2.0 for i in range(uniq.size - 1)]
        thr_candidates = [float(t) for t in thr_candidates]

    best_thr = thr_candidates[0] if thr_candidates else 0.5
    best_f1 = -1.0
    best_metrics = {}
    for thr in thr_candidates:
        preds = np.where(s >= thr, "raised", "lowered")
        res = evaluate(preds.tolist(), labels, positive_label=positive_label)
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_thr = thr
            best_metrics = res
    return float(best_thr), best_metrics


def classify_pose_rule(feature_vector, feature_names, score_threshold = SCORE_THRESHOLD, verbose = False):
    """
    Classify a single feature_vector using a rule-based score.
    If weights is None, score_with_rule will use default GOOD_FEATURES with equal weights.
    """
    score = score_with_rule(feature_vector, feature_names, verbose=verbose)
    label = "raised" if score >= score_threshold else "lowered"
    if verbose:
        print(f"[classify_pose_rule] score={score:.4f} => label={label}")
    return label


def predict_batch(feature_matrix, feature_names, score_threshold = SCORE_THRESHOLD, verbose = False):
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
        score = score_with_rule(vec, feature_names, verbose=(verbose and i < 3))
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