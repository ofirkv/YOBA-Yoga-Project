#!/usr/bin/env python3
# calibrate_classifier.py
# Place in: Mini-Project/model_free/calibrate_classifier.py
#
# Usage:
#   cd Mini-Project
#   python model_free/calibrate_classifier.py
#
# Output files (saved into model_free/):
#   - calibration_results.json
#   - feature_thresholds.json
#   - feature_weights.json
#   - feature_stats.json

from pathlib import Path
import json
import math
import numpy as np
import csv
import sys

# numeric stability
EPS = 1e-9

# project paths (relative)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_CSV = DATA_DIR / "features_processed.csv"
LABELS_CSV = DATA_DIR / "labels_processed.csv"

OUT_DIR = Path(__file__).resolve().parent
OUT_THRESHOLDS = OUT_DIR / "feature_thresholds.json"
OUT_WEIGHTS = OUT_DIR / "feature_weights.json"
OUT_STATS = OUT_DIR / "feature_stats.json"
OUT_CALIB = OUT_DIR / "calibration_results.json"

# Try optional sklearn utilities; fallback if not present
try:
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------
# Helper I/O
# ---------------------------------------------------------------------
def read_features_csv(path):
    """
    Read feature CSV with header. Return (filenames, X (N,D), feature_names).
    Expect header: filename,feat1,feat2,...
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features CSV not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2 or header[0].lower() != "filename":
            raise ValueError("Features CSV must have 'filename' as first column")
        feature_names = header[1:]
        names = []
        rows = []
        for row in reader:
            if len(row) < len(feature_names) + 1:
                continue
            names.append(row[0])
            vals = [float(x) for x in row[1:1+len(feature_names)]]
            rows.append(vals)
    X = np.array(rows, dtype=np.float32)
    return names, X, feature_names

def read_labels_csv(path):
    """
    Read labels CSV with header filename,label
    Returns dict filename->label
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename") or row.get("file") or row.get("name")
            lab = row.get("label") or ""
            if fn:
                mapping[fn] = lab
    return mapping

# ---------------------------------------------------------------------
# Per-feature statistics & thresholds
# ---------------------------------------------------------------------
def median_midpoint_threshold(pos_vals, neg_vals):
    """Return midpoint of medians (pos_med + neg_med)/2"""
    if pos_vals.size == 0 or neg_vals.size == 0:
        return float("nan")
    return float((np.median(pos_vals) + np.median(neg_vals)) / 2.0)

def youden_threshold_one_feature(vals, labels_bin):
    """
    If sklearn available: compute ROC and Youden index threshold maximizing TPR-FPR.
    labels_bin should be 1 for positive (raised), 0 for negative.
    Returns (best_threshold, auc_val)
    """
    if not SKLEARN_AVAILABLE:
        return float("nan"), float("nan")
    try:
        fpr, tpr, thresholds = roc_curve(labels_bin, vals)
    except Exception:
        # if constant values or invalid
        return float("nan"), float("nan")
    youden = tpr - fpr
    idx = np.nanargmax(youden)
    best_thr = thresholds[idx]
    auc_val = auc(fpr, tpr)
    return float(best_thr), float(auc_val)

# ---------------------------------------------------------------------
# Weights computation
# ---------------------------------------------------------------------
def compute_cohens_d(pos_vals, neg_vals):
    """Cohen's d absolute value (using pooled std)"""
    if pos_vals.size == 0 or neg_vals.size == 0:
        return 0.0
    m1 = pos_vals.mean()
    m2 = neg_vals.mean()
    s1 = pos_vals.std(ddof=1) if pos_vals.size > 1 else 0.0
    s2 = neg_vals.std(ddof=1) if neg_vals.size > 1 else 0.0
    pooled = math.sqrt((s1**2 + s2**2) / 2.0 + EPS)
    d = (m1 - m2) / pooled
    return float(abs(d))

def compute_mutual_info(X, y_bin):
    """Return mutual information per feature (sklearn), fallback zeros."""
    if not SKLEARN_AVAILABLE:
        return np.zeros(X.shape[1], dtype=np.float32)
    try:
        mi = mutual_info_classif(X, y_bin, discrete_features=False, random_state=0)
        return np.array(mi, dtype=np.float32)
    except Exception:
        return np.zeros(X.shape[1], dtype=np.float32)

# ---------------------------------------------------------------------
# Score computation (mimic classifier logic)
# ---------------------------------------------------------------------
def normalized_signal(val, thr, invert=False, std=None):
    """
    Normalize a single feature to a stable signal.
    If std provided and >0 -> (val - thr)/std (or inverted)
    else -> (val - thr)/(abs(thr)+EPS)
    clip to [-10, 10]
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

def score_linear_sigmoid(X_vec, feature_names, thresholds, weights, stds=None, invert_map=None, scale=3.0):
    """
    Compute score for a single sample vector X_vec (1D numpy).
    thresholds: dict feature->thr
    weights: dict feature->weight (not necessarily normalized)
    stds: dict feature->std
    invert_map: dict feature->bool
    """
    linear = 0.0
    for f, w in weights.items():
        if f not in feature_names:
            continue
        val = float(X_vec[feature_names.index(f)])
        thr = float(thresholds.get(f, 0.0))
        std_val = stds.get(f, None) if stds is not None else None
        invert = invert_map.get(f, False) if invert_map is not None else False
        sig = normalized_signal(val, thr, invert=invert, std=std_val)
        linear += sig * float(w)
    s = 1.0 / (1.0 + math.exp(-scale * linear)) if math.isfinite(linear) else 0.0
    return float(s)

# ---------------------------------------------------------------------
# Utilities: evaluation
# ---------------------------------------------------------------------
def evaluate_metrics_from_preds(preds, truths, positive_label="raised"):
    """
    preds and truths are arrays of strings
    Returns dict with n, accuracy, precision, recall, f1, confusion
    """
    preds = np.array(preds, dtype=object)
    truths = np.array(truths, dtype=object)
    pos = truths == positive_label
    tp = int(((preds == positive_label) & pos).sum())
    fp = int(((preds == positive_label) & ~pos).sum())
    fn = int(((preds != positive_label) & pos).sum())
    tn = int(((preds != positive_label) & ~pos).sum())
    n = len(preds)
    acc = float((tp + tn) / (n + EPS))
    prec = float(tp / (tp + fp + EPS))
    rec = float(tp / (tp + fn + EPS))
    f1 = float(2 * prec * rec / (prec + rec + EPS))
    return {
        "n": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    }

# ---------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------
def calibrate_from_csv(features_csv=FEATURES_CSV, labels_csv=LABELS_CSV, positive_label="raised",
                       method_threshold="youden_vs_midpoint", weight_method="combined", feature_subset=None, verbose=True):
    """
    Main function: reads CSVs, computes thresholds, weights, stds, and finds best score threshold.
    Returns a dict with all outputs and saves JSON files.
    """
    names, X, feature_names = read_features_csv(features_csv)
    labels_map = read_labels_csv(labels_csv)

    # align labels
    y = []
    valid_idx = []
    valid_names = []
    for i, fn in enumerate(names):
        if fn in labels_map:
            y.append(labels_map[fn])
            valid_idx.append(i)
            valid_names.append(fn)
    if len(valid_idx) == 0:
        raise RuntimeError("No labeled samples found after alignment.")
    X = X[valid_idx, :]
    y = np.array(y, dtype=object)

    if feature_subset is None:
        feature_subset = list(feature_names)  # use all by default

    # compute per-feature pos/neg arrays
    pos_mask = (y == positive_label)
    neg_mask = ~pos_mask
    y_bin = (y == positive_label).astype(int)

    # compute basic stats
    stats = {}
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        pos_vals = col[pos_mask]
        neg_vals = col[neg_mask]
        stats[fname] = {
            "mean_pos": float(np.nan if pos_vals.size == 0 else pos_vals.mean()),
            "mean_neg": float(np.nan if neg_vals.size == 0 else neg_vals.mean()),
            "std_all": float(np.nan if col.size == 0 else col.std()),
            "std_pos": float(np.nan if pos_vals.size == 0 else pos_vals.std()),
            "std_neg": float(np.nan if neg_vals.size == 0 else neg_vals.std()),
        }

    # compute candidate thresholds per-feature
    thresholds_mid = {}
    thresholds_youden = {}
    aucs = {}
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        pos_vals = col[pos_mask]
        neg_vals = col[neg_mask]

        thresholds_mid[fname] = median_midpoint_threshold(pos_vals, neg_vals)

        if SKLEARN_AVAILABLE:
            thr_you, auc_val = youden_threshold_one_feature(col, y_bin)
            thresholds_youden[fname] = thr_you
            aucs[fname] = auc_val
        else:
            thresholds_youden[fname] = float("nan")
            aucs[fname] = float("nan")

    # pick per-feature threshold: heuristics
    thresholds_chosen = {}
    for fname in feature_names:
        mid = thresholds_mid.get(fname, float("nan"))
        you = thresholds_youden.get(fname, float("nan"))
        auc_val = aucs.get(fname, float("nan"))

        # choose Youden when AUC suggests separability > 0.55 and youden is finite
        if not math.isnan(auc_val) and auc_val > 0.55 and not math.isnan(you):
            thresholds_chosen[fname] = float(you)
        else:
            thresholds_chosen[fname] = float(mid)

    # compute weights
    # method: combined of Cohen's d and mutual information
    diffs = {}
    cohens = {}
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        pos_vals = col[pos_mask]
        neg_vals = col[neg_mask]
        coh = compute_cohens_d(pos_vals, neg_vals)
        cohens[fname] = coh
        diffs[fname] = abs(coh)

    mi = compute_mutual_info(X[:, [feature_names.index(f) for f in feature_names]], y_bin)
    mi_map = {f: float(mi[i]) for i, f in enumerate(feature_names)}

    # normalize components
    coh_vals = np.array([cohens[f] for f in feature_names], dtype=np.float32)
    mi_vals = np.array([mi_map[f] for f in feature_names], dtype=np.float32)

    def norm_arr(a):
        a = np.array(a, dtype=np.float32)
        s = a.sum()
        if s == 0:
            return np.zeros_like(a)
        return a / (s + EPS)

    coh_n = norm_arr(coh_vals)
    mi_n = norm_arr(mi_vals)

    combined_score = 0.6 * coh_n + 0.4 * mi_n  # weight Cohen's d more
    # produce final weights (only for features in subset)
    raw_weights = {f: float(combined_score[i]) for i, f in enumerate(feature_names)}
    # restrict to feature_subset and renormalize
    subset_weights = {f: raw_weights.get(f, 0.0) for f in feature_subset}
    total = sum(subset_weights.values()) + EPS
    subset_weights = {f: float(subset_weights[f] / total) for f in subset_weights}

    # compute stds map for normalization in scoring
    stds_map = {f: float(stats[f]["std_all"]) for f in feature_names}

    # invert_map guess (heuristic): if mean_pos < mean_neg -> smaller means 'raised'
    invert_map = {}
    for f in feature_names:
        mp = stats[f]["mean_pos"]
        mn = stats[f]["mean_neg"]
        invert_map[f] = False
        if not math.isnan(mp) and not math.isnan(mn):
            if mp < mn:
                invert_map[f] = True

    # compute scores for all samples using the subset_weights and chosen thresholds
    scores = []
    for i in range(X.shape[0]):
        vec = X[i]
        sc = score_linear_sigmoid(vec, feature_names, thresholds_chosen, subset_weights, stds=stds_map, invert_map=invert_map, scale=3.0)
        scores.append(sc)
    scores = np.array(scores, dtype=np.float32)

    # find best score threshold (sweep)
    if SKLEARN_AVAILABLE:
        # use unique midpoints between sorted scores as candidates
        uniq = np.unique(np.sort(scores))
        thr_candidates = []
        if uniq.size == 1:
            thr_candidates = [float(uniq[0])]
        else:
            thr_candidates = [float((uniq[i] + uniq[i+1]) / 2.0) for i in range(len(uniq)-1)]
    else:
        thr_candidates = list(np.linspace(0.0, 1.0, 101))

    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = None
    best_preds = None
    for thr in thr_candidates:
        preds = np.where(scores >= thr, "raised", "lowered")
        m = evaluate_metrics_from_preds(preds, y, positive_label=positive_label)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
            best_metrics = m
            best_preds = preds

    # results object
    results = {
        "feature_names": list(feature_names),
        "feature_stats": stats,
        "thresholds_midpoint": thresholds_mid,
        "thresholds_youden": thresholds_youden,
        "thresholds_chosen": thresholds_chosen,
        "aucs": aucs,
        "raw_weights_cohen": {f: cohens[f] for f in feature_names},
        "raw_weights_mi": mi_map,
        "combined_raw_weights": raw_weights,
        "weights_subset": subset_weights,
        "stds": stds_map,
        "invert_map": invert_map,
        "scores": scores.tolist(),
        "best_score_threshold": best_thr,
        "best_metrics": best_metrics,
        "n_samples": int(X.shape[0]),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum())
    }

    # Save JSON outputs (compact)
    with OUT_THRESHOLDS.open("w", encoding="utf-8") as f:
        json.dump(results["thresholds_chosen"], f, indent=2, ensure_ascii=False)
    with OUT_WEIGHTS.open("w", encoding="utf-8") as f:
        json.dump(results["weights_subset"], f, indent=2, ensure_ascii=False)
    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(results["feature_stats"], f, indent=2, ensure_ascii=False)
    with OUT_CALIB.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose:
        print("Calibration finished.")
        print(f"Samples: {results['n_samples']}, pos: {results['n_pos']}, neg: {results['n_neg']}")
        print("Chosen thresholds (per-feature):")
        for k, v in results["thresholds_chosen"].items():
            print(f"  {k}: {v}")
        print("\nWeights (subset):")
        for k, v in results["weights_subset"].items():
            print(f"  {k}: {v:.4f}")
        print("\nBest score threshold (max F1):", results["best_score_threshold"])
        print("Best metrics:", results["best_metrics"])
        print("\nSaved files:")
        print(" -", OUT_THRESHOLDS)
        print(" -", OUT_WEIGHTS)
        print(" -", OUT_STATS)
        print(" -", OUT_CALIB)

    return results

# ---------------------------------------------------------------------
# CLI run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # optional: accept arguments (not necessary)
    try:
        print("Starting calibration...")
        calibrate_from_csv(features_csv=FEATURES_CSV, labels_csv=LABELS_CSV, positive_label="raised", feature_subset=None)
    except Exception as e:
        print("Calibration error:", e)
        raise
