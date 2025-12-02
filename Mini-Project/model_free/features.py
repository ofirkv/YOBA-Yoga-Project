# Mini-Project/model_free/features.py
from pathlib import Path
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # מוסיף את תיקיית model_free ל-path

from io_utils import list_images, read_labels_csv
import conv_layer as conv_layer_module
import preprocessing as preprocessing_module

EPS = 1e-9

def _compute_center_of_mass_2d(map2d):
    """
    Compute center of mass of a 2D array.
    Returns normalized coordinates (y_c, x_c) in range [0, 1].
    If sum is zero, returns (0.5, 0.5).
    """
    if map2d.size == 0:
        return 0.5, 0.5

    total = float(map2d.sum())
    if total <= 0:
        return 0.5, 0.5

    H, W = map2d.shape
    # row indices (y) and column indices (x)
    rows = np.arange(H, dtype=np.float32).reshape(H, 1)
    cols = np.arange(W, dtype=np.float32).reshape(1, W)

    y_sum = float((map2d * rows).sum())
    x_sum = float((map2d * cols).sum())

    # normalize to [0,1]
    y_c = y_sum / (total * max(1, H - 1))
    x_c = x_sum / (total * max(1, W - 1))

    # clamp
    y_c = min(max(0.0, y_c), 1.0)
    x_c = min(max(0.0, x_c), 1.0)
    return y_c, x_c


def compute_per_map_stats(feature_map, threshold = None):
    """
    Compute statistics for a single feature map.

    Returns dict:
        {
          'sum_total': float,
          'mean': float,
          'std': float,
          'max': float,
          'percent_above_threshold': float,  # 0..1
          'center_of_mass': (y_c, x_c)    # normalized 0..1
        }

    Notes:
        - feature_map expected to be 2D float32 (after ReLU typically).
        - if threshold is None: threshold = max_value * 0.1 (or small EPS when max==0).
    """
    if feature_map.ndim != 2:
        raise ValueError("feature_map must be 2D")

    fmap = feature_map.astype(np.float32).copy()
    
    mean_signed = float(fmap.mean()) if fmap.size > 0 else 0.0

    abs_map = np.abs(fmap)


    max_abs = float(abs_map.max()) if abs_map.size > 0 else 0.0
    total = float(abs_map.sum())
    mean_abs = float(abs_map.mean()) if abs_map.size > 0 else 0.0
    std_abs = float(abs_map.std()) if abs_map.size > 0 else 0.0

    if threshold is None:
        threshold_local = max(0.1 * max_abs, EPS)
    else:
        threshold_local = float(threshold)
        

    if abs_map.size == 0:
        pct = 0.0
    else:
        cnt = float((abs_map > threshold_local).sum())
        pct = cnt / float(abs_map.size)

    y_c, x_c = _compute_center_of_mass_2d(fmap)

    return {
        "sum_total": total,
        "mean": mean_abs,
        "std": std_abs,
        "max": max_abs,
        "percent_above_threshold": pct,
        "center_of_mass": (y_c, x_c),
        "mean_signed": mean_signed,
    }


def aggregate_stats_across_maps(feature_maps, kernel_names = None, threshold = None):
    """
    Aggregate per-map stats and produce global aggregates.

    Output dict contains:
        - 'per_map': list of per-map dicts (as from compute_per_map_stats)
        - 'total_energy_per_kernel': list of floats
        - 'sum_total_global': float
        - 'mean_of_means', 'std_of_means'
        - 'max_overall'
        - 'avg_percent_above_threshold'
        - 'center_of_mass_global': (y_c, x_c) computed from combined map
        - optionally: 'vertical_energy','horizontal_energy','diagonal_energy','vertical_vs_horizontal'
    """
    if feature_maps.ndim != 3:
        raise ValueError("feature_maps must be 3D array (n_kernels, H, W)")

    n_k, H, W = feature_maps.shape
    per_map = []
    totals = []
    means = []
    percent_list = []
    max_list = []
    signed_means = []

    for i in range(n_k):
        pm = compute_per_map_stats(feature_maps[i], threshold=threshold)
        per_map.append(pm)
        totals.append(pm["sum_total"])
        means.append(pm["mean"])
        percent_list.append(pm["percent_above_threshold"])
        max_list.append(pm["max"])
        signed_means.append(pm["mean_signed"])

    totals = np.array(totals, dtype=np.float32)
    means = np.array(means, dtype=np.float32)
    max_list = np.array(max_list, dtype=np.float32)
    percent_list = np.array(percent_list, dtype=np.float32)
    signed_means = np.array(signed_means, dtype=np.float32)

    sum_total_global = float(totals.sum())
    mean_of_means = float(means.mean()) if means.size > 0 else 0.0
    std_of_means = float(means.std()) if means.size > 0 else 0.0
    max_overall = float(max_list.max()) if max_list.size > 0 else 0.0
    avg_percent_above = float(percent_list.mean()) if percent_list.size > 0 else 0.0

    # global center of mass computed from combined map (using abs energy)
    combined_map = feature_maps.sum(axis=0)
    com_global = _compute_center_of_mass_2d(np.abs(combined_map))

    result = {
        "per_map": per_map,
        "total_energy_per_kernel": totals.tolist(),
        "sum_total_global": sum_total_global,
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "max_overall": max_overall,
        "avg_percent_above_threshold": avg_percent_above,
        "center_of_mass_global": com_global,
        "H": H,
        "W": W,
    }

    kernel_direction = {
        "sobel_vertical": "vertical",
        "sobel_horizontal": "horizontal",
        "diagonal_main": "diagonal",
        "diagonal_anti": "diagonal",
        "laplacian": "other",
        "sharpen": "other",
        "identity": "other"
    }

    vertical_energy = 0.0
    horizontal_energy = 0.0
    diagonal_energy = 0.0
    other_energy = 0.0

    signed_weighted_vert = 0.0
    signed_weighted_horiz = 0.0

    for name, tot, signed_mean in zip(kernel_names, totals, signed_means):
        lname = str(name).lower()

        # use explicit mapping; default to "other"
        direction = kernel_direction.get(lname, "other")

        if direction == "vertical":
            vertical_energy += float(tot)
            signed_weighted_vert += float(signed_mean) * float(tot)

        elif direction == "horizontal":
            horizontal_energy += float(tot)
            signed_weighted_horiz += float(signed_mean) * float(tot)

        elif direction == "diagonal":
            diagonal_energy += float(tot)

        else:
            other_energy += float(tot)

    # Update result
    result.update({
        "vertical_energy": vertical_energy,
        "horizontal_energy": horizontal_energy,
        "diagonal_energy": diagonal_energy,
        "other_energy": other_energy
    })

    result["vertical_vs_horizontal"] = vertical_energy / (horizontal_energy + EPS)
    result["vertical_ratio_of_total"] = vertical_energy / (
        vertical_energy + horizontal_energy + diagonal_energy + other_energy + EPS
    )

    result["mean_signed_vertical"] = signed_weighted_vert / (vertical_energy + EPS)
    result["mean_signed_horizontal"] = signed_weighted_horiz / (horizontal_energy + EPS)

    return result


def feature_names_for_set():
    """
    Return ordered feature names for a given feature_set.
    Supported sets: 'default' (compact, 8 features), 'full' (per-map expanded).
    """
    return [
        "sum_total_global",
        "ratio_top_bottom",
        "center_of_mass_y",
        "avg_percent_above_threshold",
        "horizontal_vs_vertical_ratio",
        "max_overall",
        "mean_of_means",
        "std_of_means",
        "mean_signed_vertical",
        "mean_signed_horizontal",
    ]


def compute_feature_vector(feature_maps, kernel_names = None, threshold = None):
    """
    Compute a compact feature vector from stacked feature_maps.
    Returns: feature_vector (np.ndarray, dtype=float32), feature_names (List[str])
    """
    if feature_maps.ndim != 3:
        raise ValueError("feature_maps must be an array with shape (n_kernels, H, W)")

    agg = aggregate_stats_across_maps(feature_maps, kernel_names=kernel_names, threshold=threshold)

    H = int(agg.get("H", 0))
    combined = np.array(feature_maps.sum(axis=0), dtype=np.float32)
    sum_total = float(agg["sum_total_global"])
    sum_total_nonzero = sum_total if sum_total > EPS else EPS

    # ratio top2/bottom2 using 4 stripes (more robust than half)
    if H > 3:
        band = max(1, H // 4)
        top2 = float(combined[:2*band, :].sum())
        bottom2 = float(combined[-2*band:, :].sum())
    else:
        # fallback to simple half if too small
        half = H // 2 if H > 1 else 1
        top2 = float(combined[:half, :].sum())
        bottom2 = float(combined[half:, :].sum())

    ratio_top_bottom = top2 / (bottom2 + EPS)
    
    center_of_mass_y = float(agg["center_of_mass_global"][0]) if agg.get("center_of_mass_global") is not None else 0.5

    avg_pct = float(agg["avg_percent_above_threshold"])
    max_overall = float(agg["max_overall"])
    mean_of_means = float(agg["mean_of_means"])
    std_of_means = float(agg["std_of_means"])

    # horizontal_vs_vertical_ratio
    if "vertical_energy" in agg and "horizontal_energy" in agg:
        horiz = float(agg["horizontal_energy"])
        vert = float(agg["vertical_energy"])
        horizontal_vs_vertical_ratio = vert / max(horiz, EPS)
        horizontal_vs_vertical_ratio = np.clip(horizontal_vs_vertical_ratio, 0.0, 10.0)
    else:
        horizontal_vs_vertical_ratio = 0.0

    mean_signed_vertical = float(agg.get("mean_signed_vertical", 0.0))
    mean_signed_horizontal = float(agg.get("mean_signed_horizontal", 0.0))

    feature_names = feature_names_for_set()
    vec = np.array([
        sum_total,
        ratio_top_bottom,
        center_of_mass_y,
        avg_pct,
        horizontal_vs_vertical_ratio,
        max_overall,
        mean_of_means,
        std_of_means,
        mean_signed_vertical,
        mean_signed_horizontal
    ], dtype=np.float32)
    return vec, feature_names


def calibrate_thresholds(feature_matrix, labels, feature_names, positive_label = "raised", method = "percentile"):
    """
    Simple calibration helper to compute candidate thresholds per-feature.

    Supports method 'percentile' (median midpoint) and 'manual' (not implemented here).
    Returns dict: {feature_name: threshold}

    Note: For robust calibration use leaderboard / notebook and ROC-based search.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2D (N, D)")
    if len(labels) != feature_matrix.shape[0]:
        raise ValueError("labels length does not match number of rows in feature_matrix")

    labels_arr = np.array(labels, dtype=object)
    pos_mask = labels_arr == positive_label
    neg_mask = ~pos_mask

    thresholds = {}
    D = feature_matrix.shape[1]

    if method == "percentile":
        for j in range(D):
            pos_vals = feature_matrix[pos_mask, j] if pos_mask.any() else np.array([], dtype=np.float32)
            neg_vals = feature_matrix[neg_mask, j] if neg_mask.any() else np.array([], dtype=np.float32)
            if pos_vals.size == 0 or neg_vals.size == 0:
                # not enough data to calibrate this feature
                thresholds[feature_names[j]] = float(np.nan)
                continue
            pos_med = float(np.median(pos_vals))
            neg_med = float(np.median(neg_vals))
            # pick midpoint as candidate threshold
            thresholds[feature_names[j]] = float((pos_med + neg_med) / 2.0)
        return thresholds
    else:
        raise ValueError(f"Unknown calibration method: {method}")