# Mini-Project/model_free/features.py
from pathlib import Path
import csv
import numpy as np

# Local imports from the package
from .io_utils import list_images, read_labels_csv
from . import conv_layer as conv_layer_module  # used for default conv fn
from . import preprocessing as preprocessing_module  # used for default preprocess fn

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
    
    fmap = np.abs(fmap)

    if fmap.max() > 0:
        fmap /= fmap.max()

    total = float(fmap.sum())
    mean = float(fmap.mean()) if fmap.size > 0 else 0.0
    std = float(fmap.std()) if fmap.size > 0 else 0.0
    maxval = float(fmap.max()) if fmap.size > 0 else 0.0

    if threshold is None:
        threshold_local = max(0.1 * maxval, EPS)
    else:
        threshold_local = float(threshold)
        
    # percent above threshold
    if fmap.size == 0:
        pct = 0.0
    else:
        cnt = float((fmap > threshold_local).sum())
        pct = cnt / float(fmap.size)

    y_c, x_c = _compute_center_of_mass_2d(fmap)

    return {
        "sum_total": total,
        "mean": mean,
        "std": std,
        "max": maxval,
        "percent_above_threshold": pct,
        "center_of_mass": (y_c, x_c),
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

    for i in range(n_k):
        pm = compute_per_map_stats(feature_maps[i], threshold=threshold)
        per_map.append(pm)
        totals.append(pm["sum_total"])
        means.append(pm["mean"])
        percent_list.append(pm["percent_above_threshold"])
        max_list.append(pm["max"])

    totals = np.array(totals, dtype=np.float32)
    means = np.array(means, dtype=np.float32)
    max_list = np.array(max_list, dtype=np.float32)
    percent_list = np.array(percent_list, dtype=np.float32)

    sum_total_global = float(totals.sum())
    mean_of_means = float(means.mean()) if means.size > 0 else 0.0
    std_of_means = float(means.std()) if means.size > 0 else 0.0
    max_overall = float(max_list.max()) if max_list.size > 0 else 0.0
    avg_percent_above = float(percent_list.mean()) if percent_list.size > 0 else 0.0

    # global center of mass computed from combined map
    combined_map = feature_maps.sum(axis=0)
    com_global = _compute_center_of_mass_2d(combined_map)

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

    # If kernel_names provided, compute coarse directional energies
    if kernel_names is not None and len(kernel_names) == n_k:
        vertical_energy = 0.0
        horizontal_energy = 0.0
        diagonal_energy = 0.0
        other_energy = 0.0

        for name, tot in zip(kernel_names, totals):
            lname = str(name).lower()
            if "vertical" in lname or "sobel_v" in lname or "vert" in lname:
                vertical_energy += float(tot)
            elif "horizontal" in lname or "sobel_h" in lname or "horiz" in lname:
                horizontal_energy += float(tot)
            elif "diag" in lname or "diagonal" in lname:
                diagonal_energy += float(tot)
            else:
                other_energy += float(tot)

        result.update({
            "vertical_energy": vertical_energy,
            "horizontal_energy": horizontal_energy,
            "diagonal_energy": diagonal_energy,
            "other_energy": other_energy,
        })
        # ratio safely
        denom = horizontal_energy + EPS
        result["vertical_vs_horizontal"] = vertical_energy / denom
        result["vertical_ratio_of_total"] = vertical_energy / (vertical_energy + horizontal_energy + diagonal_energy + other_energy + EPS)

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
    # combined map for top/bottom computations
    combined = np.array(feature_maps.sum(axis=0), dtype=np.float32)
    sum_total = float(agg["sum_total_global"])
    sum_total_nonzero = sum_total if sum_total > EPS else EPS

    # ratio top/bottom
    half = H // 2 if H > 1 else 1
    top_energy = float(combined[:half, :].sum())
    ratio_top_bottom = top_energy / sum_total_nonzero

    center_of_mass_y = float(agg["center_of_mass_global"][0])  # normalized 0..1

    avg_pct = float(agg["avg_percent_above_threshold"])
    max_overall = float(agg["max_overall"])
    mean_of_means = float(agg["mean_of_means"])
    std_of_means = float(agg["std_of_means"])

    # horizontal_vs_vertical_ratio
    if "vertical_energy" in agg and "horizontal_energy" in agg:
        horiz = float(agg["horizontal_energy"])
        vert = float(agg["vertical_energy"])
        horizontal_vs_vertical_ratio = vert / (horiz + EPS)
    else:
        horizontal_vs_vertical_ratio = 0.0

    feature_names = feature_names_for_set()
    vec = np.array([
        sum_total,
        ratio_top_bottom,
        center_of_mass_y,
        avg_pct,
        horizontal_vs_vertical_ratio,
        max_overall,
        mean_of_means,
        std_of_means
    ], dtype=np.float32)
    return vec, feature_names


# def save_feature_vectors_csv(csv_path, feature_matrix, feature_names, filenames, labels = None,overwrite = False):
#     """
#     Save feature_matrix to CSV.
#     Header: filename,label,feat1,feat2,...
#     feature_matrix shape: (N, D); filenames length N
#     labels optional list length N
#     """
#     csv_p = Path(csv_path)
#     if csv_p.exists() and not overwrite:
#         raise FileExistsError(f"File exists: {csv_p}")

#     csv_p.parent.mkdir(parents=True, exist_ok=True)

#     with csv_p.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         header = ["filename"]
#         if labels is not None:
#             header.append("label")
#         header.extend(feature_names)
#         writer.writerow(header)

#         N = feature_matrix.shape[0]
#         for i in range(N):
#             row = [filenames[i]]
#             if labels is not None:
#                 row.append(labels[i])
#             row.extend([float(x) for x in feature_matrix[i]])
#             writer.writerow(row)


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