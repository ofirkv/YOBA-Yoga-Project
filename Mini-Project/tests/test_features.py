# Mini-Project/tests/test_features.py
from pathlib import Path
import sys
import numpy as np

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_free import features

def test_compute_per_map_stats_basic():
    """Test compute_per_map_stats with simple array"""
    fmap = np.array([[0, 1], [2, 3]], dtype=np.float32)
    stats = features.compute_per_map_stats(fmap)

    # fmap is normalized inside function
    fmap_norm = np.abs(fmap) / fmap.max()
    expected_sum = float(fmap_norm.sum())

    assert np.isclose(stats["sum_total"], expected_sum)
    assert 0 <= stats["percent_above_threshold"] <= 1
    y_c, x_c = stats["center_of_mass"]
    assert 0.0 <= y_c <= 1.0 and 0.0 <= x_c <= 1.0
    print("[TEST] compute_per_map_stats_basic passed")


def test_compute_per_map_stats_empty():
    """Test compute_per_map_stats with all-zero map"""
    fmap = np.zeros((1, 1), dtype=np.float32)  # 0x0 causes ValueError in np.max()
    stats = features.compute_per_map_stats(fmap)
    assert stats["sum_total"] == 0
    assert stats["center_of_mass"] == (0.5, 0.5)
    print("[TEST] compute_per_map_stats_empty passed")


def test_aggregate_stats_across_maps_basic():
    """Test aggregate_stats_across_maps with random maps"""
    maps = np.random.rand(3, 4, 4).astype(np.float32)
    kernel_names = ["vertical", "horizontal", "diag"]
    agg = features.aggregate_stats_across_maps(maps, kernel_names=kernel_names)
    
    assert "sum_total_global" in agg
    assert "center_of_mass_global" in agg
    assert agg["vertical_energy"] >= 0
    assert agg["horizontal_energy"] >= 0
    assert 0 <= agg["vertical_vs_horizontal"] <= 1e6  # ratio could be large if horiz small
    print("[TEST] aggregate_stats_across_maps_basic passed")


def test_compute_feature_vector_basic():
    """Test compute_feature_vector returns correct shape and names"""
    maps = np.random.rand(2, 4, 4).astype(np.float32)
    kernel_names = ["vertical", "horizontal"]
    vec, names = features.compute_feature_vector(maps, kernel_names=kernel_names)

    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == len(names) == 8
    assert np.all(np.isfinite(vec)), "feature vector contains non-finite values"
    print("[TEST] compute_feature_vector_basic passed")


def test_calibrate_thresholds_basic():
    """Test calibrate_thresholds with simple positive/negative data"""
    feature_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    labels = ["raised", "raised", "lowered", "lowered"]
    feature_names = ["feat1", "feat2"]
    thresholds = features.calibrate_thresholds(feature_matrix, labels, feature_names, positive_label="raised")
    
    assert set(thresholds.keys()) == set(feature_names)
    for v in thresholds.values():
        assert np.isfinite(v), "threshold should be finite"
    print("[TEST] calibrate_thresholds_basic passed")


if __name__ == "__main__":
    test_compute_per_map_stats_basic()
    test_compute_per_map_stats_empty()
    test_aggregate_stats_across_maps_basic()
    test_compute_feature_vector_basic()
    test_calibrate_thresholds_basic()
    print("[TEST] All feature module tests passed")
