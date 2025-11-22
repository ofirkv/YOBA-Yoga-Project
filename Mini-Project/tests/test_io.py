# Mini-Project/tests/test_io.py
from pathlib import Path
import sys

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_free.io_utils import ensure_dir, list_images, load_image, validate_image_file, get_image_shape, collect_dataset_summary, read_labels_csv

BASE_DIR = PROJECT_ROOT
RAW_DIR = BASE_DIR / "data" / "processed"
CSV_PATH = BASE_DIR / "data" / "labels_processed.csv"


def test_ensure_dir():
    test_dir = BASE_DIR / "data" / "temp_test_dir"
    ensure_dir(test_dir)
    assert test_dir.exists() and test_dir.is_dir()
    print("[TEST] ensure_dir passed")


def test_list_images():
    images = list_images(RAW_DIR)
    assert len(images) > 0
    print(f"[TEST] list_images found {len(images)} images")


def test_load_image():
    images = list_images(RAW_DIR)
    img = load_image(images[0])
    assert img is not None
    print("[TEST] load_image passed")


def test_validate_image_file():
    images = list_images(RAW_DIR)
    valid = validate_image_file(images[0])
    assert valid is True
    print("[TEST] validate_image_file passed")


def test_get_image_shape():
    images = list_images(RAW_DIR)
    shape = get_image_shape(images[0])
    assert shape is not None and len(shape) == 3
    print(f"[TEST] get_image_shape returned {shape}")


def test_read_labels_csv():
    mapping = read_labels_csv(CSV_PATH)
    assert isinstance(mapping, dict)
    print(f"[TEST] read_labels_csv loaded {len(mapping)} entries")


def test_collect_dataset_summary():
    summary = collect_dataset_summary(RAW_DIR, CSV_PATH)
    assert "total" in summary
    assert "labeled" in summary
    assert "unlabeled" in summary
    assert "invalid" in summary
    print(f"[TEST] collect_dataset_summary: {summary}")


def run_all_tests():
    print("[TEST] Starting io_utils tests...")
    test_ensure_dir()
    test_list_images()
    test_load_image()
    test_validate_image_file()
    test_get_image_shape()
    test_read_labels_csv()
    test_collect_dataset_summary()
    print("[TEST] All tests passed successfully")


if __name__ == "__main__":
    run_all_tests()
