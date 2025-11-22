# Mini-Project/tests/test_preprocessing.py
from pathlib import Path
import numpy as np
import cv2
import sys

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_free.preprocessing import to_grayscale, resize, normalize, preprocess_image, list_images

def test_to_grayscale():
    """Test grayscale conversion"""
    base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    images = list_images(base_dir)
    if not images:
        print("[TEST] No images found in data/processed")
        return

    img = cv2.imread(str(images[0]))  # BGR image
    gray = to_grayscale(img)
    assert gray.ndim == 2, "to_grayscale failed: result should be 2D"
    print("[TEST] to_grayscale passed")


def test_resize():
    """Test resizing to 128x128"""
    dummy = np.zeros((200, 300), dtype=np.uint8)
    resized = resize(dummy)
    assert resized.shape == (128, 128), f"resize failed: got {resized.shape}"
    print("[TEST] resize passed")


def test_normalize():
    """Test normalization to 0..1"""
    dummy = np.array([[0, 128, 255]], dtype=np.uint8)
    norm = normalize(dummy)
    assert norm.min() >= 0 and norm.max() <= 1, "normalize failed: values not in [0,1]"
    print("[TEST] normalize passed")


def test_preprocess_image():
    """Test full preprocessing pipeline on first image"""
    base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    images = list_images(base_dir)
    if not images:
        print("[TEST] No images found in data/processed")
        return

    # Test with path
    processed = preprocess_image(images[0])
    assert processed.shape == (128, 128), f"preprocess_image failed: got {processed.shape}"
    assert 0 <= processed.min() <= processed.max() <= 1, "preprocess_image failed: values out of range"
    print("[TEST] preprocess_image with path passed")

    # Test with numpy array
    img = cv2.imread(str(images[0]))
    processed2 = preprocess_image(img)
    assert np.allclose(processed, processed2, atol=1e-6), "preprocess_image numpy input mismatch"
    print("[TEST] preprocess_image with numpy array passed")


if __name__ == "__main__":
    test_to_grayscale()
    test_resize()
    test_normalize()
    test_preprocess_image()
    print("[TEST] All preprocessing tests passed")
