# Mini-Project/tests/test_conv.py
from pathlib import Path
import sys
import numpy as np

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_free import conv_layer as conv

def test_list_and_get_kernel():
    """Test listing kernels and get_kernel"""
    kernels = conv.list_kernel_names()
    assert "identity" in kernels, "identity kernel missing"
    assert "sobel_vertical" in kernels, "sobel_vertical kernel missing"

    k = conv.get_kernel("identity")
    assert isinstance(k, np.ndarray), "get_kernel should return np.ndarray"
    assert k.shape == (3, 3), "identity kernel shape mismatch"

    try:
        conv.get_kernel("non_existing_kernel")
    except KeyError:
        pass
    else:
        assert False, "get_kernel should raise KeyError for unknown kernel"
    print("[TEST] list_and_get_kernel passed")


def test_apply_convolution_identity():
    """Test apply_convolution with identity kernel"""
    img = np.random.rand(8, 8).astype(np.float32)
    identity_kernel = conv.get_kernel("identity")
    out = conv.apply_convolution(img, identity_kernel, padding="same")
    assert out.shape == img.shape, "output shape mismatch"
    # center pixel should match original
    center_val = out[4, 4]
    assert np.isclose(center_val, img[4, 4], atol=1e-5), "center pixel mismatch"
    print("[TEST] apply_convolution_identity passed")


def test_activation_relu_abs():
    """Test activation_relu with and without abs_before"""
    fm = np.array([[-1, -0.5, 0], [0.5, 1, -2]], dtype=np.float32)

    relu_only = conv.activation_relu(fm, abs_before=False)
    assert np.all(relu_only >= 0), "ReLU failed: negative values remain"
    assert relu_only[0, 0] == 0, "ReLU failed on -1"

    relu_abs = conv.activation_relu(fm, abs_before=True)
    assert np.all(relu_abs >= 0), "ReLU abs_before failed: negative values remain"
    assert relu_abs[0, 0] == 1.0, "ReLU abs_before incorrect value"
    assert relu_abs[1, 2] == 2.0, "ReLU abs_before incorrect value"
    print("[TEST] activation_relu_abs passed")


def test_max_pooling_basic():
    """Test max_pooling"""
    fm = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 1],
        [7, 8, 9, 2],
        [0, 1, 2, 3]
    ], dtype=np.float32)

    pooled = conv.max_pooling(fm, pool_size=(2, 2), stride=(2, 2), padding="valid")
    expected = np.array([[5, 6], [8, 9]], dtype=np.float32)
    assert np.allclose(pooled, expected), "max_pooling output mismatch"
    print("[TEST] max_pooling_basic passed")


def test_apply_multiple_kernels_output_shape():
    """Test apply_multiple_kernels"""
    img = np.random.rand(16, 16).astype(np.float32)
    kernels = ["identity", "sobel_horizontal", "laplacian"]

    stacked = conv.apply_multiple_kernels(img, kernels)
    assert stacked.shape[0] == len(kernels), "number of stacked maps mismatch"
    H_out, W_out = stacked.shape[1:]
    assert H_out <= img.shape[0] and W_out <= img.shape[1], "output size too large"
    assert stacked.dtype == np.float32, "dtype should be float32"
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0, "values not normalized"
    print("[TEST] apply_multiple_kernels_output_shape passed")


def test_process_image_with_kernels_default():
    """Test process_image_with_kernels default"""
    img = np.random.rand(16, 16).astype(np.float32)
    out = conv.process_image_with_kernels(img)
    assert out.shape[0] == len(conv.DEFAULT_IMPORTANT_KERNELS), "default kernels mismatch"
    assert out.min() >= 0.0 and out.max() <= 1.0, "values out of range"
    print("[TEST] process_image_with_kernels_default passed")


if __name__ == "__main__":
    test_list_and_get_kernel()
    test_apply_convolution_identity()
    test_activation_relu_abs()
    test_max_pooling_basic()
    test_apply_multiple_kernels_output_shape()
    test_process_image_with_kernels_default()
    print("[TEST] All convolution tests passed")
