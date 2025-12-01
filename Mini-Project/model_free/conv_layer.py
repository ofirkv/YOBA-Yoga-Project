# Mini-Project/model_free/conv_layer.py
import numpy as np

DEFAULT_CONV_CONFIG = {
    "pre_smooth": False,
    "pre_smooth_kernel": "gaussian_5x5",
    "apply_relu": True,
    "abs_before_relu": False,
    "apply_pool": False,
    "pool_size": (2, 2),
    "conv_kwargs": {
        "stride": 1,
        "padding": "same",
        "padding_mode": "constant"
    }
}

DEFAULT_IMPORTANT_KERNELS = [
    "sobel_vertical",
    "sobel_horizontal",
    "laplacian",
    "diagonal_main",
    "diagonal_anti"
]

KERNELS = {
# "Identity / pass-through filter. Useful for sanity checks."
    "identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32),

# "Sobel-like horizontal edge detector. Highlights horizontal gradients (changes in Y)."
    "sobel_horizontal": np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32),

# "Sobel-like vertical edge detector. Highlights vertical gradients (changes in X)."
    "sobel_vertical": np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ], dtype=np.float32),

# "Diagonal detector (main diagonal). Sensitive to ~45° oriented edges."
    "diagonal_main": np.array([
        [ 2,  1,  0],
        [ 1,  0, -1],
        [ 0, -1, -2]
    ], dtype=np.float32),

# "Diagonal detector (anti diagonal). Sensitive to ~135° oriented edges."
    "diagonal_anti": np.array([
        [ 0,  1,  2],
        [-1,  0,  1],
        [-2, -1,  0]
    ], dtype=np.float32),

# "Laplacian filter (second derivative). Highlights blobs / points of abrupt change."
    "laplacian": np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float32),

# "Sharpen filter - emphasizes high-frequency details."
    "sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),

# "Gaussian blur 5x5 (normalized). Use as pre-smoothing to reduce noise."
    "gaussian_5x5": np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float32) / 256.0

}

# Helper functions
def list_kernel_names():
    """Return a sorted list of available kernel names."""
    return sorted(KERNELS.keys())


def get_kernel(name):
    """
    Return the kernel array for the given name.
    Raises KeyError if name not found.
    """
    if name not in KERNELS:
        raise KeyError(f"Kernel not found: {name}")
    return KERNELS[name].copy()


# ==== Convolution (cross-correlation) implementation ====
def _normalize_stride(stride):
    if isinstance(stride, int):
        return (stride, stride)
    if isinstance(stride, tuple) and len(stride) == 2:
        return stride
    raise ValueError("stride must be int or tuple of two ints")


def apply_convolution(image, kernel, stride = 1, padding = "same", padding_mode = "constant"):
    """
    Parameters:
        image: 2D numpy array (H, W).
        kernel: 2D numpy array (kH, kW)
        stride: int or (sy, sx). Default 1.
        padding: 'same' or 'valid'. Default 'same'.
            - 'same' pads with floor(k/2) on each side (keeps H_out ~= H when stride=1)
            - 'valid' no padding
        padding_mode: passed to np.pad (e.g. 'constant', 'reflect', 'edge').

    Returns:
        feature_map: 2D numpy array (H_out, W_out), dtype float32
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError("input image must be 2D (grayscale)")

    img = image.astype(np.float32)
    k = kernel.astype(np.float32)
    kH, kW = k.shape
    H, W = img.shape
    sy, sx = _normalize_stride(stride)

    # Determine padding amounts
    if padding == "same":
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    elif padding == "valid":
        pad_h = pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding_mode)
    else:
        img_padded = img

    H_p, W_p = img_padded.shape

    # Compute output size
    out_h = (H_p - kH) // sy + 1
    out_w = (W_p - kW) // sx + 1

    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    out = np.zeros((out_h, out_w), dtype=np.float32)

    # Naive sliding-window (cross-correlation)
    for y in range(out_h):
        y0 = y * sy
        y1 = y0 + kH
        for x in range(out_w):
            x0 = x * sx
            x1 = x0 + kW
            window = img_padded[y0:y1, x0:x1]
            # element-wise multiplication and sum
            out[y, x] = np.sum(window * k)

    return out.astype(np.float32)


# === Activation functions ===
def activation_relu(feature_map, inplace = False, abs_before = False):
    """
    Apply ReLU activation: max(0, x)
    Parameters:
        feature_map: 2D numpy array
        inplace: whether to modify the input array in-place
        abs_before: if True, apply absolute value first (useful when negative responses also carry energy)
    Returns:
        activated_map: 2D numpy array (float32)
    Note: abs_before can be useful for Sobel-like outputs where direction is not important,
    and you want the magnitude of the response before zeroing negatives.
    """
    if not inplace:
        fm = feature_map.astype(np.float32).copy()
    else:
        fm = feature_map.astype(np.float32)

    if abs_before:
        fm = np.abs(fm)

    # ReLU
    np.maximum(fm, 0.0, out=fm)
    return fm.astype(np.float32)


# === Pooling functions ===
def max_pooling(feature_map, pool_size = (2, 2), stride = None, padding = "same"):
    """
    Parameters:
        feature_map: 2D numpy array (H, W)
        pool_size: (ph, pw)
        stride: if None -> stride = pool_size
        padding: 'valid' or 'same'
            - 'valid': ignore leftover pixels
            - 'same': pad with zeros so pooling windows cover entire map (pads at end)

    Returns:
        pooled_map: 2D numpy array
    """
    if feature_map.ndim != 2:
        raise ValueError("feature_map must be 2D")

    ph, pw = pool_size
    sy, sx = _normalize_stride(stride if stride else pool_size)
    H, W = feature_map.shape

    if padding == "same":
        # compute padding needed so that ceil(H / sy) windows fit
        out_h = int(np.ceil(H / sy))
        out_w = int(np.ceil(W / sx))
        pad_h_total = max(0, (out_h - 1) * sy + ph - H)
        pad_w_total = max(0, (out_w - 1) * sx + pw - W)
        
        fm = np.pad(feature_map,
                             ((pad_h_total // 2, pad_h_total - pad_h_total // 2),
                              (pad_w_total // 2, pad_w_total - pad_w_total // 2)),
                             mode="constant")
    elif padding == "valid":
        fm = feature_map
        out_h = (H - ph) // sy + 1
        out_w = (W - pw) // sx + 1
        if out_h <= 0 or out_w <= 0:
            return np.zeros((0, 0), dtype=np.float32)
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    Hf, Wf = fm.shape
    out_h = (Hf - ph) // sy + 1
    out_w = (Wf - pw) // sx + 1
    pooled = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        y0 = y * sy
        y1 = y0 + ph
        for x in range(out_w):
            x0 = x * sx
            x1 = x0 + pw
            window = fm[y0:y1, x0:x1]
            pooled[y, x] = np.max(window)

    return pooled.astype(np.float32)


# === Multiple kernels pipeline ===
def apply_multiple_kernels(image, kernels, config = DEFAULT_CONV_CONFIG):
    """
    Apply multiple kernels to `image` and return stacked feature maps.

    Parameters:
        image: 2D numpy array (grayscale)
        kernels: list of kernel names (strings from KERNELS) or numpy arrays
        config: dictionary with keys
    Returns:
        stacked_maps: numpy array with shape (n_kernels, H_out, W_out)
    """

    img = image.astype(np.float32).copy()

    if config["pre_smooth"]:
        img = apply_convolution(img, get_kernel(config["pre_smooth_kernel"]), **config["conv_kwargs"])

    maps = []
    for k in kernels:
        kernel = get_kernel(k)
        fmap = apply_convolution(img, kernel, **config["conv_kwargs"])

        if config["apply_relu"]:
            fmap = activation_relu(fmap, abs_before=config["abs_before_relu"])

        if config["apply_pool"]:
            fmap = max_pooling(fmap, pool_size=config["pool_size"])

        if fmap.max() > 0:
            fmap = fmap / fmap.max()

        maps.append(fmap)

    if len(maps) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # Stack into array shape (n_kernels, H_out, W_out)
    stacked = np.stack(maps, axis=0).astype(np.float32)
    return stacked


# === Convenience pipeline function ===
def process_image_with_kernels(image, kernels = None, config = DEFAULT_CONV_CONFIG):
    """
    High-level utility: run a simple conv pipeline:
      optional smoothing -> apply multiple kernels -> relu -> optional pooling
    Returns stacked feature maps.

    By default (use_default_kernels=True), uses ['sobel_vertical','sobel_horizontal','diagonal_main','laplacian'].
    """
    if kernels is None:
        kernels = DEFAULT_IMPORTANT_KERNELS

    return apply_multiple_kernels(image, kernels, config)