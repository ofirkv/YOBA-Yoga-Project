# Mini-Project/model_free/preprocessing.py
from pathlib import Path

import cv2
import numpy as np

from .io_utils import load_image, list_images

def to_grayscale(image):
    """
    Convert BGR image to grayscale.
    If already single channel, return as-is.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image, size = (128, 128)):
    """
    Resize image to a fixed size.
    Size format: (width, height)
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image):
    """
    Normalize pixel values to range [0, 1] as float32.
    """
    image = image.astype("float32")
    return image / 255.0


def preprocess_image(path_or_image):
    """
    Full preprocessing pipeline:
    - load if path
    - grayscale
    - resize to 128x128
    - normalize to 0..1

    Returns processed numpy array (128,128).
    """
    if isinstance(path_or_image, (str, Path)):
        image = load_image(path_or_image, as_gray=False)
        if image is None:
            raise ValueError(f"Failed to load image: {path_or_image}")
    elif isinstance(path_or_image, np.ndarray):
        image = path_or_image
    else:
        raise TypeError("Input must be file path or numpy array")

    gray = to_grayscale(image)
    resized = resize(gray)
    normalized = normalize(resized)

    return normalized