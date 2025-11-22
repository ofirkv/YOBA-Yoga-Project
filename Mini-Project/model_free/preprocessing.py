# Mini-Project/model_free/preprocessing.py
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np

from .io_utils import load_image, list_images

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    If already single channel, return as-is.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Resize image to a fixed size.
    Size format: (width, height)
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to range [0, 1] as float32.
    """
    image = image.astype("float32")
    return image / 255.0


def preprocess_image(path_or_image: Union[str, Path, np.ndarray]) -> np.ndarray:
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


# -------------------- DEMO FUNCTION --------------------

def demo_preprocessing():
    """
    Demonstration function for validating preprocessing pipeline.
    Runs on 5 images from data/processed, prints stats and saves the processed images
    with names prcd_001.jpg, prcd_002.jpg, etc.
    """

    import cv2
    import numpy as np
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"

    images = list_images(processed_dir)
    sample = images[-5:]

    print("[DEMO] Running preprocessing on 5 images...")

    for idx, img_path in enumerate(sample, start=1):
        processed = preprocess_image(img_path)

        print(f"[DEMO] {img_path.name}")
        print(f"       shape: {processed.shape}")
        print(f"       min: {processed.min():.4f} | max: {processed.max():.4f}")
        print("-" * 40)

        # Convert from normalized float (0-1) to uint8 (0-255) for saving
        image_to_save = (processed * 255).astype(np.uint8)

        output_name = f"prcd_{idx:03}.jpg"
        output_path = processed_dir / output_name

        cv2.imwrite(str(output_path), image_to_save)


if __name__ == "__main__":
    demo_preprocessing()
