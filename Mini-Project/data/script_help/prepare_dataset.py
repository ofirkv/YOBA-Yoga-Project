# Mini-Project/data/prepare_dataset.py
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# === Global setup ===
mp_hands = mp.solutions.hands


def load_image(path: Path):
    """Load image with OpenCV and auto-fix EXIF rotation if needed."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"Warning: Could not read image: {path}")
        return None

    # Convert BGR to RGB for Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def detect_hand(img_rgb):
    """Run Mediapipe Hands and return True if at least one hand is detected."""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            return True
        return False


def ensure_orientation(img):
    """
    Fix images that have wrong orientation.
    Checks dimensions only — most common issue in raw datasets.
    If height >> width, likely rotated 90 degrees.
    """
    h, w = img.shape[:2]

    # If image is extremely vertical → assume incorrect rotation
    if h > w * 1.6:  
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif w > h * 1.6:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def process_image(src_path: Path, dst_path: Path):
    """Full pipeline for a single image."""
    original, img_rgb = load_image(src_path)
    if original is None:
        return False

    # Fix orientation
    fixed = ensure_orientation(original)

    # Convert again for Mediapipe after rotation
    fixed_rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)

    # Detect hand
    if not detect_hand(fixed_rgb):
        print(f"Skipped (no hand detected): {src_path.name}")
        return False

    # Save processed image
    cv2.imwrite(str(dst_path), fixed)
    print(f"Saved: {dst_path.name}")
    return True


def prepare_dataset():
    """Main pipeline for processing the dataset."""
    root = Path(__file__).resolve().parent
    raw_dir = root / "raw"
    processed_dir = root / "processed"

    print("Raw path:", raw_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))

    print(f"Found {len(images)} images in raw directory.")

    for img_path in images:
        dst_path = processed_dir / img_path.name
        process_image(img_path, dst_path)

    print("Dataset preparation completed.")


if __name__ == "__main__":
    prepare_dataset()
