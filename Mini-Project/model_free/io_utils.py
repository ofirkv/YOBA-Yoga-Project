# Mini-Project/model_free/io_utils.py
from pathlib import Path
import csv
import shutil

import cv2
import numpy as np

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CSV_HEADER = ["filename", "label"]

# --- Filesystem helpers ---
def ensure_dir(path):
    """
    Ensure a directory exists. Returns the Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder, extensions = DEFAULT_IMAGE_EXTENSIONS, recursive = False):
    """
    List image file paths in a folder filtered by extensions.
    Returns a sorted list of Path objects.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    ext_set = set(e.lower() for e in extensions)
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in ext_set]
    files_sorted = sorted(files)
    return files_sorted


def image_exists(path):
    """
    Return True if the path exists and is a file.
    """
    p = Path(path)
    return p.exists() and p.is_file()


# --- Image IO helpers ---
def load_image(path, as_gray = False):
    """
    Load image with OpenCV.
    - path: file path
    - as_gray: if True load grayscale (single channel), otherwise BGR (3-channel)

    Returns numpy array or None if file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        print(f"[io_utils] load_image: file not found: {p}")
        return None

    if as_gray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        print(f"[io_utils] load_image: OpenCV failed to read image (may be corrupted): {p}")
    return img


def validate_image_file(path):
    """
    Quick validation: file exists and OpenCV can decode it.
    """
    img = load_image(path, as_gray=True)
    if img is None:
        return False
    # check minimal size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return False
    return True


def get_image_shape(path):
    """
    Return shape (height, width, channels) if readable, otherwise None.
    """
    img = load_image(path, as_gray=False)
    if img is None:
        return None
    # OpenCV BGR: if grayscale it returns 2 dims
    if img.ndim == 2:
        return img.shape[0], img.shape[1], 1
    return img.shape[0], img.shape[1], img.shape[2]


def save_image(path, image, create_dir = True, overwrite = False):
    """
    Save image (numpy array) to path using OpenCV.
    - If create_dir is True, parent directory is created automatically.
    - If overwrite is False and file exists, function returns False.
    Returns True on success.
    """
    p = Path(path)
    if create_dir:
        ensure_dir(p.parent)

    if p.exists() and not overwrite:
        print(f"[io_utils] save_image: file exists and overwrite=False: {p}")
        return False

    success = cv2.imwrite(str(p), image)
    if not success:
        print(f"[io_utils] save_image: failed to write image: {p}")
    return bool(success)


def copy_images_to_folder(src_paths, dst_folder, overwrite = False):
    """
    Copy multiple image files to dst_folder. Returns list of destination Paths for successfully copied files.
    """
    dst_folder = ensure_dir(dst_folder)
    copied = []
    for src in src_paths:
        src_p = Path(src)
        if not src_p.exists() or not src_p.is_file():
            print(f"[io_utils] copy_images_to_folder: skip not-found file {src}")
            continue
        dst = dst_folder / src_p.name
        if dst.exists() and not overwrite:
            print(f"[io_utils] copy_images_to_folder: skip existing {dst}")
            continue
        shutil.copy2(str(src_p), str(dst))
        copied.append(dst)
    return copied


# --- CSV / labels helpers ---

def read_labels_csv(csv_path):
    """
    Read labels CSV into a dictionary: filename -> label (possibly empty string).
    If the file does not exist, returns empty dict.
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        return {}

    mapping = {}
    with csv_p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # allow files with or without header; expect 'filename' and 'label' columns
        for row in reader:
            fn = row.get("filename") or row.get("file") or row.get("name")
            label = row.get("label") or ""
            if fn:
                mapping[fn] = label
    return mapping


def write_labels_csv(csv_path, mapping, overwrite = True):
    """
    Write mapping (filename -> label) to csv_path.
    If overwrite is False and file exists, raises FileExistsError.
    """
    csv_p = Path(csv_path)
    if csv_p.exists() and not overwrite:
        raise FileExistsError(f"File exists: {csv_p}")

    ensure_dir(csv_p.parent)
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for fn, label in sorted(mapping.items()):
            writer.writerow([fn, label])


def append_label_to_csv(csv_path, filename, label):
    """
    Append a row to CSV. Creates file with header if not exists.
    """
    csv_p = Path(csv_path)
    exists = csv_p.exists()
    ensure_dir(csv_p.parent)
    with csv_p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(CSV_HEADER)
        writer.writerow([filename, label])


def update_label_in_csv(csv_path, filename, new_label):
    """
    Update a filename label in the CSV. Returns True if updated, False if filename not found.
    Will rewrite the CSV file in-place.
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        print(f"[io_utils] update_label_in_csv: csv file not found: {csv_p}")
        return False

    mapping = read_labels_csv(csv_p)
    if filename not in mapping:
        print(f"[io_utils] update_label_in_csv: filename not in CSV: {filename}")
        return False

    mapping[filename] = new_label
    write_labels_csv(csv_p, mapping, overwrite=True)
    return True


# --- Utilities for quick manual checks ---

def collect_dataset_summary(folder, csv_path = None):
    """
    Return a small summary: total images, number labeled, number unlabeled, invalid images.
    Optionally takes a csv_path to compare mapping.
    """
    image_files = list_images(folder)
    total = len(image_files)
    mapping = read_labels_csv(csv_path) if csv_path else {}
    labeled = sum(1 for p in image_files if p.name in mapping and mapping.get(p.name, "") != "")
    unlabeled = total - labeled
    invalid = sum(1 for p in image_files if not validate_image_file(p))
    return {"total": total, "labeled": labeled, "unlabeled": unlabeled, "invalid": invalid}