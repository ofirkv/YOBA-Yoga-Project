# Mini-Project/model_free/demo.py
"""
Demo script: run full pipeline on folder data/processed
- preprocess -> conv -> features -> classify (rule-based)
- copy images into data/classified/raised and data/classified/lowered
- save per-image results csv: data/classification_results.csv
- save evaluation metrics (if ground-truth available) using save_report_csv/json into data/
"""
from pathlib import Path
import sys
import shutil
import csv
import time
from typing import List

# ensure project root is on path (relative imports in package)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# local imports from project
from model_free.io_utils import list_images, ensure_dir, read_labels_csv
from model_free.preprocessing import preprocess_image
from model_free.conv_layer import process_image_with_kernels, DEFAULT_IMPORTANT_KERNELS
from model_free.features import compute_feature_vector, feature_names_for_set
from model_free.classifier import predict_batch, evaluate, save_report_csv, save_report_json

import numpy as np

# Config (change if needed)
INPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
CLASSIFIED_FOLDER = PROJECT_ROOT / "data" / "classified"
REPORT_CSV_PATH = PROJECT_ROOT / "data" / "classification_report.csv"
REPORT_JSON_PATH = PROJECT_ROOT / "data" / "classification_report.json"
PER_IMAGE_CSV = PROJECT_ROOT / "data" / "classification_results.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "labels_processed.csv"  # ground truth mapping (optional)

def run_pipeline(input_folder: Path = INPUT_FOLDER):
    print("Demo pipeline start")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {input_folder}")

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # gather images
    image_paths = list_images(input_folder)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {input_folder}")

    print(f"Found {len(image_paths)} images")

    feature_vectors: List[np.ndarray] = []
    filenames: List[str] = []
    failed_images: List[str] = []

    # process each image: preprocess -> conv pipeline -> extract feature vector
    for p in image_paths:
        try:
            # preprocess_image accepts path or ndarray
            processed = preprocess_image(p)
            stacked_maps = process_image_with_kernels(processed)
            vec, feature_names = compute_feature_vector(stacked_maps, kernel_names=DEFAULT_IMPORTANT_KERNELS)
            feature_vectors.append(vec)
            filenames.append(p.name)
        except Exception as e:
            print(f"[WARN] failed processing {p.name}: {e}")
            failed_images.append(p.name)
            continue

    if len(feature_vectors) == 0:
        raise RuntimeError("No feature vectors produced. Aborting.")

    # build feature matrix
    X = np.stack(feature_vectors, axis=0)  # shape (N, D)

    # classify batch
    labels_pred, scores = predict_batch(X, feature_names_for_set())

    # prepare classified folders
    raised_dir = CLASSIFIED_FOLDER / "raised"
    lowered_dir = CLASSIFIED_FOLDER / "lowered"
    ensure_dir(raised_dir)
    ensure_dir(lowered_dir)

    # copy images into respective folders
    num_copied = 0
    for fname, label in zip(filenames, labels_pred):
        src = input_folder / fname
        if not src.exists():
            print(f"[WARN] source image not found for copying: {src}")
            continue
        dst = (raised_dir if label == "raised" else lowered_dir) / fname
        shutil.copy2(src, dst)
        num_copied += 1
    print(f"Copied {num_copied} images into {CLASSIFIED_FOLDER}")

    # per-image CSV with predictions and scores and optional ground-truth if available
    labels_map = read_labels_csv(LABELS_CSV) if (Path(LABELS_CSV).exists()) else {}
    with PER_IMAGE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "score", "ground_truth"])
        for fname, pred, score in zip(filenames, labels_pred, scores):
            gt = labels_map.get(fname, "")
            writer.writerow([fname, pred, f"{score:.6f}", gt])
    print(f"Wrote per-image results to {PER_IMAGE_CSV}")

    # compute evaluation metrics only for images that have ground-truth labels in labels_map
    eval_indices = [i for i, fn in enumerate(filenames) if fn in labels_map and labels_map[fn] in ("raised", "lowered")]
    if len(eval_indices) == 0:
        print("No ground-truth labels found in data/labels.csv for these filenames. Skipping evaluation metrics.")
        metrics = {
            "n": len(filenames),
            "note": "no_ground_truth_available",
            "num_failed_processing": len(failed_images)
        }
        # still save a JSON report notifying lack of GT
        save_report_json(REPORT_JSON_PATH, metrics, extra_info={"failed_images": failed_images}, overwrite=True)
        # write a small CSV note (to keep API consistent)
        with REPORT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["n", metrics["n"]])
            writer.writerow(["note", metrics["note"]])
            writer.writerow(["num_failed_processing", metrics["num_failed_processing"]])
        print(f"Wrote placeholder reports to {REPORT_CSV_PATH} and {REPORT_JSON_PATH}")
        return

    preds_eval = [labels_pred[i] for i in eval_indices]
    gts_eval = [labels_map[filenames[i]] for i in eval_indices]

    metrics = evaluate(preds_eval, gts_eval, positive_label="raised")

    # extra_info: counts, totals, time
    extra_info = {
        "total_processed": len(filenames),
        "num_failed_processing": len(failed_images),
        "num_with_ground_truth": len(eval_indices),
        "num_predicted_raised": int(sum(1 for l in labels_pred if l == "raised")),
        "num_predicted_lowered": int(sum(1 for l in labels_pred if l == "lowered")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # save metrics (CSV + JSON)
    save_report_csv(REPORT_CSV_PATH, metrics, extra_info=extra_info, overwrite=True)
    save_report_json(REPORT_JSON_PATH, metrics, extra_info=extra_info, overwrite=True)

    print("Evaluation metrics computed and saved:")
    print(f"- CSV: {REPORT_CSV_PATH}")
    print(f"- JSON: {REPORT_JSON_PATH}")
    print("Metrics summary:")
    print(f"  n (eval): {metrics.get('n')}")
    print(f"  accuracy: {metrics.get('accuracy'):.4f}")
    print(f"  precision: {metrics.get('precision'):.4f}")
    print(f"  recall: {metrics.get('recall'):.4f}")
    print(f"  f1: {metrics.get('f1'):.4f}")
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
