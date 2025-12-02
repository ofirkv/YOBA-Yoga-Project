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
import csv
import time
import argparse

# ensure project root is on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# local imports from project
from model_free.io_utils import list_images, ensure_dir, read_labels_csv
from model_free.preprocessing import preprocess_image
from model_free.conv_layer import process_image_with_kernels, DEFAULT_IMPORTANT_KERNELS
from model_free.features import compute_feature_vector, feature_names_for_set
from model_free.classifier import predict_batch, evaluate, save_report_csv, save_report_json

import numpy as np

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels_processed.csv"
DEFAULT_OUT = PROJECT_ROOT / "data"

PER_IMAGE_CSV_NAME = "classification_results.csv"
REPORT_CSV_NAME = "classification_report.csv"
REPORT_JSON_NAME = "classification_report.json"


def run_pipeline(input_folder: Path = DEFAULT_INPUT,
                 labels_csv: Path = DEFAULT_LABELS,
                 out_folder: Path = DEFAULT_OUT,
                 kernels = None,
                 verbose: bool = False):
    """
    Run full pipeline on folder of preprocessed images.

    Args:
        input_folder: Path to folder with preprocessed images (images should be readable).
        labels_csv: optional path to CSV with ground-truth: filename,label
        out_folder: where to save per-image CSV and reports
        kernels: list of kernel names to pass to conv_layer (defaults to DEFAULT_IMPORTANT_KERNELS)
        verbose: print extra logs for first few images
    """
    print("Demo pipeline start")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {input_folder}")
    print(f"Labels CSV (optional): {labels_csv if Path(labels_csv).exists() else 'NOT FOUND'}")
    print(f"Output folder: {out_folder}")

    input_folder = Path(input_folder)
    out_folder = Path(out_folder)
    ensure_dir(out_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # gather images
    image_paths = list_images(input_folder)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {input_folder}")

    print(f"Found {len(image_paths)} images")

    # choose kernels
    if kernels is None:
        kernels = DEFAULT_IMPORTANT_KERNELS

    feature_vectors = []
    filenames = []
    failed_images = []

    # process each image: preprocess -> conv pipeline -> extract feature vector
    for i, p in enumerate(image_paths):
        try:
            processed = preprocess_image(p)  # accepts path
            stacked_maps = process_image_with_kernels(processed, kernels=kernels)
            vec, feature_names = compute_feature_vector(stacked_maps, kernel_names=kernels)
            feature_vectors.append(vec)
            filenames.append(p.name)
            if verbose and i < 3:
                print(f"[INFO] processed {p.name} -> feature vector shape {vec.shape}")
        except Exception as e:
            print(f"[WARN] failed processing {p.name}: {e}")
            failed_images.append(p.name)
            continue

    if len(feature_vectors) == 0:
        raise RuntimeError("No feature vectors produced. Aborting.")

    # build feature matrix
    X = np.stack(feature_vectors, axis=0)  # shape (N, D)

    # classify batch (uses feature_names returned by features)
    # ensure the classifier gets the same feature order
    feature_names_list = feature_names_for_set()
    labels_pred, scores = predict_batch(X, feature_names_list)

    # per-image CSV with predictions and scores and optional ground-truth if available
    labels_map = read_labels_csv(labels_csv) if Path(labels_csv).exists() else {}
    per_image_csv_path = out_folder / PER_IMAGE_CSV_NAME
    with per_image_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "score", "ground_truth"])
        for fname, pred, score in zip(filenames, labels_pred, scores):
            gt = labels_map.get(fname, "")
            writer.writerow([fname, pred, f"{score:.6f}", gt])
    print(f"Wrote per-image results to {per_image_csv_path}")

    # compute evaluation metrics only for images that have ground-truth labels
    eval_indices = [i for i, fn in enumerate(filenames) if fn in labels_map and labels_map[fn] in ("raised", "lowered")]
    report_csv = out_folder / REPORT_CSV_NAME
    report_json = out_folder / REPORT_JSON_NAME

    if len(eval_indices) == 0:
        print("No ground-truth labels found for these filenames. Skipping evaluation metrics.")
        metrics = {
            "n": len(filenames),
            "note": "no_ground_truth_available",
            "num_failed_processing": len(failed_images)
        }
        # save placeholder reports
        save_report_json(report_json, metrics, extra_info={"failed_images": failed_images}, overwrite=True)
        # write a small CSV note (to keep API consistent)
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["n", metrics["n"]])
            writer.writerow(["note", metrics["note"]])
            writer.writerow(["num_failed_processing", metrics["num_failed_processing"]])
        print(f"Wrote placeholder reports to {report_csv} and {report_json}")
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
        "kernels_used": list(kernels),
    }

    # save metrics (CSV + JSON)
    save_report_csv(report_csv, metrics, extra_info=extra_info, overwrite=True)
    save_report_json(report_json, metrics, extra_info=extra_info, overwrite=True)

    # print summary
    print("Evaluation metrics computed and saved:")
    print(f"- CSV: {report_csv}")
    print(f"- JSON: {report_json}")
    print("Metrics summary:")
    print(f"  n (eval): {metrics.get('n')}")
    print(f"  accuracy: {metrics.get('accuracy'):.4f}")
    print(f"  precision: {metrics.get('precision'):.4f}")
    print(f"  recall: {metrics.get('recall'):.4f}")
    print(f"  f1: {metrics.get('f1'):.4f}")
    print("Done.")


def _parse_args():
    p = argparse.ArgumentParser(description="Run demo pipeline (preprocess -> conv -> features -> classify).")
    p.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT), help="Input folder with preprocessed images")
    p.add_argument("--labels", "-l", type=str, default=str(DEFAULT_LABELS), help="CSV with ground-truth labels (filename,label)")
    p.add_argument("--out", "-o", type=str, default=str(DEFAULT_OUT), help="Output folder for results/reports")
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose prints")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(input_folder=Path(args.input), labels_csv=Path(args.labels), out_folder=Path(args.out), verbose=args.verbose)
