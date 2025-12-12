# Mini-Project/model_free/demo_from_features.py
"""
Demo script using precomputed feature vectors from CSV
- read features CSV -> classify -> save per-image CSV -> save evaluation metrics
"""
from pathlib import Path
import sys
import csv
import time
import argparse
import numpy as np

# ensure project root is on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model_free.classifier import predict_batch, evaluate, save_report_csv, save_report_json
from model_free.features import feature_names_for_set
from model_free.io_utils import read_labels_csv, ensure_dir

DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "features_processed.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels_processed.csv"
DEFAULT_OUT = PROJECT_ROOT / "data"

PER_IMAGE_CSV_NAME = "classification_results.csv"
REPORT_CSV_NAME = "classification_report.csv"
REPORT_JSON_NAME = "classification_report.json"

def run_from_features(features_csv: Path = DEFAULT_FEATURES_CSV,
                      labels_csv: Path = DEFAULT_LABELS,
                      out_folder: Path = DEFAULT_OUT,
                      verbose: bool = False):
    print("Demo pipeline (from features CSV) start")
    print(f"Features CSV: {features_csv}")
    print(f"Labels CSV (optional): {labels_csv if labels_csv.exists() else 'NOT FOUND'}")
    print(f"Output folder: {out_folder}")
    
    ensure_dir(out_folder)

    # read features CSV: expect first column filename, rest = features
    filenames = []
    feature_vectors = []
    with features_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            filenames.append(row[0])
            vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
            feature_vectors.append(vec)

    if len(feature_vectors) == 0:
        raise RuntimeError("No feature vectors found in CSV.")

    X = np.stack(feature_vectors, axis=0)
    
    # classify
    feature_names_list = feature_names_for_set()
    labels_pred, scores = predict_batch(X, feature_names_list)

    # per-image CSV
    labels_map = read_labels_csv(labels_csv) if labels_csv.exists() else {}
    per_image_csv_path = out_folder / PER_IMAGE_CSV_NAME
    with per_image_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "score", "ground_truth"])
        for fname, pred, score in zip(filenames, labels_pred, scores):
            gt = labels_map.get(fname, "")
            writer.writerow([fname, pred, f"{score:.6f}", gt])
    print(f"Wrote per-image results to {per_image_csv_path}")

    # evaluation metrics
    eval_indices = [i for i, fn in enumerate(filenames) if fn in labels_map and labels_map[fn] in ("raised", "lowered")]
    report_csv = out_folder / REPORT_CSV_NAME
    report_json = out_folder / REPORT_JSON_NAME

    if len(eval_indices) == 0:
        print("No ground-truth labels found. Skipping evaluation metrics.")
        metrics = {
            "n": len(filenames),
            "note": "no_ground_truth_available"
        }
        save_report_json(report_json, metrics, extra_info={}, overwrite=True)
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["n", metrics["n"]])
            writer.writerow(["note", metrics["note"]])
        print(f"Wrote placeholder reports to {report_csv} and {report_json}")
        return

    preds_eval = [labels_pred[i] for i in eval_indices]
    gts_eval = [labels_map[filenames[i]] for i in eval_indices]
    metrics = evaluate(preds_eval, gts_eval, positive_label="raised")

    extra_info = {
        "total_processed": len(filenames),
        "num_with_ground_truth": len(eval_indices),
        "num_predicted_raised": int(sum(1 for l in labels_pred if l == "raised")),
        "num_predicted_lowered": int(sum(1 for l in labels_pred if l == "lowered")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    save_report_csv(report_csv, metrics, extra_info=extra_info, overwrite=True)
    save_report_json(report_json, metrics, extra_info=extra_info, overwrite=True)

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
    p = argparse.ArgumentParser(description="Run demo pipeline from precomputed features CSV.")
    p.add_argument("--features", "-f", type=str, default=str(DEFAULT_FEATURES_CSV), help="CSV with precomputed features")
    p.add_argument("--labels", "-l", type=str, default=str(DEFAULT_LABELS), help="CSV with ground-truth labels")
    p.add_argument("--out", "-o", type=str, default=str(DEFAULT_OUT), help="Output folder for results/reports")
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose prints")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_from_features(features_csv=Path(args.features), labels_csv=Path(args.labels), out_folder=Path(args.out), verbose=args.verbose)