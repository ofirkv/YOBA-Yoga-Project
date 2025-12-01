import csv
from pathlib import Path

def main():
    # Define paths
    project_root = Path(__file__).resolve().parents[2]  # go from data/script_help back to project root
    processed_dir = project_root / "data" / "processed"
    output_csv = project_root / "data" / "labels_processed_final.csv"

    # Load images
    image_files = sorted(
        [f for f in processed_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda x: x.name
    )

    print(f"Found {len(image_files)} images in: {processed_dir}")

    # Prepare rows
    rows = []
    cutoff_id = 369  # img_369 is last raised

    for img in image_files:
        name = img.stem  # "img_001"
        try:
            num = int(name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name format: {img.name}")
            continue

        label = "raised" if num <= cutoff_id else "lowered"
        rows.append([img.name, label])

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"CSV written to: {output_csv}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
