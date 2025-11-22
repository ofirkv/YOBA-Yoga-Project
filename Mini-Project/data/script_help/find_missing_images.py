# Mini-Project/data/find_missing_images.py
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).resolve().parent

    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    csv_path = base_dir / "labels.csv"

    # Load CSV
    df = pd.read_csv(csv_path)

    # Collect filenames
    raw_images = {p.name for p in raw_dir.glob("*.jpg")}
    processed_images = {p.name for p in processed_dir.glob("*.jpg")}

    # Images that failed filtering
    missing_images = sorted(raw_images - processed_images)

    print(f"Total images in RAW: {len(raw_images)}")
    print(f"Total images in PROCESSED: {len(processed_images)}")
    print(f"Images removed by filtering: {len(missing_images)}")

    # Show failed images
    for img_name in missing_images:
        img_path = raw_dir / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img_rgb)
        plt.title(f"REMOVED: {img_name}")
        plt.axis("off")
        plt.show()

    # Remove missing images from CSV
    cleaned_df = df[df["filename"].isin(processed_images)]

    # Save cleaned CSV
    output_path = base_dir / "labels_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)

    print(f"Cleaned CSV created: {output_path}")


if __name__ == "__main__":
    main()
