# Mini-Project/data/script_help/rename_images.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")

START_INDEX = 636

def rename_new_images():
    if not os.path.exists(RAW_FOLDER):
        print(f"Folder not found: {RAW_FOLDER}")
        return

    files = sorted(os.listdir(RAW_FOLDER))
    index = START_INDEX

    for file_name in files:
        file_path = os.path.join(RAW_FOLDER, file_name)

        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        if name.startswith("img_"):
            continue

        new_name = f"img_{index}{ext}"
        new_path = os.path.join(RAW_FOLDER, new_name)

        while os.path.exists(new_path):
            index += 1
            new_name = f"img_{index}{ext}"
            new_path = os.path.join(RAW_FOLDER, new_name)

        os.rename(file_path, new_path)
        print(f"Renamed {file_name} -> {new_name}")

        index += 1


if __name__ == "__main__":
    rename_new_images()
