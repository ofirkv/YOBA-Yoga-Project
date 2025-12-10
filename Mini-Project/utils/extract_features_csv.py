# # Mini-Project/model_free/extract_features_csv.py
# # !!! PUT ON PROJECT ROOT TO IMPORT PATH !!!
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# sys.stdout.reconfigure(encoding='utf-8')


# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT / "model_free"))


# from features import compute_feature_vector, feature_names_for_set
# from io_utils import list_images, read_labels_csv
# import preprocessing as preprocessing_module
# import conv_layer as conv_layer_module

# DATA_DIR = PROJECT_ROOT / "data"
# IMG_DIR = DATA_DIR / "processed"
# LABELS_CSV = DATA_DIR / "labels_processed.csv"
# OUTPUT_CSV = DATA_DIR / "features_processed.csv"

# # --- Load images and labels ---
# img_paths = list_images(IMG_DIR)
# labels_df = read_labels_csv(LABELS_CSV)

# feature_vectors = []
# image_names = []

# feature_names = feature_names_for_set()

# # --- Loop through images ---
# for img_path in img_paths:
#     print(f"Processing image: {img_path.name}")
#     img_name = img_path.name

#     # Load and preprocess image
#     img_array = preprocessing_module.preprocess_image(img_path)  # should be float32

#     # Apply conv layers to get feature maps
#     # The function returns only feature_maps, kernel names are taken from KERNELS
#     feature_maps = conv_layer_module.apply_multiple_kernels(
#         img_array, conv_layer_module.KERNELS
#     )

#     # Compute feature vector
#     vec, _ = compute_feature_vector(feature_maps, kernel_names=conv_layer_module.KERNELS)
#     feature_vectors.append(vec)
#     image_names.append(img_name)

# # --- Convert to DataFrame and save CSV ---
# df = pd.DataFrame(feature_vectors, columns=feature_names)
# df.insert(0, "filename", image_names)

# df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
# print(f"Saved feature matrix to: {OUTPUT_CSV}")
