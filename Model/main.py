# Model/main.py
import json
from pose_detector import PoseDetector
from pose_utils import compute_all_angles, compute_all_angle_directions
from feature_extractor import FeatureExtractor
from pose_feedback import PoseFeedback

# Initialize pose detector and feature extractor
detector = PoseDetector()
extractor = FeatureExtractor()
feedback = PoseFeedback(threshold_deg=10.0)

# Choose an image filename from IMAGES folder
image_filename = "1.png"

# Detect pose
results, keypoints, confidence = detector.detect_pose(image_filename)

# Check detection confidence
CONFIDENCE_THRESHOLD = 0.8 
if confidence < CONFIDENCE_THRESHOLD:
    print("Low detection confidence. Please improve your lighting, camera position, or make sure your full body is visible.")

# Compute angles and directions
angles = compute_all_angles(results)
directions = compute_all_angle_directions(results)

# Extract features
features = extractor.extract_features(angles, directions)

pose_name = "downward_dog"  # 👈 Replace manually before each run

# Load reference pose (angles + directions) from JSON
with open(f"Model/json_reference/{pose_name}_reference.json", "r") as f:
    ref_data = json.load(f)

# Print current detection
print("Detection confidence:", confidence)
print("Computed angles:", angles)
print("Computed angle directions:", directions)

# Compare with reference
print("\n### Coach feedback compared to reference ###")
fixes = feedback.compare_poses(features, ref_data)

if not fixes:
    print("Great job! Your pose matches the reference.")
else:
    for joint, data in fixes.items():
        print("-", data["message_en"])

print("\nFeature vector length:", len(features))
print("features:", features)
