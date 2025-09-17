import json
from pose_detector import PoseDetector
from pose_utils import compute_all_angles, compute_all_angle_directions, compare_poses
from feature_extractor import FeatureExtractor

detector = PoseDetector()
extractor = FeatureExtractor()

image_filename = "1.png"
results, keypoints, confidence = detector.detect_pose(image_filename)

angles = compute_all_angles(results)
directions = compute_all_angle_directions(results)

features = extractor.extract_features(keypoints, angles)

pose_name = "downward_dog"

with open(f"Model/json_reference/{pose_name}_reference.json", "r") as f:
    ref_data = json.load(f)
angles_ref = ref_data["angles"]
directions_ref = ref_data["directions"]

print("Detection confidence:", confidence)
print("Computed angles:", angles)
print("Computed angle directions:", directions)

print("\n### Coach feedback compared to reference ###")
fixes = compare_poses(angles, angles_ref, directions, directions_ref)

if not fixes:
    print("Great job! Your pose matches the reference.")
else:
    for joint, data in fixes.items():
        print("-", data["message_en"])

print("\nFeature vector length:", len(features))
print("First 10 features:", features[:10])
