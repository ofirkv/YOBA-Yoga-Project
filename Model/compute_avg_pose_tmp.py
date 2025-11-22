import json
from pose_detector import PoseDetector
from pose_utils import compute_all_angles, compute_all_angle_directions
from collections import Counter

# ===========================
# Pose name input
# ===========================
pose_name = "tree"

# Initialize pose detector
detector = PoseDetector()

# List of 10 image filenames
# image_filenames = [
#     "1.png", "2.png", "3.png", "4.png", "5.png",
#     "6.png", "7.png", "8.png", "9.png", "10.png"
# ]

image_filenames = [
    "1.jpeg", "2.jpeg", "3.jpeg", "4.jpeg", "5.jpeg",
    "6.jpeg", "7.jpeg", "8.jpeg", "9.jpeg", "10.jpg"
]

all_angles = []
all_directions = []

# Loop through images
for idx, image_filename in enumerate(image_filenames, start=1):
    results, keypoints, confidence = detector.detect_pose(f"{image_filename}")

    if results is None:
        print(f"Image {idx} ({image_filename}): Pose not detected")
        continue

    # Compute angles and directions
    angles = compute_all_angles(results)
    directions = compute_all_angle_directions(results)

    all_angles.append(angles)
    all_directions.append(directions)

    # Print per image
    print(f"\n--- Image {idx}: {image_filename} ---")
    print("Detection confidence:", confidence)
    print("Computed angles:", angles)
    print("Computed directions:", directions)

# ===========================
# Compute average pose
# ===========================

avg_angles = {}
avg_directions = {}

if all_angles:
    # Average angles
    for key in all_angles[0].keys():
        values = [float(d[key]) for d in all_angles]
        avg = sum(values) / len(values)
        avg_angles[key] = round(avg, 3)

if all_directions:
    # Most common direction per joint
    for key in all_directions[0].keys():
        values = [d[key] for d in all_directions]
        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]
        avg_directions[key] = most_common

print("\n### Average Pose over all images ###")
print("Average angles:", avg_angles)
print("Average directions:", avg_directions)

# ===========================
# Save to JSON
# ===========================

pose_data = {
    "pose_name": pose_name,
    "angles": avg_angles,
    "directions": avg_directions
}

output_filename = f"{pose_name.lower()}_reference.json"
with open(output_filename, "w") as f:
    json.dump(pose_data, f, indent=4)

print(f"\nAverage pose saved to {output_filename}")

#some change to check git!!
