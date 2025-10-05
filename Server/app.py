from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
import mediapipe as mp
import os
import time

import json

import sys
import pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


from Model.pose_detector import PoseDetector
from Model.pose_utils import compute_all_angles, compute_all_angle_directions, ANGLE_NAMES
from Model.feature_extractor import FeatureExtractor
from Model.pose_feedback import PoseFeedback

app = Flask(__name__, template_folder="../UI/templates", static_folder="../UI/static")
SAVE_DIR = "Model/Images"
SAVE_LANDMARK_DIR = "Model/Landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Serve HTML page
@app.route("/")
def index():
    return render_template("index.html")

# Serve ready page
@app.route("/ready")
def ready():
    return render_template("ready.html")

# Receive image from JS and save skeleton
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        data = request.json["image"]
        img_data = base64.b64decode(data.split(",")[1])  # remove "data:image/png;base64,"

        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save normal photo
        filename_normal = f"captured.png"
        filepath_normal = os.path.join(SAVE_DIR, filename_normal)
        cv2.imwrite(filepath_normal, img)

        # Run MediaPipe Pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

        # # Save skeleton photo
        # filename_skel = f"captured_{int(time.time())}_landmark.png"
        # filepath_skel = os.path.join(SAVE_LANDMARK_DIR, filename_skel)
        # cv2.imwrite(filepath_skel, img)

        #######################################
        ## ANALYZE PHOTO COMPARE TO XXX POSE ##
        #######################################

        # Initialize pose detector and feature extractor
        detector = PoseDetector()
        extractor = FeatureExtractor()
        feedback = PoseFeedback(threshold_deg=10.0)

        # Choose an image filename from IMAGES folder
        image_filename = f"captured.png"

        # Detect pose
        results, keypoints, confidence = detector.detect_pose(image_filename)

        # Check detection confidence
        CONFIDENCE_THRESHOLD = 0.8
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({"status": "error", "message": "Low detection confidence"})

        # Compute angles and directions
        angles = compute_all_angles(results)
        directions = compute_all_angle_directions(results)

        # Extract features
        features = extractor.extract_features(angles, directions)

        # ********
        # !!! change automatically later !!!
        pose_name = "utkata_konasana" 
        # ********

        # Load reference pose (angles + directions) from JSON
        with open(f"Model/json_reference/{pose_name}_reference.json", "r") as f:
            ref_data = json.load(f)

        wrongs = feedback.compare_poses(features, ref_data)
        if len(wrongs) > 0:
            instr = wrongs[ANGLE_NAMES[0]]["message_en"]
        else:
            instr = "Good job!"

        return jsonify({"status": "ok",
                        "normal": filepath_normal,
                        "len": len(wrongs),
                        "msg": instr})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/first_scan", methods=["POST"])
def first_scan():
    try:
        data = request.json["image"]
        img_data = base64.b64decode(data.split(",")[1])

        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run MediaPipe Pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                return jsonify({"status": "error", "message": "No body detected"})

            # Collect needed landmarks
            lm = results.pose_landmarks.landmark
            key_points = [
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                lm[mp_pose.PoseLandmark.LEFT_HIP],
                lm[mp_pose.PoseLandmark.RIGHT_HIP],
                lm[mp_pose.PoseLandmark.LEFT_KNEE],
                lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
            ]

            # Check visibility and position (rough heuristic for full body)
            full_body = all(pt.visibility > 0.6 for pt in key_points)

            if not full_body:
                return jsonify({"status": "error", "message": "Full body not visible"})

            # Save normal + skeleton photo
            filename_normal = f"firstscan_{int(time.time())}.png"
            filepath_normal = os.path.join(SAVE_DIR, filename_normal)
            cv2.imwrite(filepath_normal, img)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            filename_skel = f"firstscan_{int(time.time())}_landmark.png"
            filepath_skel = os.path.join(SAVE_DIR, filename_skel)
            cv2.imwrite(filepath_skel, img)

            return jsonify({"status": "ok", "skeleton": filepath_skel})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
