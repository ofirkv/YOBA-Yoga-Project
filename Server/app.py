# Server/app.py
# === Imports ===
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import base64
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import json
import sys
import pathlib
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import Error

# === Project Path Setup ===
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # already calculated as project_root
sys.path.append(str(PROJECT_ROOT))

# === Model Imports ===
from Model.pose_detector import PoseDetector
from Model.pose_utils import compute_all_angles, compute_all_angle_directions, ANGLE_NAMES
from Model.feature_extractor import FeatureExtractor
from Model.pose_feedback import PoseFeedback

# === Flask App Setup ===
app = Flask(
    __name__,
    template_folder="../UI/templates",
    static_folder="../UI/static"
)

# === Directories Setup ===
CONFIG_PATH = pathlib.Path(__file__).resolve().parent / 'config.txt'
MODEL_DIR = PROJECT_ROOT / "Model"
SAVE_DIR = str(MODEL_DIR / "Images")
SAVE_LANDMARK_DIR = str(MODEL_DIR / "Landmarks")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_LANDMARK_DIR, exist_ok=True)

# === Mediapipe Setup ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# === Database Configuration ===
DB_USER = "ofir"
DB_PASSWORD = "OFIRKVETNY1"

app.secret_key = os.environ.get("SECRET_KEY", "change_this_secret_to_env_value")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "user": os.environ.get("DB_USER", DB_USER),
    "password": os.environ.get("DB_PASSWORD", DB_PASSWORD),
    "database": os.environ.get("DB_NAME", "yoba_db"),
    "auth_plugin": "mysql_native_password"
}


# === Database Connection Helper ===
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print("DB connection error:", e)
        return None


@app.route("/test_db")
def test_db():
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            db_info = connection.get_server_info()
            return f"✅ Database connection successful! MySQL Server version: {db_info}"
        else:
            return "❌ Connection attempt failed (no connection established)."
    except Error as e:
        return f"❌ Database connection failed: {str(e)}"
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


# === Registration Route ===
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    # POST logic
    data = request.form
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    password_confirm = data.get("password_confirm", "")

    if not name or not email or not password:
        flash("Please fill required fields", "error")
        return redirect(url_for("register"))

    if password != password_confirm:
        flash("Passwords do not match", "error")
        return redirect(url_for("register"))

    password_hash = generate_password_hash(password)
    conn = get_db_connection()
    if conn is None:
        flash("Database connection error", "error")
        return redirect(url_for("register"))

    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid

        # Create empty profile row
        cursor.execute(
            "INSERT INTO user_profile (user_id) VALUES (%s)",
            (user_id,)
        )
        conn.commit()
        cursor.close()
        conn.close()

        # Auto login after register
        session["user_id"] = user_id
        session["user_name"] = name
        return redirect(url_for("personal_data"))

    except mysql.connector.IntegrityError:
        flash("Email already registered", "error")
        return redirect(url_for("register"))
    except Exception as e:
        print("Register error:", e)
        flash("Registration failed", "error")
        return redirect(url_for("register"))


# === Login Route ===
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    conn = get_db_connection()
    if conn is None:
        flash("Database connection error", "error")
        return redirect(url_for("login"))

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("welcome"))
        else:
            flash("Invalid credentials", "error")
            return redirect(url_for("login"))

    except Exception as e:
        print("Login error:", e)
        flash("Login failed", "error")
        return redirect(url_for("login"))


# === Logout Route ===
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# === Personal Data Route ===
@app.route("/personal_data", methods=["GET", "POST"])
def personal_data():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    conn = get_db_connection()
    if conn is None:
        flash("Database connection error", "error")
        return redirect(url_for("welcome"))

    if request.method == "GET":
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM user_profile WHERE user_id = %s", (user_id,))
        profile = cursor.fetchone()
        cursor.close()
        conn.close()
        return render_template("personal_data.html", profile=profile)

    # POST - save profile
    form = request.form
    if form.get("skip"):
        return redirect(url_for("welcome"))

    age = form.get("age") or None
    gender = form.get("gender") or None
    weight = form.get("weight") or None
    height = form.get("height") or None
    experience_level = form.get("experience_level") or None
    preferred_length = form.get("preferred_length") or None
    injuries = form.getlist("injuries")
    injuries_text = ",".join(injuries) if injuries else form.get("injuries_text") or None

    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_profile SET
                age=%s, gender=%s, weight=%s, height=%s,
                experience_level=%s, preferred_length=%s, injuries=%s
            WHERE user_id=%s
        """, (age, gender, weight, height, experience_level, preferred_length, injuries_text, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for("welcome"))

    except Exception as e:
        print("Save profile error:", e)
        flash("Failed to save profile", "error")
        return redirect(url_for("personal_data"))


# === Page Rendering Routes ===
@app.route("/welcome")
def welcome():
    user_name = session.get("user_name")
    if not user_name:
        return redirect(url_for("login"))
    return render_template("welcome.html", user_name=user_name)


@app.route("/choose_program")
def choose_program():
    return render_template("choose_program.html")


@app.route("/body_scan")
def body_scan():
    return render_template("body_scan.html")


@app.route("/train")
def train():
    return render_template("train_page.html")

# === Image Upload & Pose Analysis ===
@app.route("/upload_burst", methods=["POST"])
def upload_burst():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            data_config = json.load(f)

    LEFT_RIGHT_MAP = {
        "left": "right",
        "Left": "Right",
        "right": "left",
        "Right": "Left"
    }

    def flip_left_right(text: str) -> str:
        """Swap left/right words in instruction text."""
        for k, v in LEFT_RIGHT_MAP.items():
            text = text.replace(k, v)
        return text

    try:
        pose_name = request.json["pose"]
        images_base64 = request.json["images"]

        # --- Initialize models ---
        detector = PoseDetector()
        extractor = FeatureExtractor()
        feedback = PoseFeedback(threshold_deg=20.0, instructions_config=data_config["instructions"],
                                emphasises=data_config["emphasises"], pose_name=pose_name)

        if not images_base64:
            return jsonify({"status": "error", "message": "No images received"})

        # --- Save images ---
        image_paths = []
        for i, data_url in enumerate(images_base64, start=1):
            img_data = base64.b64decode(data_url.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            filepath = os.path.join(SAVE_DIR, f"captured{i}.png")
            cv2.imwrite(filepath, img)
            image_paths.append(filepath)

        # --- Load reference JSON ---
        json_path = PROJECT_ROOT / "Model" / "json_reference" / f"{pose_name}_reference.json"
        with open(json_path, "r") as f:
            ref_data = json.load(f)


        # --- Initialize best variables ---
        best_len = 100
        current_best_confidence = 0
        best_filepath = ""
        best_wrongs = {}
        best_message = ""
        best_angles = None
        best_directions = None

        CONFIDENCE_THRESHOLD = 0.8

        # --- Process each image ---
        for i, filepath in enumerate(image_paths):
            img = cv2.imread(filepath)

            # Prepare both normal and mirrored versions
            versions = {
                "normal": img,
                "mirrored": cv2.flip(img, 1)  # horizontal flip
            }

            for version_name, version_img in versions.items():
                # --- Detect pose ---
                if version_name == "mirrored":
                    # Save mirrored image to disk
                    mirrored_path = os.path.join(SAVE_DIR, f"captured{(i+1)}_mirrored.png")
                    cv2.imwrite(mirrored_path, version_img)
                    detect_path = mirrored_path
                else:
                    detect_path = filepath  # normal image path

                results_pose, keypoints, confidence = detector.detect_pose(detect_path)

                if confidence < CONFIDENCE_THRESHOLD:
                    continue  # skip low-confidence images

                # --- Compute angles and directions ---
                angles = compute_all_angles(results_pose)
                directions = compute_all_angle_directions(results_pose)

                # --- Extract features ---
                features = extractor.extract_features(angles, directions)

                # --- Compare pose with reference ---
                wrongs = feedback.compare_poses(features, ref_data)

                # --- Prepare instruction message ---
                if len(wrongs) > 0:
                    first_key = list(wrongs.keys())[0]
                    instr = wrongs[first_key]["message_en"]
                else:
                    instr = "Good job!"

                # --- Flip instructions if mirrored ---
                if version_name == "mirrored":
                    instr = flip_left_right(instr)

                # --- Update best if necessary ---
                if len(wrongs) < best_len or (len(wrongs) == best_len and confidence > current_best_confidence):
                    best_len = len(wrongs)
                    best_wrongs = wrongs
                    best_angles = angles
                    best_directions = directions
                    best_message = instr
                    current_best_confidence = confidence

        return jsonify({
            "status": "ok",
            "len": best_len,
            "msg": best_message
        })

    except Exception as e:
        print("Upload burst error:", e)
        return jsonify({"status": "error", "message": str(e)})

# === Initial Body Scan Route ===
@app.route("/first_scan", methods=["POST"])
def first_scan():
    try:
        data = request.json["image"]
        img_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print("Received image shape:", img.shape)

        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return jsonify({"status": "error", "message": "No body detected"})

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

            full_body = all(pt.visibility > 0.6 for pt in key_points)
            if not full_body:
                return jsonify({"status": "error", "message": "Full body not visible"})

            # Save normal + skeleton
            filename_normal = "firstscan.png"
            filepath_normal = os.path.join(SAVE_DIR, filename_normal)
            cv2.imwrite(filepath_normal, img)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            filename_skel = "firstscan_landmark.png"
            filepath_skel = os.path.join(SAVE_LANDMARK_DIR, filename_skel)
            cv2.imwrite(filepath_skel, img)

            return jsonify({"status": "ok", "skeleton": filepath_skel})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# === Index & Score Routes ===
@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/score")
def score():
    perfect = request.args.get("perfect", 0)
    total = request.args.get("total", 0)
    return render_template("score.html", perfect=perfect, total=total)

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.get_json()
    # Save as JSON
    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    return jsonify({"status": "success"})

# NEW: serve the config file as JSON for JS
@app.route('/get_config', methods=['GET'])
def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Config not found"}), 404

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)