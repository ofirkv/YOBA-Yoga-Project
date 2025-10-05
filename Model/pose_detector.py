import cv2
import mediapipe as mp
import numpy as np
import os

class PoseDetector:
    def __init__(self, static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize Mediapipe Pose model with given parameters.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = static_image_mode,
            model_complexity = model_complexity,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Define directories
        self.images_dir = os.path.join(os.path.dirname(__file__), "Images")
        self.landmarks_dir = os.path.join(os.path.dirname(__file__), "Landmarks")

        # Create directories if not exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)

    def detect_pose(self, image_name):
        """
        Detect pose landmarks from an image in IMAGES folder.
        Returns:
            keypoints: numpy array of shape (33, 4) [x, y, z, visibility]
            confidence: average visibility score
        """
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_name} not found in IMAGES directory.")

        # Convert to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, 0.0

        keypoints = self.extract_keypoints(results)
        confidence = np.mean(keypoints[:, 3])  # average visibility

        # Save image with landmarks drawn
        self.draw_landmarks(image, results, image_name)

        return results, keypoints, confidence

    def extract_keypoints(self, results):
        """
        Extract keypoints from Mediapipe results.
        Returns numpy array with shape (33, 4) -> [x, y, z, visibility]
        """
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
        return keypoints

    def draw_landmarks(self, image, results, image_name):
        """
        Draw landmarks on image and save it into LANDMARKS folder.
        """
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        output_path = os.path.join(self.landmarks_dir, f"landmarks_{image_name}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Landmarked image saved at: {output_path}")