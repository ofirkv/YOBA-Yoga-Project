# Model/feature_extractor.py
import numpy as np
from Model.pose_utils import ANGLE_NAMES

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, angles, directions):
        """
        Generate a feature vector consisting of:
        - 8 angles
        - 8 directions (encoded numerically)

        Parameters:
            angles: dict with angle names and values
            directions: dict with angle names and direction strings

        Returns:
            1D numpy array of size 16
        """
        features = []

        # Append angles in the same order as ANGLE_NAMES
        for angle_name in ANGLE_NAMES:
            val = angles.get(angle_name)
            features.append(val if val is not None else 0.0)

        # Encode directions as numbers
        dir_encoding = {
            "up": 1.0,
            "down": 2.0,
            "left": 3.0,
            "right": 4.0,
            "up-right": 5.0,
            "up-left": 6.0,
            "down-right": 7.0,
            "down-left": 8.0,
            None: 0.0,
        }

        for angle_name in ANGLE_NAMES:
            d = directions.get(angle_name)
            features.append(dir_encoding.get(d, 0.0))

        return np.array(features, dtype=np.float32)
    
    def vector_to_dicts(self, vector):
        """
        Convert a 1D numpy array (angles + directions) back into two dicts:
        angles_dict and directions_dict.
        """
        n_angles = len(ANGLE_NAMES)
        angles_values = vector[:n_angles]
        directions_values = vector[n_angles:]
        dir_decoding = {
            1.0: "up",
            2.0: "down",
            3.0: "left",
            4.0: "right",
            5.0: "up-right",
            6.0: "up-left",
            7.0: "down-right",
            8.0: "down-left",
            0.0: None
        }
        angles_dict = {name: float(val) for name, val in zip(ANGLE_NAMES, angles_values)}
        directions_dict = {name: dir_decoding.get(float(val), None)
                        for name, val in zip(ANGLE_NAMES, directions_values)}
        return angles_dict, directions_dict