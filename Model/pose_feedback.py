# pose_feedback.py
from Model.pose_utils import ANGLE_NAMES
from Model.feature_extractor import FeatureExtractor

SIDE_EN = {"left": "left", "right": "right"}
JOINT_LABEL_EN = {"elbow": "elbow", "shoulder": "shoulder", "knee": "knee", "hip": "hip"}
DISTAL_SEGMENT_EN = {"elbow": "forearm", "shoulder": "upper arm", "knee": "shin", "hip": "knee"}

class PoseFeedback:
    def __init__(self, threshold_deg=10.0):
        self.threshold_deg = threshold_deg

    def angle_action_en(self, joint_kind, is_open):
        if joint_kind in ("elbow", "knee"):
            return "Bend" if is_open else "Straighten"
        if joint_kind in ("shoulder", "hip"):
            return "Close" if is_open else "Open"
        return "Adjust"

    def direction_to_en(self, d):
        m = {
            "up": "Lift",
            "down": "Lower",
            "left": "Shift left",
            "right": "Shift right",
            "up-right": "Lift slightly and shift right",
            "up-left": "Lift slightly and shift left",
            "down-right": "Lower slightly and shift right",
            "down-left": "Lower slightly and shift left",
        }
        return m.get(d, "Hold direction")

    def adverb_for_diff(self, d):
        if d < (self.threshold_deg * 0.8): return "slightly"
        if d < (self.threshold_deg * 1.5): return "a bit"
        return f"about {int(round(d))}°"

    def parse_joint_meta(self, name):
        side = "left" if name.startswith("left") else "right"
        if "elbow" in name: kind = "elbow"
        elif "shoulder" in name: kind = "shoulder"
        elif "knee" in name: kind = "knee"
        else: kind = "hip"
        return side, kind

    def compare_poses(self, user_vector, ref_data):
        """
        Compare user pose vector with reference data loaded from JSON.
        user_vector: np.array with angles+directions from FeatureExtractor.
        ref_data: dict loaded from JSON with keys 'angles' and 'directions'.
        """
        # Decode user vector to dicts
        angles_user, dirs_user = FeatureExtractor().vector_to_dicts(user_vector)

        # Get reference dicts directly from JSON
        angles_ref = ref_data["angles"]
        dirs_ref = ref_data["directions"]
        
        fixes = {}
        wrongs = {}
        for name in ANGLE_NAMES:
            flag = 0
            au, ar = angles_user.get(name), angles_ref.get(name)
            du, dr = dirs_user.get(name), dirs_ref.get(name)
            if au is None or ar is None:
                continue
            side, kind = self.parse_joint_meta(name)
            side_en, joint_en, distal_en = SIDE_EN[side], JOINT_LABEL_EN[kind], DISTAL_SEGMENT_EN[kind]
            diff, abs_diff, is_open = au - ar, abs(au - ar), (au - ar) > 0
            angle_instr = self.angle_action_en(kind, is_open) if abs_diff > self.threshold_deg else "Hold"
            dir_instr = self.direction_to_en(dr) if du != dr and dr is not None else "Hold direction"
            how_much = self.adverb_for_diff(abs_diff)
            cues = []
            if angle_instr != "Hold":
                cues.append(f"{angle_instr} your {name.replace('_', ' ')} {how_much}")
            if dir_instr != "Hold direction":
                cues.append(f"{'and ' if cues else ''}{dir_instr} your {side_en} {distal_en}")
            if not cues: #alls good!
                message = f"{side_en.capitalize()} {joint_en}: Nice form - hold it steady. Great work!"
                flag = 1
            else:
                message = " ".join(cues)
            angle_diff_deg = int(round(abs_diff))
            fixes[name] = {
                "angle_diff_deg": angle_diff_deg,
                "angle_action": angle_instr,
                "direction_action": dir_instr,
                "message_en": message
            }
            if flag==0: #theres a problem
                wrongs[name] = {
                    "angle_diff_deg": angle_diff_deg,
                    "angle_action": angle_instr,
                    "direction_action": dir_instr,
                    "message_en": message
                }
        #wrongs = fixes but only the problematic parts (>threshold)
        return wrongs