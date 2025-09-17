import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

ANGLE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle"
]

LANDMARK_MAP = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE
}

def calculate_angle(pA, pB, pC):
    a, b, c = np.array(pA), np.array(pB), np.array(pC)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def safe_calculate_angle(pA, pB, pC):
    try:
        return calculate_angle(pA, pB, pC)
    except Exception:
        return None

def compute_all_angles(results):
    if results is None or not hasattr(results, "pose_landmarks") or results.pose_landmarks is None:
        return {name: None for name in ANGLE_NAMES}
    lm = results.pose_landmarks.landmark
    def get_point(name):
        if name not in LANDMARK_MAP:
            return None
        idx = LANDMARK_MAP[name].value
        landmark = lm[idx]
        return (landmark.x, landmark.y)
    return {
        "left_elbow_angle": safe_calculate_angle(get_point("left_shoulder"), get_point("left_elbow"), get_point("left_wrist")),
        "right_elbow_angle": safe_calculate_angle(get_point("right_shoulder"), get_point("right_elbow"), get_point("right_wrist")),
        "left_shoulder_angle": safe_calculate_angle(get_point("left_elbow"), get_point("left_shoulder"), get_point("left_hip")),
        "right_shoulder_angle": safe_calculate_angle(get_point("right_elbow"), get_point("right_shoulder"), get_point("right_hip")),
        "left_knee_angle": safe_calculate_angle(get_point("left_hip"), get_point("left_knee"), get_point("left_ankle")),
        "right_knee_angle": safe_calculate_angle(get_point("right_hip"), get_point("right_knee"), get_point("right_ankle")),
        "left_hip_angle": safe_calculate_angle(get_point("left_shoulder"), get_point("left_hip"), get_point("left_knee")),
        "right_hip_angle": safe_calculate_angle(get_point("right_shoulder"), get_point("right_hip"), get_point("right_knee")),
    }

def get_angle_direction(a, b, c):
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    if np.linalg.norm(ab) == 0 or np.linalg.norm(cb) == 0:
        return "undefined"
    ab, cb = ab / np.linalg.norm(ab), cb / np.linalg.norm(cb)
    dx, dy = ab[0] + cb[0], ab[1] + cb[1]
    if dx == 0 and dy == 0:
        return "undefined"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx):
        return "down" if dy > 0 else "up"
    else:
        if dx > 0 and dy > 0: return "down-right"
        if dx > 0 and dy < 0: return "up-right"
        if dx < 0 and dy > 0: return "down-left"
        return "up-left"

def compute_all_angle_directions(results):
    if results is None or not hasattr(results, "pose_landmarks") or results.pose_landmarks is None:
        return {name: None for name in ANGLE_NAMES}
    lm = results.pose_landmarks.landmark
    def get_point(name):
        if name not in LANDMARK_MAP:
            return None
        idx = LANDMARK_MAP[name].value
        landmark = lm[idx]
        return (landmark.x, landmark.y)
    return {
        "left_elbow_angle": get_angle_direction(get_point("left_shoulder"), get_point("left_elbow"), get_point("left_wrist")),
        "right_elbow_angle": get_angle_direction(get_point("right_shoulder"), get_point("right_elbow"), get_point("right_wrist")),
        "left_shoulder_angle": get_angle_direction(get_point("left_elbow"), get_point("left_shoulder"), get_point("left_hip")),
        "right_shoulder_angle": get_angle_direction(get_point("right_elbow"), get_point("right_shoulder"), get_point("right_hip")),
        "left_knee_angle": get_angle_direction(get_point("left_hip"), get_point("left_knee"), get_point("left_ankle")),
        "right_knee_angle": get_angle_direction(get_point("right_hip"), get_point("right_knee"), get_point("right_ankle")),
        "left_hip_angle": get_angle_direction(get_point("left_shoulder"), get_point("left_hip"), get_point("left_knee")),
        "right_hip_angle": get_angle_direction(get_point("right_shoulder"), get_point("right_hip"), get_point("right_knee")),
    }

# === NEW COACH-TONE COMPARE ===
SIDE_EN = {"left": "left", "right": "right"}
JOINT_LABEL_EN = {"elbow": "elbow", "shoulder": "shoulder", "knee": "knee", "hip": "hip"}
DISTAL_SEGMENT_EN = {"elbow": "forearm", "shoulder": "upper arm", "knee": "shin", "hip": "knee"}

def angle_action_en(joint_kind, is_open):
    if joint_kind in ("elbow", "knee"):
        return "Bend" if is_open else "Straighten"
    if joint_kind in ("shoulder", "hip"):
        return "Close" if is_open else "Open"
    return "Adjust"

def direction_to_en(d):
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

def adverb_for_diff(d):
    if d < 8: return "slightly"
    if d < 15: return "a bit"
    return f"about {int(round(d))}°"

def parse_joint_meta(name):
    side = "left" if name.startswith("left") else "right"
    if "elbow" in name: kind = "elbow"
    elif "shoulder" in name: kind = "shoulder"
    elif "knee" in name: kind = "knee"
    else: kind = "hip"
    return side, kind

def compare_poses(angles_user, angles_ref, dirs_user, dirs_ref, threshold_deg=10.0):
    fixes = {}
    for name in ANGLE_NAMES:
        au, ar = angles_user.get(name), angles_ref.get(name)
        du, dr = dirs_user.get(name), dirs_ref.get(name)
        if au is None or ar is None:
            continue
        side, kind = parse_joint_meta(name)
        side_en, joint_en, distal_en = SIDE_EN[side], JOINT_LABEL_EN[kind], DISTAL_SEGMENT_EN[kind]
        diff, abs_diff, is_open = au - ar, abs(au - ar), (au - ar) > 0
        angle_instr = angle_action_en(kind, is_open) if abs_diff > threshold_deg else "Hold"
        dir_instr = direction_to_en(dr) if du != dr and dr is not None else "Hold direction"
        how_much = adverb_for_diff(abs_diff)
        cues = []
        if angle_instr != "Hold":
            cues.append(f"{angle_instr} {how_much}")
        if dir_instr != "Hold direction":
            cues.append(f"{'and ' if cues else ''}{dir_instr.lower()} the {distal_en}")
        if not cues:
            message = f"{side_en.capitalize()} {joint_en}: Nice form—hold it steady. Great work!"
        else:
            message = f"{side_en.capitalize()} {joint_en}: " + " ".join(cues) + ". You've got this!"
        fixes[name] = {
            "angle_diff_deg": int(round(abs_diff)),
            "angle_action": angle_instr,
            "direction_action": dir_instr,
            "message_en": message
        }
    return fixes
