MIN_CROP_KEYPOINT_SCORE = 0.4
THRESHOLD_CONFIDENCE = 0.5

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

EDGES_COLORS = {
    (0, 1, (255, 0, 255)),
    (0, 2, (0, 255, 255)),
    (1, 3, (255, 0, 255)),
    (2, 4, (0, 255, 255)),
    (5, 7, (255, 0, 255)),
    (7, 9, (255, 0, 255)),
    (6, 8, (0, 255, 255)),
    (8, 10, (0, 255, 255)),
    (5, 6, (255, 255, 0)),
    (5, 11, (255, 0, 255)),
    (6, 12, (0, 255, 255)),
    (11, 12, (255, 255, 0)),
    (11, 13, (255, 0, 255)),
    (13, 15, (255, 0, 255)),
    (12, 14, (0, 255, 255)),
    (14, 16, (0, 255, 255))
}
