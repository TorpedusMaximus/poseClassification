import cv2
import numpy as np

from DatasetGenerator.constants import *


def draw_keypoints(frame, keypoints):
    for ky, kx, kp_conf in keypoints:
        if kp_conf > THRESHOLD_CONFIDENCE:
            cv2.circle(frame, (int(kx), int(ky)), 3, (37, 130, 255), -1)

    neck = np.average([keypoints[5], keypoints[6]], axis=0)
    cv2.circle(frame, (int(neck[1]), int(neck[0])), 3, (37, 130, 255), -1)


def draw_connections(frame, keypoints):
    for p1, p2, color in EDGES_COLORS:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if (c1 > THRESHOLD_CONFIDENCE) & (c2 > THRESHOLD_CONFIDENCE):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

    neck = np.average([keypoints[5], keypoints[6]], axis=0)
    nose = keypoints[0]

    cv2.line(frame, (int(nose[1]), int(nose[0])), (int(neck[1]), int(neck[0])), color, 5)

# draw bbox
