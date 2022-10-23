import cv2

from constants import *


def draw_keypoints(frame, keypoints):
    for i in range(17):
        kp = keypoints[i]
        ky, kx, kp_conf = kp
        if kp_conf > THRESHOLD:
            cv2.circle(frame, (int(kx), int(ky)), 7, POINTS[i], -1)


def draw_connections(frame, keypoints):
    for p1, p2 in EDGES:
        color = EDGES[(p1, p2)]
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if (c1 > THRESHOLD) and (c2 > THRESHOLD):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

    # y1, x1, c1 = keypoints[5]
    # y2, x2, c2 = keypoints[6]
    # y3, x3, c3 = keypoints[0]
    # if c1 > THRESHOLD and c2 > THRESHOLD and c3 > THRESHOLD:
    #     x = int((x1 + x2) / 2)
    #     y = int((y1 + y2) / 2)
    #     color = (255, 255, 0)
    #     cv2.line(frame, (int(x), int(y)), (int(x3), int(y3)), color, 4)
    # elif c2 > THRESHOLD and c3 > THRESHOLD:
    #     color = (0, 255, 255)
    #     cv2.line(frame, (int(x2), int(y2)), (int(x3), int(y3)), color, 4)
    #     pass
    # elif c1 > THRESHOLD and c3 > THRESHOLD:
    #     color = (255, 0, 255)
    #     cv2.line(frame, (int(x1), int(y1)), (int(x3), int(y3)), color, 4)
    #     pass
