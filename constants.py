from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent

FOLDS = 10
np.random.seed(420)

CLASS_TO_NUMBER = {
    0: "balance",
    1: "boat",
    2: "bow",
    3: "bridge",
    4: "butterfly",
    5: "camel",
    6: "cat",
    7: "chair",
    8: "cobra",
    9: "cow",
    10: "crow",
    11: "downdog",
    12: "embryo_in_womb",
    13: "extended_hand-to-big-toe",
    14: "extended_side_angle",
    15: "garland",
    16: "godess",
    17: "half_moon",
    18: "handstand",
    19: "happy_baby",
    20: "high_lunge",
    21: "legs-up-the-wall",
    22: "lord_of_the_dance",
    23: "lotus",
    24: "one-legged_king_pigeon",
    25: "pigeon",
    26: "plank",
    27: "pyramid",
    28: "scorpion",
    29: "seated_forward_bend",
    30: "splits",
    31: "staff",
    32: "standing_forward_bend",
    33: "tree",
    34: "warrior",
}
