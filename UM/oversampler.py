import csv
from pathlib import Path

import cv2
import numpy as np
from imblearn.over_sampling import SMOTE

from constants import ROOT_DIR
from um.draw_utils import draw_connections, draw_keypoints


def save_images(X, y) -> None:
    save_path = ROOT_DIR / "test"
    save_path.mkdir(parents=True, exist_ok=True)

    boat = [
        X[i]
        for i in range(len(X))
        if y[i] == "boat"
    ]

    for i, keypoints in enumerate(boat):
        image = np.zeros((100, 100, 3)).astype(np.uint8)

        coordinates = []
        for ii in range(17):
            coordinates.append((
                int(keypoints[2 * ii] * 100),
                int(keypoints[2 * ii + 1] * 100),
                1
            ))

        draw_keypoints(image, coordinates)
        draw_connections(image, coordinates)

        cv2.imwrite(str(save_path / f"{i}.png"), image)


def save_outputs(outputs: dict[str, list], output_path: Path) -> None:
    images_sum = 0
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for label, keypoints_list in outputs.items():
            for keypoints in keypoints_list:
                labeled = keypoints.copy()
                labeled.append(label)
                writer.writerow(labeled)
            images_sum += len(keypoints_list)
    print(f"Saved {images_sum} images to {output_path}")


def transform_dict_to_lists(movenet_output: dict[str, list]) -> tuple[list, list]:
    X = []
    y = []
    for label, keypoints_list in movenet_output.items():
        for keypoints in keypoints_list:
            X.append(keypoints)
            y.append(label)

    return X, y


def transform_lists_to_dict(X: list, y: list) -> dict[str, list]:
    output: dict[str, list] = {}
    for value, label in zip(X, y):
        if output.get(label) is None:
            output[label] = []
        output[label].append(value)

    return output


def balance_dataset(movenet_output: dict[str, list], output_path) -> dict[str, list]:
    X, y = transform_dict_to_lists(movenet_output)
    balancer = SMOTE()
    X_resampled, y_resampled = balancer.fit_resample(X, y)

    save_images(X_resampled, y_resampled)

    balanced_keypoints = transform_lists_to_dict(X_resampled, y_resampled)

    save_outputs(balanced_keypoints, output_path)

    return balanced_keypoints
