import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from rich.progress import Progress

from constants import ROOT_DIR


class Movenet:
    def __init__(self, path=str(ROOT_DIR / "movenet.tflite")):
        self.detector = tf.lite.Interpreter(model_path=path, num_threads=16)
        self.detector.allocate_tensors()
        self.input_details = self.detector.get_input_details()
        self.output_details = self.detector.get_output_details()

    def inference(self, image: np.ndarray) -> list:
        image_resized = np.array(image).astype(np.float32)

        self.detector.set_tensor(self.input_details[0]['index'], np.expand_dims(image_resized, 0))
        self.detector.invoke()
        model_output = self.detector.get_tensor(self.output_details[0]['index'])[0][0]
        output = [
            item
            for row in model_output
            for item in row[:2]
        ]
        return output


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

def load_keypoints_from_csv(path: Path) -> dict[str, list]:
    movenet_output = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        line: list
        for line in reader:
            if not line:
                continue
            label = line[-1]
            keypoints_str = line[:-1]
            keypoints = [float(x) for x in keypoints_str]
            if label not in movenet_output.keys():
                movenet_output[label] = []
            movenet_output[label].append(keypoints)

    print("Loaded Movenet output from csv")
    return movenet_output
def generate_movenet_output(dataset_path: Path, output_path: Path, load=True) -> dict[str, list]:
    """
    Generates movenet model output
    1. Read the dataset images from the dataset path
    2. Inference MoveNet model on the prepared dataset images
    3. Save the output to the output csv
    4. Return a dictionary with labels and list of keypoints for each image
    """

    if output_path.exists() and load:
        return load_keypoints_from_csv(output_path)

    model = Movenet()
    images_sum = 0
    for label in dataset_path.iterdir():
        images_sum += len(list(label.iterdir()))

    outputs: dict[str, list] = {}
    with Progress() as progress:
        files = progress.add_task("Calculating keypoints:", total=images_sum)
        for label in dataset_path.iterdir():
            outputs[label.name] = []
            for file in label.iterdir():
                image = cv2.imread(str(file))
                output = model.inference(image)
                outputs[label.name].append(output)
                progress.update(files, advance=1)

    save_outputs(outputs, output_path)

    return outputs
