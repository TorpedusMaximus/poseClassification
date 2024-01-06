import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from rich.progress import Progress


class Movenet:
    def __init__(self, path="../movenet.tflite"):
        self.detector = tf.lite.Interpreter(model_path=path)
        self.detector.allocate_tensors()
        self.input_details = self.detector.get_input_details()
        self.output_details = self.detector.get_output_details()

    def inference(self, image: np.ndarray) -> list:
        image_resized = np.array(image).astype(np.float32)

        self.detector.set_tensor(self.input_details[0]['index'], np.expand_dims(image_resized, 0))
        self.detector.invoke()
        model_output = self.detector.get_tensor(self.output_details[0]['index'])[0][0]

        return model_output


def save_outputs(outputs: dict[str, list], output_path: Path) -> None:
    with open(output_path, 'w') as f:
        json.dump(outputs, f)


def generate_movenet_output(dataset_path: Path, output_path: Path) -> dict[str, list]:
    """
    Generates movenet model output
    1. Read the dataset images from the dataset path
    2. Inference MoveNet model on the prepared dataset images
    3. Save the output to the output path
    """

    model = Movenet()
    images_sum = 0
    for label in dataset_path.iterdir():
        images_sum += len(list(label.iterdir()))

    outputs: dict[str, list] = {}

    with Progress(transient=False) as progress:
        files = progress.add_task("Calculating keypoints:", total=images_sum)
        for label in dataset_path.iterdir():
            outputs[label.name] = []
            for file in label.iterdir():
                image = cv2.imread(str(file))
                output = model.inference(image)
                outputs[label.name].append(output)
            progress.update(files, advance=1)

    return outputs
