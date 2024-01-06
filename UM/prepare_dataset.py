import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from rich.progress import Progress

detector = tf.lite.Interpreter(model_path="../movenet.tflite")
detector.allocate_tensors()
input_details = detector.get_input_details()
output_details = detector.get_output_details()
INPUT_SIZE = input_details[0]['shape'][1]

ROOT_DIR = Path('../test') / ".."


def inference(image):
    image_resized = np.array(image).astype(np.float32)

    detector.set_tensor(input_details[0]['index'], np.expand_dims(image_resized, 0))
    detector.invoke()
    model_output = detector.get_tensor(output_details[0]['index'])[0][0]

    return model_output


dataset = ROOT_DIR / "um"

output_path = ROOT_DIR / "prepared"
output_path.mkdir(parents=True, exist_ok=True)


def mirror_image(image):
    mirror_image = cv2.flip(image, 1)
    return mirror_image


sum = 0
for dir in dataset.iterdir():
    sum += len(list(dir.iterdir()))

with (Progress(transient=False) as progress):
    poses = progress.add_task("Pose:", total=sum)
    for dir in dataset.iterdir():
        save_path = output_path / dir.name
        save_path.mkdir(parents=True, exist_ok=True)
        for file in dir.iterdir():
            image = cv2.imread(str(file))
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

            output = inference(image)

            acc: np.ndarray = output[:, 2]
            if acc.mean() > 0.2:
                head = np.mean(output[:5, 1])

                mirrored_threshold = 0.35

                if head < mirrored_threshold:
                    image = mirror_image(image)

                save_path_file = save_path / file.name

                cv2.imwrite(str(save_path_file), image)
            progress.update(poses, advance=1)
