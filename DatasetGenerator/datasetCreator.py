import csv
import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
import tqdm

from constants import THRESHOLD_CONFIDENCE

from DatasetGenerator.drawUtils import draw_keypoints, draw_connections

# This section loads the movenet_singlepose_thunder.tflite model to the interpreter.
interpreter = tf.lite.Interpreter(model_path="movenet.tflite")
interpreter.allocate_tensors()
interpreter.invoke()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = interpreter.get_input_details()[0]['shape'][1]


def inference(image):
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, 0))
    interpreter.invoke()
    model_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return model_output


def writeResult(data):
    writableOutput = ""
    count_errors = 0
    for keypoint in data:
        x, y, conf = keypoint
        if conf < THRESHOLD_CONFIDENCE:
            count_errors += 1
        writableOutput += f"{x}:{y} "

    if count_errors < 5:
        return writableOutput
    else:
        return False


dataset = '../dataset/'
outputDir = '../output/'

if os.path.exists(outputDir):
    shutil.rmtree(outputDir)

os.mkdir(outputDir)

file = open('../output/classification_dataset.csv', 'w', newline='')
fileWriter = csv.writer(file)

file = open('../output/errors.csv', 'w', newline='')
errorWriter = csv.writer(file)

total = 0

for label in os.listdir(dataset):
    labelDir = os.path.join(dataset + label)
    for filename in os.listdir(labelDir):
        total += 1

print("Found % d images" % total)

progressBar = tqdm.tqdm(total=total)

for label in os.listdir(dataset):
    labelDir = os.path.join(dataset + label)
    index = 0
    for filename in os.listdir(labelDir):
        imagePath = os.path.join(labelDir, filename)
        try:
            image = cv2.imread(imagePath)
            modelOutput = inference(image)
            writableOutput=writeResult(modelOutput)

            image_height, image_width, _ = image.shape
            modelOutput[:, 0] *= image_height
            modelOutput[:, 1] *= image_width

            draw_keypoints(image, modelOutput)
            draw_connections(image, modelOutput)

            if not os.path.exists(outputDir + '/' + label):
                os.makedirs(outputDir + '/' + label)


            if writableOutput:
                fileWriter.writerow([writableOutput, label])
                cv2.imwrite(outputDir + label + '/' + str(index) + '.jpg', image)
            else:
                errorWriter.writerow([filename])
        except:
            print(f"Something wrong with: {imagePath}")

        progressBar.update(1)
        index += 1

progressBar.close()
