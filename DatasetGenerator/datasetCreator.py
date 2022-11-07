import csv
import os

import cv2
import numpy as np
import tensorflow as tf
import tqdm

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


def writable(modelOutput):
    writableOutput = ""
    for keypoint in modelOutput:
        x, y, conf = keypoint
        writableOutput += f"{x}:{y} "

    return writableOutput


dataset = 'C:\\Users\\malko\\PycharmProjects\\poseClassification\\small_dataset\\'
outputDir = 'C:\\Users\\malko\\PycharmProjects\\poseClassification\\output\\'

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

file = open('../output/classification_dataset.csv', 'w', newline='')
fileWriter = csv.writer(file)

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
            writableOutput = writable(modelOutput)

            fileWriter.writerow([writableOutput, label])

            image_height, image_width, _ = image.shape
            modelOutput[:, 0] *= image_height
            modelOutput[:, 1] *= image_width

            draw_keypoints(image, modelOutput)
            draw_connections(image, modelOutput)

            if not os.path.exists(outputDir + '/' + label):
                os.makedirs(outputDir + '/' + label)

            cv2.imwrite(outputDir + label + '/' + str(index) + '.jpg', image)
        except:
            print(f"Something wrong with: {imagePath}")

        progressBar.update(1)
        index += 1

progressBar.close()
