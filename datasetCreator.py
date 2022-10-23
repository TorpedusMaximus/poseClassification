import os
import time
import matplotlib.pyplot as plt
import cv2
import csv
import tqdm
import numpy as np
import tensorflow as tf

from drawUtils import draw_keypoints, draw_connections

# This section loads the movenet_singlepose_thunder.tflite model to the interpreter.
interpreter = tf.lite.Interpreter(model_path="movenet.tflite")
interpreter.allocate_tensors()
interpreter.invoke()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = interpreter.get_input_details()[0]['shape'][1]


# crop_region = init_crop_region()
# crop_size = 0
# image_height = 0
# image_width = 0


def inference(image):
    # global image_height, image_width
    # image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)

    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, 0))
    interpreter.invoke()
    model_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return model_output


# def postprocessing(model_output, image):
#     """
#     Runs model inferece on the cropped region.
#     The function runs the model inference on the cropped region and updates the
#     model output to the original image coordinate system.
#     """
#     global image_height,image_width
#     for idx in range(17):
#         model_output[idx, 0] = \
#             (crop_region['y_min'] * image_height + crop_region['height'] * image_height * model_output[idx, 0]) / image_height
#         model_output[idx, 1] = \
#             (crop_region['x_min'] * image_width + crop_region['width'] * image_width * model_output[idx, 1]) / image_width
#
#     return model_output

dataset = 'dataset/'
outputDir = 'output/'

file = open('output/classification_dataset.csv', 'w', newline='')
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
        image = cv2.imread(imagePath)
        modelOutput = inference(image)

        fileWriter.writerow([modelOutput, label])

        image_height, image_width, _ = image.shape
        modelOutput[:, 0] *= image_height
        modelOutput[:, 1] *= image_width

        draw_keypoints(image, modelOutput)
        draw_connections(image, modelOutput)

        cv2.imwrite(outputDir + label + '_' + str(index) + '.jpg', image)
        progressBar.update(1)
        index += 1

progressBar.close()
