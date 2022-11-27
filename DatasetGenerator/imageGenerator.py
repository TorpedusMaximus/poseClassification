import csv

import cv2
import numpy as np
import tensorflow as tf
from DatasetGenerator.drawUtils import draw_keypoints, draw_connections
from constants import KEYPOINT_DICT

# This section loads the movenet_singlepose_thunder.tflite model to the interpreter.
interpreter = tf.lite.Interpreter(model_path="../App/movenet.tflite")
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


imagePath = "poza_pokaz.png"

image = cv2.imread(imagePath)
modelOutput = inference(image)

file = open('lista.csv','w')
fileWriter = csv.writer(file)

for i in range(17):
    x,y,conf = modelOutput[i]
    name = list(KEYPOINT_DICT.keys())[i]
    fileWriter.writerow((name,x,y,conf))


image_height, image_width, _ = image.shape
modelOutput[:, 1] *= image_width
modelOutput[:, 0] *= image_height

# image = np.empty((600, 600, 3), dtype=np.int32)
# image.fill(255)


draw_connections(image, modelOutput)
draw_keypoints(image, modelOutput)

cv2.imwrite('szkielet_pokaz.png', image)


file.close()
