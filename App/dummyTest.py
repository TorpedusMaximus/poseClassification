# Imports
import time

import cv2
import numpy as np
import joblib
import tensorflow as tf

from sklearn.svm import SVC

from App.constants import THRESHOLD
from App.cropUtils import crop_and_resize, init_crop_region, determine_crop_region
from App.drawUtils import draw_keypoints, draw_connections, printPose

# This section loads the movenet_singlepose_thunder.tflite model to the interpreter.
# detector = tf.lite.Interpreter(model_path="App/movenet.tflite")
detector = tf.lite.Interpreter(model_path="movenet.tflite")
detector.allocate_tensors()

# Initialize list of body keypoints, where x-position, y-position and confidence score of each body keypoint will be stored
# Cheat sheet: keypoints[(0-16)body part, 0=y 1=x 2=confidence]
input_details = detector.get_input_details()
output_details = detector.get_output_details()
INPUT_SIZES = input_details[0]['shape'][1]

classifier_path = '../Classifiers/5.0.4/SVM5.0.4.pkl'
classifier: SVC = joblib.load(classifier_path)

image_height = 0
image_width = 0


def inference(image, crop_region):
    input_image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region, crop_size=(INPUT_SIZES, INPUT_SIZES))
    input_image = tf.cast(input_image, dtype=tf.float32)
    detector.set_tensor(input_details[0]['index'], input_image.numpy())

    detector.invoke()

    keypoints_movenet = detector.get_tensor(output_details[0]['index'])
    keypoints_cropped = keypoints_movenet[0][0]

    return keypoints_cropped


def postprocess(keypoints_run_inference, crop_region):
    """
    Runs model inferece on the cropped region.
    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """

    for idx in range(17):
        keypoints_run_inference[idx, 0] = \
            (crop_region['y_min'] * image_height + crop_region['height'] * image_height * keypoints_run_inference[
                idx, 0])
        keypoints_run_inference[idx, 1] = \
            (crop_region['x_min'] * image_width + crop_region['width'] * image_width * keypoints_run_inference[
                idx, 1])

    return keypoints_run_inference


def classify(model_output):
    data = []
    for x, y, conf in model_output:
        data.append(x)
        data.append(y)
    result = classifier.predict(np.expand_dims(data, axis=0))
    print(result)

    return result[0]


def person_detected(model_output):
    detected_keypoints = 0
    for x, y, conf in model_output:
        if conf > THRESHOLD:
            detected_keypoints+=1

    return detected_keypoints >= 11


def main():
    # First, we need to open the cap and establish initial crop_region.
    cap = cv2.VideoCapture(0)

    frame = cv2.imread('00000000.jpg')

    global image_height, image_width
    image_height, image_width, _ = frame.shape
    crop_region = init_crop_region(image_height, image_width)

    model_output = inference(frame, crop_region)

    if person_detected(model_output):
        pose = classify(model_output)
        printPose(frame, pose)

    keypoints = postprocess(model_output, crop_region)

    draw_connections(frame, keypoints)
    draw_keypoints(frame, keypoints)

    cv2.imwrite("work.jpg", frame)

    cv2.namedWindow("Yoga Pose Classifier", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yoga Pose Classifier", 2*image_width, 2*image_height)

    while True:
        cv2.imshow("Yoga Pose Classifier", frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break




if __name__ == "__main__":
    main()
