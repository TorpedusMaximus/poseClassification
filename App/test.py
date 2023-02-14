# Imports
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


def inference(image):
    input_image = cv2.resize(image, (INPUT_SIZES, INPUT_SIZES)).astype(np.float32)
    tf_image = tf.expand_dims(input_image, 0)

    detector.set_tensor(input_details[0]['index'], tf_image)
    detector.invoke()

    model_output = detector.get_tensor(output_details[0]['index'])
    keypoints = model_output[0][0]

    keypoints[:, 0] = (keypoints[:, 0] * image_height)
    keypoints[:, 1] = (keypoints[:, 1] * image_width)

    return keypoints


def postprocess(keypoints_run_inference):
    """
    Runs model inferece on the cropped region.
    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """

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
            detected_keypoints += 1

    return detected_keypoints >= 11


def main():
    # First, we need to open the cap and establish initial crop_region.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
    else:
        global image_height, image_width
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))

        # Windows for testing on Linux (ﾉ≧ڡ≦)ﾉ
        cv2.namedWindow("Pose Classification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pose Classification", 1280, 720)

        while cap.isOpened():
            # Capture frame-by-frame. If frame is read correctly, proceed with movenet model procedures.
            ret, frame = cap.read()
            if ret:
                model_output = inference(frame)

                keypoints = postprocess(model_output)
                print(keypoints)

                draw_connections(frame, keypoints)
                draw_keypoints(frame, keypoints)

                cv2.imshow("Pose Classification", frame)

                # Press Q on keyboard to  exit
                key = cv2.waitKey(33) & 0xFF
                if key == ord("q"):
                    break

            # Break the loop if frame is read incorrectly
            else:
                break

        # Release video capture object
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
