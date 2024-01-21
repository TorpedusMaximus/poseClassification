import csv
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from constants import CLASS_TO_NUMBER


def load_movenet_model_from_disk(model_path):
    # Load the MoveNet model from the saved model format
    movenet_model = tf.keras.models.load_model(model_path, compile=False)

    return movenet_model


def extract_keypoints(image, movenet_model):
    # Convert image to TensorFlow tensor
    input_details = movenet_model.get_input_details()
    output_details = movenet_model.get_output_details()
    input_sizes = input_details[0]['shape'][1]
    image = cv2.resize(image, (input_sizes, input_sizes)).astype(np.float32)
    image = tf.expand_dims(image, 0)
    image = np.float32(image)
    movenet_model.set_tensor(input_details[0]['index'], image)
    movenet_model.invoke()

    model_output = movenet_model.get_tensor(output_details[0]['index'])

    # Run MoveNet on the input image
    keypoints = model_output[0][0]

    return keypoints


def count_keypoints_above_threshold(keypoints):
    # Example rating mechanism: Count the number of keypoints above a certain y-coordinate
    threshold_y = 0.1

    keypoints_number_above_threshold = 0

    for keypoint in keypoints:
        if keypoint[2] > threshold_y:
            keypoints_number_above_threshold += 1

    return keypoints_number_above_threshold


def load_epoch_class_images(epoch_number, class_name, root_dir="./images/"):
    images_list = []

    epoch_path = os.path.join(root_dir, f"epoch_{epoch_number}")
    class_path = os.path.join(epoch_path, class_name)

    if os.path.exists(class_path) and os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path)
            img_array = np.array(img)
            images_list.append(img_array)

    return images_list


def main():
    # movenet_model = tf.lite.Interpreter(model_path="../movenet.tflite")
    # movenet_model.allocate_tensors()
    # for epoch_number in range(100, 3800, 100):
    #     for class_number in CLASS_TO_NUMBER:
    #         class_name = CLASS_TO_NUMBER[class_number]
    #         images_list: list = load_epoch_class_images(epoch_number, class_name)
    #         for index, image_array in enumerate(images_list):
    #             keypoints = extract_keypoints(image_array, movenet_model)
    #             keypoints_above_threshold = count_keypoints_above_threshold(keypoints)
    #             if not os.path.exists(f"csvs_movenet_thunder"):
    #                 os.mkdir(f"csvs_movenet_thunder")
    #             if not os.path.exists(f"csvs_movenet_thunder/{epoch_number}"):
    #                 os.mkdir(f"csvs_movenet_thunder/{epoch_number}")
    #             with open(f"csvs_movenet_thunder/{epoch_number}/{class_name}.csv", 'a', newline='') as csvfile:
    #                 csvwriter = csv.writer(csvfile, delimiter=';')
    #                 csvwriter.writerow([index, keypoints_above_threshold])

    # highest_for_every_class = {}
    #
    # for epoch_dir in os.listdir("csvs_movenet_thunder/"):
    #     for file_class_name in os.listdir(f"csvs_movenet_thunder/{epoch_dir}"):
    #         total_sum = 0
    #         with open(f"csvs_movenet_thunder/{epoch_dir}/{file_class_name}", 'r') as csvfile:
    #             csvreader = csv.reader(csvfile, delimiter=';')
    #             for row in csvreader:
    #                 total_sum += float(row[1])
    #
    #             if epoch_dir != "100":
    #                 x = int(float(highest_for_every_class[file_class_name.split(".")[0]].split(",")[1]))
    #                 if total_sum >= x:
    #                     highest_for_every_class.update({file_class_name.split(".")[
    #                                                         0]: f"{epoch_dir},{total_sum},{round(total_sum / (17 * 50), 3)}%"})
    #                     print(f"Total sum for {epoch_dir}/{file_class_name}:", total_sum)
    #             else:
    #                 highest_for_every_class.update(
    #                     {file_class_name.split(".")[0]: f"{epoch_dir},{total_sum},{round(total_sum / (17 * 50), 3)}%"})
    #                 print(f"Total sum for {epoch_dir}/{file_class_name}:", total_sum)

    output_dict = {}

    for class_number in CLASS_TO_NUMBER:
        results_for_class = {}
        for epoch_dir in os.listdir("csvs_movenet_thunder/"):
            csv_class_name =f"{CLASS_TO_NUMBER[class_number]}.csv"
            if csv_class_name in os.listdir(f"csvs_movenet_thunder/{epoch_dir}"):
                with open(f"csvs_movenet_thunder/{epoch_dir}/{csv_class_name}", 'r') as csvfile:
                    total_sum_for_epoch = 0
                    csvreader = csv.reader(csvfile, delimiter=';')
                    for row in csvreader:
                        total_sum_for_epoch += float(row[1])
                results_for_class.update({int(epoch_dir):total_sum_for_epoch})

        results_for_class_sorted = sorted(results_for_class.items(), key=lambda x:x[1], reverse=True)
        epoch_results_sorted = [result[0] for result in results_for_class_sorted]
        output_dict.update({CLASS_TO_NUMBER[class_number]: epoch_results_sorted})

    epochs = range(100, 3800, 100)
    second_output_dict = {epoch:0 for epoch in epochs}
    for epoch_list in output_dict.values():
        for epoch in epochs:
            second_output_dict[epoch] += epoch_list.index(epoch)

    results_for_class_sorted = sorted(second_output_dict.items(), key=lambda x: x[1])
    print(results_for_class_sorted)
    pass


    # highest_for_every_class = json.dumps(highest_for_every_class, indent=1, sort_keys=True)
    #
    # # Print the pretty printed JSON
    # print(highest_for_every_class)


if __name__ == "__main__":
    main()
