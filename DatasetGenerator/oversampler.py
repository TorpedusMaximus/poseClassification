import csv
import os
import shutil

import cv2
import numpy as np
import tqdm
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from DatasetGenerator.drawUtils import draw_keypoints, draw_connections


def countLabels(set):
    count = {}
    for label in set:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    return count


dataPath = '../output/classification_dataset.csv'
outputDir = '../balanced/'

if os.path.exists(outputDir):
    shutil.rmtree(outputDir)

os.mkdir(outputDir)

# read training file
file = open(dataPath, 'r')
data = file.read().split('\n')
dataset = []
labels = []
for row in data:
    xy, label = row.split(',')
    xy = xy[:-1]
    xy = xy.split(' ')

    coordinates = []
    for pair in xy:
        x, y = pair.split(':')
        coordinates.append(float(x))
        coordinates.append(float(y))
    dataset.append(coordinates)
    labels.append(label)

dataset = np.array(dataset).astype(np.float32)

balancer = SMOTE()

X_resampled, y_resampled = balancer.fit_resample(dataset, labels)

test1 = countLabels(labels)
test2 = countLabels(y_resampled)

for label in test2:
    print(f'{label}: {test1[label]} -> {test2[label]}')
    if not os.path.exists(f'{outputDir}/{label}'):
        os.mkdir(f'{outputDir}/{label}')

file = open('../balanced/classification_dataset.csv', 'w', newline='')
fileWriter = csv.writer(file)

progressBar = tqdm.tqdm(total=len(X_resampled))

for i in range(len(X_resampled)):
    label = y_resampled[i]
    sample = X_resampled[i]

    coordinates = []
    for ii in range(17):
        coordinates.append((
            sample[2*ii] * 100,
            sample[2*ii + 1] * 100,
            1
        ))

    image = np.zeros((100, 100, 3))

    draw_keypoints(image, coordinates)
    draw_connections(image, coordinates)

    cv2.imwrite(outputDir + label + '/' + str(i) + '.jpg', image)

    write = ""
    for keypoint in coordinates:
        x, y, conf = keypoint
        write += f"{x}:{y} "

    fileWriter.writerow((write,label))

    progressBar.update(1)

progressBar.close()
