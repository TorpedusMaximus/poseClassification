import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split


def countLabels(set, alllabels):
    count = {}
    for label in alllabels:
        count[label] = 0

    for label in set:
        count[label] += 1

    return count


allLabels = []
for label in os.listdir('../output'):
    if label not in allLabels:
        allLabels.append(label)

dataPath = '../output/classification_dataset.csv'

# read training file
file = open(dataPath, 'r')
data = file.read().split('\n')
dataset = []
trai_labels = []

rowIndex = 0

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
    trai_labels.append(label)

    rowIndex += 1

dataset = np.array(dataset).astype(np.float32)

labels = countLabels(trai_labels, allLabels)

# X_train, X_test, y_train, y_test = train_test_split(dataset, trai_labels, test_size=0.2, stratify=trai_labels)

# train = countLabels(y_train, allLabels)

with open('labels.csv', 'w', newline='') as file:
    fileWriter = csv.writer(file)

    for label in labels:
        fileWriter.writerow((label, labels[label]))
