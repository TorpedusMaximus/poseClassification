import os
import numpy as np

def countLabels(set):
    count = {}
    for label in set:
        if label in count:
            count[label] +=1
        else:
            count[label] = 1

    return count


allLabels = []
os.listdir('C:\\Users\\malko\\PycharmProjects\\poseClassification\\output\\')



dataPath = 'C:\\Users\\malko\\PycharmProjects\\poseClassification\\output\\classification_dataset.csv'

# read training file
file = open(dataPath, 'r')
data = file.read().split('\n')
dataset = []
labels = []

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
    labels.append(label)

    rowIndex+=1

dataset = np.array(dataset).astype(np.float32)

print(countLabels(labels))