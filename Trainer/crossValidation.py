import csv
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def countLabels(set):
    count = {}
    for label in set:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    return count



dataPath = '../balanced/classification_dataset.csv'
classifierPath = f'../Classifiers/'


# read training file
file = open(dataPath, 'r')
data = file.read().split('\n')
file.close()

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

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, stratify=labels)

file = open('classifers.csv','w')
fileWriter = csv.writer(file)

for version in os.listdir(classifierPath):
    dir = os.path.join(classifierPath, version)
    score = []
    for filename in os.listdir(dir):
        classifier = joblib.load(os.path.join(dir, filename))
        score.append( classifier.score(X_test, y_test))

    score = np.array(score)
    score = score.mean()
    fileWriter.writerow((version,score))
    print(version)

file.close()
