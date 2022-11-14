import os

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def countLabels(set):
    count = {}
    for label in set:
        if label in count:
            count[label] +=1
        else:
            count[label] = 1

    return count


file = open('version', 'r')
version = file.read()
print(version)

file = open('version', 'w')
numbers = version.split('.')
file.write(f'{numbers[0]}.{numbers[1]}.{int(numbers[2])+1}')

dataPath = 'C:\\Users\\malko\\PycharmProjects\\poseClassification\\output\\classification_dataset.csv'
classifierPath = f'C:\\Users\\malko\\PycharmProjects\\poseClassification\\Classifiers\\{version}\\'

if not os.path.exists(classifierPath):
    os.mkdir(classifierPath)

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

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, stratify=labels)

print(countLabels(y_train))
print(countLabels(y_test))

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

names = [
    "NearestNeighbors",
    "RBFSVM",
    "GaussianProcess",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=10),
]

for i in range(len(classifiers)):
    classifier = classifiers[i]
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(score)
    joblib.dump(classifier, f'{classifierPath}\\{names[i]}' + version + '.pkl')
