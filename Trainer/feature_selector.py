import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

dataPath = '../output/classification_dataset.csv'

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

estimator = QuadraticDiscriminantAnalysis()
selector = RFE(estimator, n_features_to_select=24, step=1)

selector = selector.fit(X_train, y_train)
# rfe_mask = selector.support_  # list of booleans for selected features
#
# new_features = []
# for bool, feature in zip(rfe_mask, X_train.columns):
#     if bool:
#         new_features.append(feature)
#         new_features  # The list of your 5 best features

print(selector.get_feature_names_out())
