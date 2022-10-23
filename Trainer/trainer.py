import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

version = '1'

classifier = GaussianProcessClassifier(1.0 * RBF(1.0))

dataPath = '/home/deus/Projects/poseClassification/DatasetGenerator/output/classification_dataset.csv'

# read training file
file = open(dataPath, 'r')
data = file.read().split('\n')

for i in range(len(data)):
    row = data[i]
    data[i] = row.split(',')

data = np.array(data)

print(data)

# train classifier
classifier.fit(data[:, 0:2].astype(np.int), data[:, -1])
array = np.array(((1,4),(2,3)))
print(classifier.predict(array))

# save classifier

joblib.dump(classifier, '../classifier_v' + version + '.pkl')
