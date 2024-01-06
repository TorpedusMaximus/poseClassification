from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB





def train_classifiers(keypoints: dict[str, list], output_path):
    """
    Trains the classifiers
    2. For all the classifiers:
        2.1. Train the classifier
        2.2. Evaluate the classifier by doing the 10-fold cross validation
        2.3. Save the classifier (optional) and the evaluation results to the results directory
    """
    pass