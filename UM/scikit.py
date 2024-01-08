import csv

import numpy as np
from rich import print
from rich.progress import Progress
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from constants import FOLDS
from um.binarization import simple_binarization
from um.oversampler import transform_dict_to_lists, transform_lists_to_dict


def score(classifiers: list, X_test, y_test, progress, classifier_name) -> float:
    classifier: SVC | GaussianNB | KNeighborsClassifier
    name: str
    correct_classifications = 0
    scoring_progress = progress.add_task(f"Scoring {classifier_name}", total=len(y_test))
    for X, y in zip(X_test, y_test):
        for classifier, label in classifiers:
            prediction = classifier.predict([X])
            if prediction == y:
                correct_classifications += 1
            if prediction == ".":
                continue
            else:
                break

        progress.update(scoring_progress, advance=1)

    accuracy = correct_classifications / len(y_test)
    return accuracy


def save_result_scores(result_scores, output_path):
    results = []
    for classifier_name in result_scores.keys():
        for i, accuracy in enumerate(result_scores[classifier_name]):
            result = {
                "fold": i,
                "classifier_name": classifier_name,
                "accuracy": accuracy,
            }
            results.append(result)

    with open(output_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def train_classifiers(keypoints: dict[str, list], output_path):
    """
    Trains the classifiers
    1  Binarization
    2. k-fold train and validation
    3. Save the classifiers (optional) and the evaluation results to the results directory
    """
    X, y = transform_dict_to_lists(keypoints)

    classifiers = {
        "kNN": (KNeighborsClassifier, {'n_neighbors': 3, "n_jobs": -1}),
        "SVM": (SVC, {"gamma": 2, "C": 1}),
        "NB": (GaussianNB, {}),
    }

    result_scores = {name: [] for name in classifiers.keys()}

    random_states = np.random.randint(0, np.iinfo(np.int32).max, size=FOLDS)

    with Progress() as progress:
        fold_progress = progress.add_task("Fold", total=len(random_states))
        for random_state in random_states:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state,
                                                                stratify=y)
            keypoints_train = transform_lists_to_dict(X_train, y_train)
            binarized_dataset = simple_binarization(keypoints_train)

            for classifier_name, classifier_data in classifiers.items():
                classifier_class, params = classifier_data
                classifiers_list = []
                for X_train_binarized, y_train_binarized, label in binarized_dataset:
                    classifier = classifier_class(**params)
                    classifier.fit(X_train_binarized, y_train_binarized)
                    classifiers_list.append([classifier, label])

                result_score = score(classifiers_list, X_test, y_test, progress, classifier_name)
                result_scores[classifier_name].append(result_score)

            progress.update(fold_progress, advance=1)

    save_result_scores(result_scores, output_path)
    print(result_scores)
