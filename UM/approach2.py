


def balance_dataset(movenet_output, output_path):
    """
    Balances the dataset using SMOTE algorithm on the genereted MoveNet output
    1. Generate synthetic samples using SMOTE algorithm
    2. Save the balanced dataset to the output path
    """
    pass


def train_classifiers(balanced_movenet_output, results_directory, classifiers):
    """
    Trains the classifiers
    1. Split dataset (balanced_movenet_output) into train and test
    2. For all the classifiers:
        2.1. Train the classifier
        2.2. Evaluate the classifier by doing the 10-fold cross validation
        2.3. Save the classifier (optional) and the evaluation results to the results directory
    """
    pass


def approach2(classifiers):
    print('Approach no 2')
    generate_movenet_output("./dataset/", "./movenet_output/")
    balance_dataset("./movenet_output/", "./balanced_output/")
    train_classifiers("./balanced_output/", "./results/", classifiers)
