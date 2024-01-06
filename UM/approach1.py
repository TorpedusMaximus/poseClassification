def generate_movenet_output(dataset_path, output_path):
    """
    Generates movenet model output
    1. Read the dataset images from the dataset path
    2. Inference MoveNet model on the prepared dataset images
    3. Save the output to the output path or return the output
    """
    pass


def train_classifiers(movenet_output, results_directory, classifiers):
    """
    Trains the classifiers
    1. Split dataset (movenet_output) into train and test
    2. For all the classifiers:
        2.1. Train the classifier
        2.2. Evaluate the classifier by doing the 10-fold cross validation
        2.3. Save the classifier (optional) and the evaluation results to the results directory
    """
    pass


def approach1(classifiers):
    print('Approach no 1')
    generate_movenet_output("./dataset/", "./movenet_output/")
    train_classifiers("./movenet_output/", "./results/", classifiers)
