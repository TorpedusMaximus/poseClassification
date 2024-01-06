def train_gan_model(dataset_path, output_path):
    """
    Trains the GAN model
    1. Read the dataset images from the dataset path
    2. Train GAN model on the dataset images
    3. Save the trained model to the output path
    """
    pass


def generate_gan_output(gan_model, dataset_path):
    """
    Generates GAN model output
    1. Generate synthetic samples using GAN model
    2. Add the generated images to the dataset path
    """
    pass


def generate_movenet_output(gan_dataset_path, output_path):
    """
    Generates movenet model output
    1. Read the dataset images from the dataset path
    2. MoveNet model
    3. Inference MoveNet model on the prepared dataset images
    4. Save the output to the output path
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


def approach3(classifiers):
    print('Approach no 3')
    train_gan_model("./dataset/", "./gan_model/")
    generate_gan_output("./gan_model/", "./dataset/")
    generate_movenet_output("./dataset/", "./movenet_output/")
    train_classifiers("./balanced_output/", "./results/", classifiers)
