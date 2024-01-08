import shutil

from constants import ROOT_DIR
from um.movenet import generate_movenet_output
from um.oversampler import balance_dataset
from um.scikit import train_classifiers


def approach_1():
    print('Approach 1')
    dataset_path = ROOT_DIR / 'prepared'
    output_path = ROOT_DIR / "results" / "approach_1"
    output_path.mkdir(parents=True, exist_ok=True)

    keypoints = generate_movenet_output(dataset_path, output_path / 'movenet_results.csv')
    train_classifiers(keypoints, output_path / 'scikit_results.csv')


def approach_2():
    print('Approach no 2')
    dataset_path = ROOT_DIR / 'prepared'
    output_path = ROOT_DIR / "results" / "approach_2"
    output_path.mkdir(parents=True, exist_ok=True)

    keypoints = generate_movenet_output(dataset_path, output_path / 'movenet_results.csv')
    balanced = balance_dataset(keypoints, output_path / 'balanced_results.csv')
    train_classifiers(balanced, output_path / 'scikit_results.csv')


def approach_3():
    print('Approach no 3')
    dataset_path = ROOT_DIR / 'prepared'
    output_path = ROOT_DIR / "results" / "approach_3"
    output_path.mkdir(parents=True, exist_ok=True)

    gan_dataset_path = output_path / "gan"
    if gan_dataset_path.exists():
        shutil.rmtree(gan_dataset_path)
    shutil.copytree(dataset_path, gan_dataset_path)
    # generate_gan_output(gan_dataset_path)

    # keypoints = generate_movenet_output(gan_dataset_path, output_path / 'movenet_results.csv',load=False)
    # train_classifiers(keypoints, output_path / 'scikit_results.csv')


if __name__ == '__main__':
    approach_1()
    print("\n")
    approach_2()
    print("\n")
    # approach_3()
