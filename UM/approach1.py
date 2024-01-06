from pathlib import Path

from um.movenet import generate_movenet_output
from um.scikit import train_classifiers

ROOT_DIR = Path(__file__).resolve().parent

def approach_1():
    print('Approach 1')
    dataset_path = ROOT_DIR / 'prepared'
    output_path = ROOT_DIR / "results" / "approach_1"
    output_path.mkdir(parents=True, exist_ok=True)

    keypoints = generate_movenet_output(dataset_path, output_path / 'movenet_results.json')
    train_classifiers(keypoints, output_path / 'scikit_results.csv')
