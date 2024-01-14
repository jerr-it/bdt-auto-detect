import argparse

from src.data_gen.training_data_generation import TrainingSet

parser = argparse.ArgumentParser(description="Train Auto-Detect")
parser.add_argument("--training_set_size", type=int, default=10000)

args = parser.parse_args()

training_set = TrainingSet()
training_set.generate_training_set(args.training_set_size)
