import argparse

import dill

from src.auto_detect import AutoDetect
from src.data_gen.auto_gen_tests import generate_df_for_directory
from src.data_gen.training_data_generation import TrainingSet

parser = argparse.ArgumentParser(description="Train Auto-Detect")
parser.add_argument("--train_data_path", type=str, help="Path to training data")
parser.add_argument("--training_set_size", type=int, default=10000)
parser.add_argument("--min_precision", type=float, default=0.99)
parser.add_argument("--memory_budget", type=int, default=5*10e7) # TODO: Use fraction of hosts available memory
parser.add_argument("--training-set", type=str, help="Skips training data generation and loads training set from pickled file")

args = parser.parse_args()

dataframes = generate_df_for_directory(args.train_data_path, workers=10)
training_set = TrainingSet(dataframes)
training_set.generate_training_set(args.training_set_size)
autodetect = AutoDetect(training_set, args.min_precision, args.memory_budget)


autodetect.train()


# Remove training set from autodetect as it will be pickled separately
autodetect.trainings_set.remove_redis_connections()
dill.dump(autodetect, open("autodetect.pkl", "wb"))
