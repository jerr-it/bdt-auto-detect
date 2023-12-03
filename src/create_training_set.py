import argparse

import dill

from src.data_gen.auto_gen_tests import generate_df_for_directory
from src.data_gen.training_data_generation import TrainingSet

parser = argparse.ArgumentParser(description="Train Auto-Detect")
parser.add_argument("--train_data_path", type=str, help="Path to training data")
parser.add_argument("--dataframe-pickled-path", type=str, help="Path to pickled dataframes")
parser.add_argument("--training_set_size", type=int, default=10000)

args = parser.parse_args()

if not args.dataframe_pickled_path:
    dataframes = generate_df_for_directory(args.train_data_path, workers=10)
else:
    dataframes = dill.load(open(args.dataframe_pickled_path, "rb"))
training_set = TrainingSet(dataframes)
training_set.generate_training_set(args.training_set_size)
dill.dump(training_set, open("training_set.pkl", "wb"))
