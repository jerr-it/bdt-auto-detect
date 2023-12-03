import argparse

import dill

from src.auto_detect import AutoDetect

parser = argparse.ArgumentParser(description="Train Auto-Detect")
parser.add_argument("--training_set_path", type=str, help="Path to training set")
parser.add_argument("--min_precision", type=float, default=0.99)
parser.add_argument("--memory_budget", type=int, default=5*10e7) # TODO: Use fraction of hosts available memory

args = parser.parse_args()

if not args.training_set_path:
    raise ValueError("Training set path must be specified!")

training_set = dill.load(open(args.training_set_path, "rb"))
autodetect = AutoDetect(training_set, args.min_precision, args.memory_budget)
autodetect.train()

# Remove training set from autodetect as it will be pickled separately
autodetect.trainings_set.remove_redis_connections()
dill.dump(autodetect, open("autodetect.pkl", "wb"))
