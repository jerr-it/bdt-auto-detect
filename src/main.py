import argparse
import os.path

from src.auto_detect import AutoDetect
from src.data_gen.training_data_generation import TrainingSet

parser = argparse.ArgumentParser(description="BDT Auto-Detect")
parser.add_argument("--train_data_path", type=str, help="Path to training data")
parser.add_argument("--training_set_size", type=int, default=10000)
parser.add_argument("--min_precision", type=float, default=0.75)
parser.add_argument("--memory_budget", type=int, default=10e9) # TODO: Use fraction of hosts available memory
parser.add_argument("--predict1", type=str, help="String 1")
parser.add_argument("--predict2", type=str, help="String 2")
parser.add_argument("--model_path", type=str, help="Path to model. Train data path must be given, if model file does not exist.", default="model")
args = parser.parse_args()
print(os.path.abspath(os.path.curdir))

training_set = TrainingSet.create_or_load(args.model_path, args.train_data_path, args.training_set_size)

autodetect = AutoDetect(training_set, args.min_precision, args.memory_budget)
autodetect.train()

if args.predict1 is None or args.predict2 is None:
    print("No prediction given. Exiting.")
    exit(0)

try:
    result = autodetect.predict_nonsense(args.predict1, args.predict2)
    print(result)
except Exception as e:
    print(f"Something went wrong: {e}")
