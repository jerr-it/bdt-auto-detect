import argparse
import dill
import redis

parser = argparse.ArgumentParser(description="Predict Auto-Detect")
parser.add_argument("--predict1", type=str, help="String 1")
parser.add_argument("--predict2", type=str, help="String 2")
args = parser.parse_args()

autodetect = dill.load(open("autodetect.pkl", "rb"))
autodetect.memory_budget = 10e8
autodetect.train()
autodetect.trainings_set.add_redis_connections()
if args.predict1 is None or args.predict2 is None:
    print("No prediction given. Exiting.")
    exit(0)

try:
    result = autodetect.predict(args.predict1, args.predict2)
    print(result)
except Exception as e:
    print(f"Something went wrong: {e}")