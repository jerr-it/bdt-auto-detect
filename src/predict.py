import argparse
import dill

dill.load_module("session_file.pkl")

parser = argparse.ArgumentParser(description="Predict Auto-Detect")
parser.add_argument("--predict1", type=str, help="String 1")
parser.add_argument("--predict2", type=str, help="String 2")
args = parser.parse_args()

if args.predict1 is None or args.predict2 is None:
    print("No prediction given. Exiting.")
    exit(0)

try:
    result = autodetect.predict_nonsense(args.predict1, args.predict2)
    print(result)
except Exception as e:
    print(f"Something went wrong: {e}")