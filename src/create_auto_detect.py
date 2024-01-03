import argparse

import dill

from src.auto_detect import AutoDetect

import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

parser = argparse.ArgumentParser(description="Train Auto-Detect")
parser.add_argument("--min_precision", type=float, default=0.9)
parser.add_argument("--memory_budget", type=int, default=5*10e7) # TODO: Use fraction of hosts available memory

args = parser.parse_args()

autodetect = AutoDetect(args.min_precision, args.memory_budget)
autodetect.train()

dill.dump(autodetect, open(f"autodetect_{slugify(str(args.min_precision))}_{slugify(str(args.memory_budget / 10e8))}.pkl", "wb"))
