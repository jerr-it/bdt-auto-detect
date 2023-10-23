import glob
import os

import pandas as pd
from src.data_gen.training_data_generation import TestSet

path = os.path.join(os.path.curdir, "248", "WDC", "scsv", "248")

li = []

count = 0
for filename in os.listdir(path):

    df = pd.read_csv(os.path.join(path, filename), header=0, dtype=object)

    # iterate over all columns, and remove column if the first value has more than 15 characters
    for col in df.columns:
        if len(str(df[col].iloc[0])) > 15:
            df = df.drop(col, axis=1)

    df = df.dropna(axis=1, how="all")
    li.append(df)

test_set = TestSet(li)
gen_test = test_set.generate_test_set(10000)

for t in gen_test:
    print(t)
    print("_________")
