import glob
import os

import pandas as pd

from src.data_gen.training_data_generation import TestSet

path = os.path.join(os.path.curdir, "248", "WDC", "scsv", "248")

dataframes = []
filenames = os.listdir(path)

for index, filename in enumerate(filenames):
    df = pd.read_csv(os.path.join(path, filename), header=0, dtype=object)

    # iterate over all columns, and remove column if the first value has more than 15 characters
    for col in df.columns:
        if df.empty or df[col].empty:
            continue

        if len(str(df[col].iloc[0])) > 15:
            df = df.drop(col, axis=1)

    df = df.dropna(axis=1, how="all")
    dataframes.append(df)

    print("Reading files... " + "{:.2f}".format(index / len(filenames) * 100) + "%")

print("Generating test set...")
test_set = TestSet(dataframes)
gen_test = test_set.generate_test_set(10000)

for t in gen_test:
    print(t)
    print("_________")
