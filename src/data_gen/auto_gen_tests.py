import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

WORKERS = 10


# path = os.path.join(os.path.curdir, "248", "WDC", "scsv", "248")
# TEST_SET_SIZE = 10000
# dataframes = []


def generated_df(path_to_file):
    df = pd.read_csv(os.path.join(path_to_file), header=0, dtype=object)

    # iterate over all columns, and remove column if the first value has more than 15 characters
    for col in df.columns:
        if df.empty or df[col].empty:
            continue

        if len(str(df[col].iloc[0])) > 15:
            df = df.drop(col, axis=1)

    df = df.dropna(axis=1, how="any")
    df = df.astype(str)
    return df


def generate_df_for_directory(path, workers=10):
    dataframes = []
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generated_df, os.path.join(path, filename)) for filename in os.listdir(path)]
        for future in concurrent.futures.as_completed(futures):
            count += 1
            #print(count)
            dataframes.append(future.result())

    print("Finished generating dataframes")
    return dataframes


# Split dataframes array into WORKERS chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# def gen_test_set(dframes):
#     test_set = TrainingSet(dframes)
#     return test_set.generate_training_set(TEST_SET_SIZE)


# complete_test_set = []
# with ProcessPoolExecutor(max_workers=WORKERS) as executor:
#    futures = [executor.submit(gen_test_set, chunk) for chunk in chunks(dataframes, WORKERS)]
#
#     for future in concurrent.futures.as_completed(futures):
#         complete_test_set.extend(future.result())
