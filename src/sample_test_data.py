import os
import random

import dill
import pandas as pd
import argparse

COLUMN_SAMPLE_TARGET = 1000


# SAMPLE_DIRECTORY = "~/sampled_columns/"


def sample_colums(base_path: str, sample_directory: str):
    folders = os.listdir(base_path)
    sampled_columns = 0
    df_list = []

    while sampled_columns < COLUMN_SAMPLE_TARGET:
        folder = random.choice(folders)

        files = [f for f in os.listdir(os.path.join(base_path, folder)) if
                 os.path.isfile(os.path.join(base_path, folder, f))]
        file = random.choice(files)

        df = pd.read_csv(os.path.join(base_path, folder, file))

        # Drop all columns that contain a value with length greater than 15
        for col in df.columns:
            if df.empty or df[col].empty:
                continue

            if len(str(df[col].iloc[0])) > 15:
                df = df.drop(col, axis=1)

        sampled_columns += df.shape[1]
        print(f"Sampled {df.shape[1]} columns. Total: {sampled_columns} of {COLUMN_SAMPLE_TARGET}")
        df_list.append(df)
    
    with open("df_list.pkl", "wb") as f:
        dill.dump(df_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge random columns from CSV files in a directory.")
    parser.add_argument("input_directory", type=str, help="Path to directory containing CSV files.")
    parser.add_argument("output_file", type=str, help="Path to output file.")

    args = parser.parse_args()

    sample_colums(args.input_directory, args.output_file)
