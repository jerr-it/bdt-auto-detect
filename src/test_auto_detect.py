import argparse
import os
import pprint
from enum import Enum
from typing import Callable, Dict, List, Tuple

import numpy
import pandas as pd

from auto_detect import AutoDetect


# -------------------------------
# Models
# -------------------------------

class TestLabel(int, Enum):
    COMPATIBLE = 0
    AMBIGUOUS = 1
    INCOMPATIBLE = 2


class TestColumn(int, Enum):
    WIKIPEDIA_URL = 0
    TABLE_CAPTION = 1
    SECTION_HEADING = 2
    COLUMN_ID = 3
    COLUMN_HEADER = 4
    COLUMN_VALUES = 5
    MOST_INCOMPATIBLE_PAIR = 6
    COMPATIBILITY_SCORE = 7
    LABEL = 8


class TestFile:
    def __init__(self, file_path: str):
        self.df = read_tsv_file(file_path)
        self.name = os.path.basename(file_path)

        self.parse()

    def parse(self):
        self.fix_malformed_rows()
        self.drop_ambiguous()

        self.parse_column_values()
        self.parse_most_incompatible_pairs()
        self.parse_labels()

        self.drop_trivial()

    def fix_malformed_rows(self):
        # Identify rows where the label is missing
        malformed_rows = pd.isna(self.df[TestColumn.LABEL])

        # Shift values to the right for the identified rows
        self.df.loc[malformed_rows, self.df.columns[TestColumn.LABEL]] = self.df.loc[malformed_rows, self.df.columns[TestColumn.LABEL - 1]].astype(int)
        self.df.loc[malformed_rows, self.df.columns[TestColumn.COMPATIBILITY_SCORE]] = self.df.loc[malformed_rows, self.df.columns[TestColumn.COMPATIBILITY_SCORE - 1]].astype(float)

        # Insert NaN for the most incompatible pair
        self.df.loc[malformed_rows, self.df.columns[TestColumn.MOST_INCOMPATIBLE_PAIR]] = numpy.NAN

    def drop_ambiguous(self):
        self.df = self.df[self.df.iloc[:, TestColumn.LABEL] != TestLabel.AMBIGUOUS]

    def drop_trivial(self):
        non_trivial_mask = self.df.iloc[:, TestColumn.COLUMN_VALUES].apply(lambda values: len(values) > 1)
        self.df = self.df[non_trivial_mask]

    def parse_column_values(self):
        self.df.iloc[:, TestColumn.COLUMN_VALUES] = self.df.iloc[:, TestColumn.COLUMN_VALUES].apply(
            lambda x: numpy.NAN if pd.isna(x) else x.split("___")
        )

    def parse_most_incompatible_pairs(self):
        self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR] = self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR].apply(
            lambda x: numpy.NAN if pd.isna(x) else x.split("___")
        )

    def parse_labels(self):
        self.df.iloc[:, TestColumn.LABEL] = self.df.iloc[:, TestColumn.LABEL].apply(
            lambda x: numpy.NAN if pd.isna(x) else test_label_to_bool(x)
        )


class TestSuite:
    def __init__(self, dfs: List[pd.DataFrame]):
        # Merge data frames into one
        self.df = pd.concat(dfs).drop_duplicates(subset=0)

        # Balance label counts
        label_column = self.df.iloc[:, TestColumn.LABEL]
        compatible_count = (label_column == True).sum()
        incompatible_count = (label_column == False).sum()

        limit = min(compatible_count, incompatible_count)
        self.df = pd.concat([
            self.df[self.df[TestColumn.LABEL] == True].head(limit),
            self.df[self.df[TestColumn.LABEL] == False].head(limit)
        ])

    def test(self, classifier: Callable[[List[str]], List[str]]) -> Dict[str, float]:
        # Basic statistics
        correct_labels = 0  # Greatest importance since incompatibility should be independent of specifics

        # !!! Note that in the test format each row represents a Wikipedia column
        for _, column_data in self.df.iterrows():
            test_label = column_data.iloc[TestColumn.LABEL]
            predicted_label = self.test_column(column_data, classifier)

            # Basic statistics
            if predicted_label == test_label:
                correct_labels += 1

        # Basic statistics
        labels_precision = correct_labels / len(self.df)

        return {
            "Label precision": labels_precision
        }

    def test_column(self, column_data: pd.Series, classifier: Callable[[str, str], Tuple[bool, float]]) -> bool:
        column_values = column_data.iloc[TestColumn.COLUMN_VALUES]

        for i in range(len(column_values)):
            for j in range(i + 1, len(column_values)):
                value1 = column_values[i]
                value2 = column_values[j]
                label = classifier(value1, value2)
                if not label: return False

        return True


# -------------------------------
# Utility
# -------------------------------

def read_tsv_file(file_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path, header=None, sep='\t', na_values=['\t\t'], encoding='unicode_escape')
    return dataframe


def read_test_files(path: str) -> list[TestFile]:
    test_files = []
    for file_name in os.listdir(path):
        if not file_name.endswith('.tsv'):
            continue

        file_path = os.path.join(path, file_name)
        test_files.append(TestFile(file_path))

    return test_files


def list_to_tuple(value: list) -> tuple:
    if len(value) != 2:
        raise ValueError(f"Cannot convert list {value} to tuple")

    return value[0], value[1]


def test_label_to_bool(value: TestLabel) -> bool:
    match value:
        case TestLabel.COMPATIBLE:
            return True
        case TestLabel.INCOMPATIBLE:
            return False
        case _:
            raise ValueError(f"Ambiguous test labels cannot be converted to a boolean")


# -------------------------------
# Main
# -------------------------------

parser = argparse.ArgumentParser(description="Test Auto-Detect")
parser.add_argument("--test_path", type=str, help="Path to the folder containing labeled '.tsv'-files")
args = parser.parse_args()

# TODO: Add AutoDetect

pp = pprint.PrettyPrinter(depth=1)

test_files = read_test_files(args.test_path)
test_dfs = list(map(lambda file: file.df, test_files))
test_suite = TestSuite(test_dfs)

statistics = test_suite.test(autodetect.predict)
statistics_decline_all = test_suite.test(lambda _: [None])

print()
print(f"=== Test results ===")
print(f"Precision: {statistics['Label precision']}")
print(f"Baseline precision (allow everything): {1 - statistics_decline_all['Label precision']}")
print(f"Baseline precision (decline everything): {statistics_decline_all['Label precision']}")
