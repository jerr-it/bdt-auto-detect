import argparse
import os
import pprint
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import dill
import pandas as pd

from auto_detect import AutoDetect


# -------------------------------
# Models
# -------------------------------

class TestFile:
    def __init__(self, file_path: str):
        self.df = read_tsv_file(file_path)
        self.name = os.path.basename(file_path)

    def parse(self):
        self.drop_ambiguous()

        self.parse_column_values()
        self.parse_most_incompatible_pairs()
        self.parse_labels()

    def drop_ambiguous(self):
        self.df = self.df[self.df.iloc[:, TestColumn.LABEL] != TestLabel.AMBIGUOUS]

    def parse_column_values(self):
        self.df.iloc[:, TestColumn.COLUMN_VALUES] = self.df.iloc[:, TestColumn.COLUMN_VALUES].apply(
            lambda x: x.split("___")
        )

    def parse_most_incompatible_pairs(self):
        self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR] = self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR].apply(
            lambda x: x.split("___")
        )
        self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR] = self.df.iloc[:, TestColumn.MOST_INCOMPATIBLE_PAIR].apply(
            lambda x: list_to_tuple(x)
        )

    def parse_labels(self):
        self.df.iloc[:, TestColumn.LABEL] = self.df.iloc[:, TestColumn.LABEL].apply(
            lambda x: test_label_to_bool(x)
        )

    def test(self, classifier: Callable[[str, str], Tuple[bool, float]]) -> Dict[str, float]:
        # Basic statistics
        correct_pairs = 0  # Medium importance since the pairs may vary depending on the training data
        correct_scores = 0  # Very low importance since the scores are algorithm dependent
        correct_labels = 0  # Greatest importance since incompatibility should be independent of specifics

        # Advanced statistics
        score_delta = 0

        # !!! Note that in the test format each row represents a Wikipedia column
        for _, column_data in self.df.iterrows():
            test_pair = column_data.iloc[TestColumn.MOST_INCOMPATIBLE_PAIR]
            test_score = column_data.iloc[TestColumn.COMPATIBILITY_SCORE]
            test_label = column_data.iloc[TestColumn.LABEL]

            predicted_pair, predicted_score, predicted_label = self.test_column(column_data, classifier)

            # Basic statistics
            if predicted_pair == test_pair:
                correct_pairs += 1
            if predicted_score == test_score:
                correct_scores += 1
            if predicted_label == test_label:
                correct_labels += 1

            # Advanced statistics
            score_delta += abs(predicted_score - test_score)

        # Basic statistics
        pairs_precision = correct_pairs / len(self.df)
        scores_precision = correct_scores / len(self.df)
        labels_precision = correct_labels / len(self.df)

        return {
            "Pair precision": pairs_precision,
            "Score precision": scores_precision,
            "Score delta": score_delta,
            "Label precision": labels_precision
        }

    def test_column(self, column_data: pd.Series, classifier: Callable[[str, str], Tuple[bool, float]]) \
            -> Tuple[Tuple[str, str], float, bool]:
        column_values = column_data.iloc[TestColumn.COLUMN_VALUES]

        worst_pair = None
        worst_score = float("inf")
        worst_label = None
        for i in range(len(column_values)):
            for j in range(i + 1, len(column_values)):
                value1 = column_values[i]
                value2 = column_values[j]
                label, score = classifier(value1, value2)

                if score < worst_score:
                    worst_pair = (value1, value2)
                    worst_score = score
                    worst_label = label

        return worst_pair, worst_score, worst_label


class TestLabel(Enum):
    COMPATIBLE = 0
    AMBIGUOUS = 1
    INCOMPATIBLE = 2


class TestColumn(Enum):
    WIKIPEDIA_URL = 0
    TABLE_CAPTION = 1
    SECTION_HEADING = 2
    COLUMN_ID = 3
    COLUMN_HEADER = 4
    COLUMN_VALUES = 5
    MOST_INCOMPATIBLE_PAIR = 6
    COMPATIBILITY_SCORE = 7
    LABEL = 8


# -------------------------------
# Utility
# -------------------------------

def read_tsv_file(file_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path, sep='\t')
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
        raise ValueError(f"Cannot convert list of length {len(value)} (!= 2) to tuple")

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
parser.add_argument("--session_path", type=str, help="Path to the session file")
args = parser.parse_args()

autodetect: Optional[AutoDetect] = None
dill.load_module(args.session_path)

if autodetect is None:
    raise ValueError("Auto detect could not be loaded properly from session file")

pp = pprint.PrettyPrinter(depth=1)

test_files = read_test_files(args.test_path)
for test_file in test_files:
    statistics = test_file.test(autodetect.predict_nonsense)
    print(f"=== Testing against {test_file.name} ===")
    print(f"Statistics:")
    pp.pprint(statistics)
    print()
    print()
