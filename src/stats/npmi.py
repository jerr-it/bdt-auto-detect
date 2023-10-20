"""
This module contains the necessary functions to generalize values into patterns and calculate the NPMI score of patterns.
"""
import math

import pandas as pd


def convert_to_pattern(series: pd.Series):
    """
    Generalizes all entries in a series by converting them all into a pattern. The pattern is created by grouping the
    characters into classes (digits, upper and lower case letters) and leaving all other characters as they are.
    (Also known as G() in the paper)
    """


class ValueColumnList:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data.values.T.tolist()  # Transpose DataFrame and convert to a list of columns
        elif all(isinstance(col, list) for col in data):
            self.data = data
        else:
            raise ValueError("Invalid input data format. Use a DataFrame or a list of columns.")

    def single_probability(self, value):
        occurrences = sum(value in col for col in self.data)
        return occurrences / len(self.data)

    def paired_probability(self, value1, value2):
        occurrences = sum((value1 in col) and (value2 in col) for col in self.data)
        return occurrences / len(self.data)

    def smoothed_probability(self, value1, value2, smoothing=0.2):
        actual_occurrences = sum((value1 in col) and (value2 in col) for col in self.data)
        value1_occurrences = sum(value1 in col for col in self.data)
        value2_occurrences = sum(value2 in col for col in self.data)
        expected_occurrences = (value1_occurrences * value2_occurrences) / len(self.data)
        smoothed_occurrences = (1 - smoothing) * actual_occurrences + smoothing * expected_occurrences
        return smoothed_occurrences / len(self.data)

    def pmi(self, value1, value2):
        denominator = self.single_probability(value1) * self.single_probability(value2)
        if denominator == 0: return float("-inf")

        ratio = self.paired_probability(value1, value2) / denominator
        return safe_log10(ratio)

    def smoothed_pmi(self, value1, value2, smoothing=0.2):
        ratio = self.smoothed_probability(value1, value2, smoothing) / (
                self.single_probability(value1) * self.single_probability(value2))
        return safe_log10(ratio)

    def npmi(self, value1, value2):
        denominator = -safe_log10(self.paired_probability(value1, value2))
        if denominator == 0: return 1

        return self.pmi(value1, value2) / denominator

    def smoothed_npmi(self, value1, value2, smoothing=0.2):
        denominator = -safe_log10(self.smoothed_probability(value1, value2, smoothing))
        if denominator == 0: return 1

        return self.smoothed_pmi(value1, value2, smoothing) / denominator

    def compatible(self, value1, value2, threshold):
        return self.npmi(value1, value2) > threshold

    def smoothed_compatible(self, value1, value2, threshold, smoothing=0.2):
        return self.smoothed_npmi(value1, value2, smoothing) > threshold


def safe_log10(value):
    if value == 0: return float("-inf")
    return math.log10(value)


# Example usage with either a DataFrame or a list of columns
data1 = {'col1': [1, 2, 1, 2, 1],
         'col2': [2, 2, 1, 2, 2]}
df = pd.DataFrame(data1)

data2 = [[1, 2, 1, 2, 1],
         [2, 2, 1, 2, 2]]
value_column_list_df = ValueColumnList(df)
value_column_list_list = ValueColumnList(data2)

# Should be compatible
print(value_column_list_df.compatible(1, 2, -0.5))

# Should be incompatible
print(value_column_list_list.compatible(1, 3, -0.5))
