"""
This module contains the necessary functions to generalize
values into patterns and calculate the NPMI score of patterns.
"""
import math
import pandas as pd
import itertools

from src.stats.language import Language
from src.utils.hash_factory import hash_function
from src.utils.count_min_sketch import CountMinSketch


def convert_to_pattern(df: pd.DataFrame, language: Language) -> pd.DataFrame:
    """
    Generalizes all entries in a series by converting them all into a pattern. The pattern is created by grouping the
    characters into classes (digits, upper and lower case letters) and leaving all other characters as they are.
    (Also known as G() in the paper)
    """
    for column in df:
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(language.convert)

    return df


class PatternCountCache:
    """
    This class is used to compute and cache the count of patterns in all given columns.
    """

    def __init__(self):
        self.column_count = 0

        # TODO calculate
        depth = 8
        width = 2 ** 22

        hash_functions = [hash_function(i) for i in range(depth)]
        self.cmk = CountMinSketch(depth, width, hash_functions)

        self.dict = {}

    def pattern_occurrences(self, pattern: str) -> int:
        """
        Returns the count of a pattern.
        """
        # return self.dict[pattern]
        return self.cmk.query(pattern)

    def pattern_pair_occurrences(self, pattern1: str, pattern2: str) -> int:
        """
        Return the count of a pattern pair.
        """
        key = "Ä".join(sorted([pattern1, pattern2]))
        # return self.dict[key] if key in self.dict else 0
        return self.cmk.query(key)

    def total_length(self) -> int:
        """
        Returns the total number of columns |C|
        """
        return self.column_count

    def add_data(self, df: pd.DataFrame, language: Language) -> None:
        """
        Computes the count of all patterns in the given data and caches the result.
        # TODO complete pair calc
        """
        converted = convert_to_pattern(df, language)
        self.column_count += df.shape[1]

        for column in converted:
            column_unique = df[column].unique()

            unique_tuples = itertools.combinations(column_unique, 2)

            for pattern in column_unique:
                self.cmk.add(pattern)

                # self.dict[pattern] = self.dict.get(pattern, 0) + 1

            for combo in unique_tuples:
                key = "\0".join(sorted(combo))
                # self.dict[key] = self.dict.get(key, 0) + 1
                self.cmk.add(key)


class ValueColumnList:
    def __init__(self, cache: PatternCountCache):
        if not isinstance(cache, PatternCountCache):
            raise ValueError("Cache must be of type PatternCountCache")

        self.cache = cache

    def single_probability(self, value):
        occurrences = self.cache.pattern_occurrences(value)

        return occurrences / self.cache.total_length()

    def paired_probability(self, value1, value2):
        occurrences = self.cache.pattern_pair_occurrences(value1, value2)
        return occurrences / self.cache.total_length()

    def smoothed_probability(self, value1, value2, smoothing=0.2):
        actual_occurrences = self.cache.pattern_pair_occurrences(value1, value2)
        value1_occurrences = self.cache.pattern_occurrences(value1)
        value2_occurrences = self.cache.pattern_occurrences(value2)

        expected_occurrences = (value1_occurrences * value2_occurrences) / self.cache.total_length()
        smoothed_occurrences = (1 - smoothing) * actual_occurrences + smoothing * expected_occurrences

        return smoothed_occurrences / self.cache.total_length()

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
