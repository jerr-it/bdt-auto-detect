"""
This module contains the necessary functions to generalize
values into patterns and calculate the NPMI score of patterns.
"""
import math
import pandas as pd

import src.stats.language as language


def convert_to_pattern(series: pd.Series):
    """
    Generalizes all entries in a series by converting them all into a pattern. The pattern is created by grouping the
    characters into classes (digits, upper and lower case letters) and leaving all other characters as they are.
    (Also known as G() in the paper)
    """
    G = language.G
    series_copy = series.copy()  # Maybe don't need copy
    return series_copy.apply(G)


class PatternCountCache:
    """
    This class is used to compute and cache the count of patterns in all given columns.
    """

    def __init__(self):
        self.cache: dict[int] = {}
        self.total_columns = 0
        self.columns: list[pd.Series] = []

    def pattern_occurrences(self, pattern: str) -> int | None:
        """
        Returns the count of a pattern if it is cached, otherwise returns None.
        """
        ...

    def pattern_pair_occurrences(self, pattern1: str, pattern2: str) -> int:
        """
        Computes the count of a patter pair on the fly.
        """
        ...

    def total_length(self) -> int:
        """
        Returns the total length of all columns.
        """
        ...

    def add_data(self, pattern: str, columns: list[pd.Series] | pd.DataFrame | pd.Series) -> None:
        """
        Computes the count of all patterns in the given data and caches the result.
        """
        ...


class ValueColumnList:
    def __init__(self, cache: PatternCountCache):
        if isinstance(cache, PatternCountCache):
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
