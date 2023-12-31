"""
This module contains the necessary functions to generalize
values into patterns and calculate the NPMI score of patterns.
"""
import itertools
import math
import os
import numpy as np
import pandas as pd
import redis
from redis import Redis

from src.stats.language import Language
from src.utils.label import Label


def convert_to_pattern(df: pd.DataFrame, language: Language) -> pd.DataFrame:
    """
    Generalizes all entries in a series by converting them all into a pattern. The pattern is created by grouping the
    characters into classes (digits, upper and lower case letters) and leaving all other characters as they are.
    (Also known as G() in the paper)
    """
    dfcopy = df.copy()
    for column in dfcopy:
        dfcopy[column] = dfcopy[column].astype(str)
        dfcopy[column] = dfcopy[column].apply(language.convert)

    return dfcopy


class PatternCountCache:
    """
    This class is used to compute and cache the count of patterns in all given columns.
    """

    def __init__(self, language: Language, skip_db_init=False):
        self.redis = redis.Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
        self.cmk = self.redis.cms()
        if not skip_db_init:
            self.cmk.initbyprob(str(language.__hash__()), 0.00001, 0.00001)

        self.memory_usage = 2**22 * 10 * np.dtype(np.int32).itemsize

        #self.cmk = CMSketch(str(language.__hash__()), 2*20, 10)

        self.dict = {}
        self.language = language
        self.col_size = int(self.redis.get(f"total_columns"))

    def pattern_occurrences(self, pattern: str) -> int:
        """
        Returns the count of a pattern.
        """
        try:
            return self.cmk.query(str(self.language.__hash__()), pattern)[0]
            # return self.dict[pattern]
        except KeyError:
            return 0
            # raise KeyError(f"Pattern {pattern} not found in cache")

    def pattern_pair_occurrences(self, pattern1: str, pattern2: str) -> int:
        """
        Return the count of a pattern pair.
        """
        try:
            key = "Ä".join(sorted([pattern1, pattern2]))
            return self.cmk.query(str(self.language.__hash__()), key)[0]
            # return self.dict[key] if key in self.dict else 0
        except KeyError:
            return 0
            # raise KeyError(f"Pattern pair {pattern1} and {pattern2} not found in cache")

    def total_length(self) -> int:
        """
        Returns the total number of columns |C|
        """
        return self.col_size

    def add_data(self, df: pd.DataFrame) -> None:
        """
        Computes the count of all patterns in the given data and caches the result.
        # TODO complete pair calc
        """
        converted = convert_to_pattern(df, self.language)

        for column in converted:
            column_unique = converted[column].unique()

            unique_tuples = itertools.combinations_with_replacement(column_unique, 2)
            combos = ["Ä".join(sorted(combo)) for combo in unique_tuples]
            combo_increment = [1 for _ in combos]
            if len(combos) == 0: continue

            self.cmk.incrby(str(self.language.__hash__()), [p for p in column_unique], [1 for _ in column_unique])
            self.cmk.incrby(str(self.language.__hash__()), combos, combo_increment)
            # for pattern in column_unique:
            #     self.cmk.incrby(str(hash(self.language)), pattern, 1)
            #     # self.dict[pattern] = self.dict.get(pattern, 0) + 1
            #
            # for combo in unique_tuples:
            #     key = "Ä".join(sorted(combo))
            #     # self.dict[key] = self.dict.get(key, 0) + 1
            #     self.cmk.incbry(str(hash(self.language)), key, 1)

    @staticmethod
    def add_data_for_language(language: Language, df: pd.DataFrame, conn: Redis):
        converted = convert_to_pattern(df, language)

        for column in converted:
            column_unique = converted[column].unique()

            unique_tuples = itertools.combinations_with_replacement(column_unique, 2)
            combos = ["Ä".join(sorted(combo)) for combo in unique_tuples]
            combo_increment = [1 for _ in combos]
            if len(combos) == 0: continue

            conn.cms().incrby(str(language.__hash__()), [p for p in column_unique], [1 for _ in column_unique])
            conn.cms().incrby(str(language.__hash__()), combos, combo_increment)


class Scoring:
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
        denominator = self.single_probability(value1) * self.single_probability(value2)

        if denominator == 0: return float("-inf")

        ratio = self.smoothed_probability(value1, value2, smoothing) / denominator
        return safe_log10(ratio)

    def npmi(self, value1, value2):
        denominator = -safe_log10(self.paired_probability(value1, value2))
        if denominator == 0: return 1

        pmi = self.pmi(value1, value2)
        if pmi == float("-inf"): return -1

        return np.clip(pmi / denominator, 0.0, 1.0)

    def smoothed_npmi(self, value1, value2, smoothing=0.2):
        denominator = -safe_log10(self.smoothed_probability(value1, value2, smoothing))
        if denominator == 0: return 1

        smoothed_pmi = self.smoothed_pmi(value1, value2, smoothing)
        if smoothed_pmi == float("-inf"): return -1

        return np.clip(smoothed_pmi / denominator, 0.0, 1.0)

    def compatible(self, value1, value2, threshold):
        return self.npmi(value1, value2) > threshold

    def smoothed_compatible(self, value1, value2, threshold, smoothing=0.2):
        return self.smoothed_npmi(value1, value2, smoothing) > threshold


def safe_log10(value):
    if value == 0: return float("-inf")
    return math.log10(value)


def st_aggregate(conn: Redis, scoring: Scoring, min_precision: float) -> (
        float, set, set):
    """
    n -> number of training tuples
    2n + 100n
    """

    # converted_samples = [
    #     (
    #         scoring.cache.language.convert(training_sample[0]),
    #         scoring.cache.language.convert(training_sample[1]),
    #         training_sample[0],
    #         training_sample[1],
    #         training_sample[2]
    #     ) for training_sample in training_set
    # ]

    # scores = [
    #     (
    #         scoring.smoothed_npmi(training_sample[0],
    #                               training_sample[1]),
    #         (training_sample[2], training_sample[3], training_sample[4])
    #     ) for training_sample in converted_samples
    # ]

    # print(scores)

    tuple_count = int(conn.get("training_set_size"))

    h_plus_list: list[set] = [set() for _ in np.arange(1.0, -1.01, -0.01)]
    h_minus_list: list[set] = [set() for _ in np.arange(1.0, -1.01, -0.01)]

    print("Adding all tuples to hplus/minus for language:", hash(scoring.cache.language))
    for i in range(tuple_count):
        if i % 50000 == 0:
            print(i, "out of", tuple_count, "for language", hash(scoring.cache.language))
        training_smpl = tuple(conn.json().get(f"training_tuples:{i + offset}"))
        score = scoring.smoothed_npmi(
            scoring.cache.language.convert(training_smpl[0]),
            scoring.cache.language.convert(training_smpl[1]),
        )

        for threshold in np.arange(1.0, -1.01, -0.01):
            if score <= threshold:
                # detected as not compatible
                if training_smpl[2] == Label.POSITIVE.value:
                    h_plus_list[t_index(threshold)].add(training_smpl)
                else:
                    h_minus_list[t_index(threshold)].add(training_smpl)
        

    for threshold in np.arange(1.0, -1.01, -0.01):
        plus = h_plus_list[t_index(threshold)]
        minus = h_minus_list[t_index(threshold)]

        print("plus:", len(plus))
        print("minus:", len(minus))

        if len(plus) + len(minus) == 0:
            continue
        precision = len(minus) / (len(minus) + len(plus))
        print("Precision:", precision)
        if precision >= min_precision:
            print("Threshold found", str(threshold))
            return threshold, minus, plus

#    total = np.arange(-1.0, 1.1, 0.01).size
#    for idx, threshold in enumerate(np.arange(-1.0, 1.01, 0.01)):
#        h_plus = set()
#        h_minus = set()
#
#        print("Checking threshold", threshold, "Progress:", idx / total)
#        for i in range(tuple_count):
#            # Potential error if not retrieved as tuple
#            training_smpl = conn.json().get(f"training_tuples:{i}")
#            score = scoring.smoothed_npmi(
#                scoring.cache.language.convert(training_smpl[0]),
#                scoring.cache.language.convert(training_smpl[1]),
#            )
#            
#            if score <= threshold:
#                if training_smpl[2] == Label.POSITIVE.value:
#                    h_plus.add(training_smpl)
#                else:
#                    h_minus.add(training_smpl)
#
#        if len(h_minus) + len(h_plus) == 0: continue
#        precision = len(h_minus) / (len(h_minus) + len(h_plus))
#
#        if precision >= min_precision:
#            print("found threshold")
#            return threshold, h_minus, h_plus

    print("No threshold")
    raise SyntaxError("No threshold found")

thresholds = list(np.arange(1.0, -1.01, -0.01))
# threshold index
def t_index(threshold):
    global thresholds
    return thresholds.index(threshold)