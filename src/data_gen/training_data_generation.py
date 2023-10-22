from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.stats.language import G
from src.stats.npmi import ValueColumnList, PatternCountCache


class Label(Enum):
    """
    Represents a label, either + or -.
    """
    POSITIVE = "+"
    NEGATIVE = "-"


class CleanColumns:
    """
    Represents a set of clean columns C+, taken from the corpus C and verified to be clean.
    """

    def __init__(self, corpus: list[pd.DataFrame]):
        if not isinstance(corpus, list):
            raise ValueError("Corpus must be of type list[pd.DataFrame]")

        # Usually verify if data is clean, but for now we assume it is clean
        self.corpus = corpus

    def sample_column(self) -> pd.Series:
        """
        Returns a random column from the corpus.
        """
        df = np.random.choice(self.corpus)
        return np.random.choice(df)


class TestSet:
    """
    Represents a test set, consisting of a list of tuples (u, v, +/-).
    """

    def __init__(self, corpus: list[pd.DataFrame]):
        self.cache = PatternCountCache()
        for df in corpus:
            self.cache.add_data(df, G)

        self.vcl = ValueColumnList(self.cache)
        self.columns = CleanColumns(corpus)

    def generate_clean_test_set(self, size: int, samples_per_iteration=1) -> list[tuple[Any, Any, Label]]:
        """
        Used to generate a clean test set.
        T+ = (u, v, +) | u, v in C, C in C+
        """
        tuples_generated = 0
        result = []

        while tuples_generated < size:
            C = self.columns.sample_column()
            for i in range(samples_per_iteration):
                result.append((C.sample(), C.sample(), Label.POSITIVE))
                tuples_generated += 1

        return result

    def generate_dirty_test_set(self, size: int, samples_per_iteration=1) -> list[tuple[Any, Any, Label]]:
        """
        Used to generate a dirty test set.
        T- = (u, v, -) | u in C1, v in C2, C1, C2 in C+, C1 and C2 are incompatible
        """
        tuples_generated = 0
        result = []

        while tuples_generated < size:
            C1 = self.columns.sample_column()
            C2 = self.columns.sample_column()

            if not self.is_compatible(C1, C2):
                continue

            if C1 == C2:
                continue

            for i in range(samples_per_iteration):
                result.append((C1.sample(), C2.sample(), Label.NEGATIVE))
                tuples_generated += 1

        return result

    def generate_test_set(self, size: int) -> list[tuple[Any, Any, Label]]:
        """
        Used to generate a mixed test set consisting of T+ (clean test set) and T- (dirty test set).
        T = T+ U T-
        """
        clean_test_set = self.generate_clean_test_set(size // 2)
        dirty_test_set = self.generate_dirty_test_set(size // 2)
        return clean_test_set + dirty_test_set

    def is_compatible(self, c1: pd.Series, c2: pd.Series) -> bool:
        """
        Used to check if two columns are compatible.
        """
        for e1 in c1:
            for e2 in c2:
                if not self.vcl.compatible(G.convert(e1), G.convert(e2), G.threshold):
                    return False

        return True

