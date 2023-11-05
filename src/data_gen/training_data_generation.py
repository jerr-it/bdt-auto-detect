from __future__ import annotations

import concurrent.futures
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import Any
import dill
from pathos.multiprocessing import ProcessingPool as Pool

import pandas as pd

from src.stats.language import G
from src.stats.npmi import Scoring, PatternCountCache
from src.stats.language import L
from src.utils.label import Label

MAX_WORKERS = 20


class CleanColumns:
    """
    Represents a set of clean columns C+, taken from the corpus C and verified to be clean.
    """

    def __init__(self, corpus: list[pd.DataFrame]):
        if not isinstance(corpus, list):
            raise ValueError("Corpus must be of type list[pd.DataFrame]")

        # Usually verify if data is clean, but for now we assume it is clean
        for corp in corpus:
            print("------------------------")
            print(corp)
        self.corpus = [df.copy() for df in corpus]

    def sample_column(self) -> pd.Series:
        """
        Returns a random column from the corpus.
        """
        df = random.choice(self.corpus)

        while len(df.columns) == 0:
            df = random.choice(self.corpus)

        col = random.choice(df.columns)
        return df[col]


class TrainingSet:
    """
    Represents a training set, consisting of a list of tuples (u, v, +/-).
    """

    def calculate_language_cache(self, language):
        cache = PatternCountCache(language)
        self.caches[language] = cache
        self.scorings[language] = Scoring(cache)

    def __init__(self, corpus: list[pd.DataFrame]):
        self.caches = {}
        self.scorings = {}
        global MAX_WORKERS

        for index, language in enumerate(L):
            cache = PatternCountCache(language)
            self.caches[language] = cache
            self.scorings[language] = Scoring(cache)
            print("Progress:" + str(index / len(L)))

        # with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = [executor.submit(self.calculate_language_cache, lang) for lang in L]
        #
        #     for future in concurrent.futures.as_completed(futures):
        #         future.result()

        self.cache = PatternCountCache(G)
        for _index, df in enumerate(corpus):
            self.cache.add_data(df)
            for index, (language, cache) in enumerate(self.caches.items()):
                cache.add_data(df)
                print("Adding data... " + str(index / len(self.caches)))

            print("Adding corpus... " + str(_index / len(corpus)))

        self.scoring = Scoring(self.cache)
        self.columns = CleanColumns(corpus)
        self.tuples = []

    def generate_clean_training_set(self, size: int, samples_per_iteration=1) -> list[tuple[str, str, Label]]:
        """
        Used to generate a clean test set.
        T+ = (u, v, +) | u, v in C, C in C+
        """
        tuples_generated = 0
        result = []

        while tuples_generated < size:
            original = self.columns.sample_column()
            for i in range(samples_per_iteration):
                result.append((str(original.sample().to_numpy()[0]), str(original.sample().to_numpy()[0]), Label.POSITIVE))
                tuples_generated += 1

        return result

    def generate_dirty_training_set(self, size: int, samples_per_iteration=1) -> list[tuple[str, str, Label]]:
        """
        Used to generate a dirty test set.
        T- = (u, v, -) | u in C1, v in C2, C1, C2 in C+, C1 and C2 are incompatible
        """
        tuples_generated = 0
        result = []

        while tuples_generated < size:
            o1 = self.columns.sample_column()
            C1 = o1.copy().apply(G.convert)
            o2 = self.columns.sample_column()
            C2 = o2.copy().apply(G.convert)

            if C1.equals(C2):
                continue

            if self.is_compatible(C1, C2):
                continue

            for i in range(samples_per_iteration):
                result.append((str(o1.sample(1).to_numpy()[0]), str(o2.sample(1).to_numpy()[0]), Label.NEGATIVE))
                tuples_generated += 1

        return result

    def generate_training_set(self, size: int) -> list[tuple[str, str, Label]]:
        """
        Used to generate a mixed test set consisting of T+ (clean test set) and T- (dirty test set).
        T = T+ U T-
        """
        clean_test_set = self.generate_clean_training_set(size // 2)
        dirty_test_set = self.generate_dirty_training_set(size // 2)
        self.tuples = clean_test_set + dirty_test_set
        return self.tuples

    def is_compatible(self, c1: pd.Series, c2: pd.Series) -> bool:
        """
        Used to check if two columns are compatible.
        """
        # TODO only take one random value from C1 and check against every value of C2
        for e1 in c1:
            for e2 in c2:
                if not self.scoring.compatible(e1, e2, G.threshold):
                    return False

        return True

    # TODO change to dill, pickle wont pickle
    def save(self, filename: str):
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str) -> TrainingSet:
        with open(f"{filename}.pkl", "rb") as f:
            return pickle.load(f)
