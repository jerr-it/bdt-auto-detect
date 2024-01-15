from __future__ import annotations

import os
import os.path
import random
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent
import multiprocessing

import dill

import pandas as pd
import numpy as np
import redis
from redis import Redis

from src.data_gen.auto_gen_tests import generate_df_for_directory

from src.stats.language import G
from src.stats.language import Language
from src.stats.npmi import Scoring, PatternCountCache
from src.stats.language import L
from src.utils.label import Label

MAX_WORKERS = 20
SAMPLES_PER_ITERATION = 20

class CleanColumns:
    """
    Represents a set of clean columns C+, taken from the corpus C and verified to be clean.
    """

    def __init__(self, conn: Redis):
        self.conn = conn

    def sample_column(self) -> pd.Series:
        """
        Returns a random column from the corpus.
        """
        df_count = int(self.conn.get("corpus_length"))

        df = pd.DataFrame()
        while len(df.columns) == 0:
            df_random_index = random.randint(0, df_count - 1)
            df_dict = self.conn.json().get(f"corpus:{df_random_index}", "$")[0]
            df = pd.DataFrame.from_dict(df_dict)
        
        col = random.choice(df.columns)
        return df[col].astype(str)


class TrainingSet:
    """
    Represents a training set, consisting of a list of tuples (u, v, +/-).
    """

    def calculate_language_cache(self, language):
        cache = PatternCountCache(language)
        self.caches[language] = cache
        self.scorings[language] = Scoring(cache)

    def __init__(self, populate_cms=False):
        self.conn = redis.Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", ""))

        self.caches = {}
        self.scorings = {}
        global MAX_WORKERS

        self.columns = CleanColumns(self.conn)
        self.cache = PatternCountCache(G)
        self.scoring = Scoring(self.cache)
        self.tuples = []

        print("Creating pattern count caches ...")
        for index, language in enumerate(L):
            cache = PatternCountCache(language)
            print(f"Creating count cache for language {str(cache.language)}")
            self.caches[language] = cache
            self.scorings[language] = Scoring(cache)
            cache.redis = Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
            cache.cmk = cache.redis.cms()

        if not populate_cms: return
        
        corpus_length = int(self.conn.get("corpus_length"))
        max_workers = os.getenv("MAX_WORKERS", 20)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [
                executor.submit(fill_cache_worker, start, end, L) for start, end in chunks(range(corpus_length), corpus_length // max_workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def generate_clean_training_set(self, size: int, samples_per_iteration=1) -> list[tuple[str, str, Label]]:
        """
        Used to generate a clean test set.
        T+ = (u, v, +) | u, v in C, C in C+
        """
        tuples_generated = 0
        result = []
        global TUPLE_IDX

        while tuples_generated < size:
            print(f"Sampling columns for clean set ... {tuples_generated} of {size}")
            original = self.columns.sample_column()
            if len(original) == 0:
                continue
            for _ in range(samples_per_iteration):
                result.append((str(original.sample().to_numpy()[0]), str(original.sample().to_numpy()[0]), Label.POSITIVE))
                tuples_generated += 1


        return result

    def generate_dirty_training_set(self, size: int, samples_per_iteration=10) -> list[tuple[str, str, Label]]:
        """
        Used to generate a dirty test set.
        T- = (u, v, -) | u in C1, v in C2, C1, C2 in C+, C1 and C2 are incompatible
        """
        result = []
        print("Starting to generate dirty training set ...")

        #with ProcessPoolExecutor(max_workers=int(os.getenv("WORKERS", 20))) as executor:
        #    futures = [executor.submit(generate_dirty_training_set_worker, self.columns, size // MAX_WORKERS, samples_per_iteration) for _ in range(MAX_WORKERS)]
        #for future in as_completed(futures):
        #    result.extend(future.result())
        result.extend(generate_dirty_training_set_worker(self.columns, size, samples_per_iteration=SAMPLES_PER_ITERATION))
        print("Finished generating dirty training set")
        return result

    def generate_training_set(self, size: int) -> list[tuple[str, str, Label]]:
        """
        Used to generate a mixed test set consisting of T+ (clean test set) and T- (dirty test set).
        T = T+ U T-
        """
        print("Starting to generate training set ...")
        clean_test_set = self.generate_clean_training_set(size // 2)
        dirty_test_set = self.generate_dirty_training_set(size // 2)
        self.tuples = clean_test_set + dirty_test_set

        for idx, t in enumerate(self.tuples):
            print("Adding: ", idx, t)
            self.conn.json().set(f"training_tuples:{idx}", "$", t)
        
        self.conn.set("training_set_size", len(self.tuples))

        return self.tuples

    def is_compatible(self, c1: pd.Series, c2: pd.Series) -> bool:
        """
        Used to check if two columns are compatible.
        """
        # TODO only take one random value from C1 and check against every value of C2
        if len(c1) > len(c2):
            c1, c2 = c2, c1

        sample_value = c1.sample().to_numpy()[0]
        for e2 in c2:
            if not self.scoring.smoothed_compatible(sample_value, e2, G.threshold):
                return False

        return True

    @staticmethod
    def is_compatible_worker(c1: pd.Series, c2: pd.Series) -> bool:
        """
        Used to check if two columns are compatible.
        """
        scoring = Scoring(PatternCountCache(G, skip_db_init=True))
        if len(c1) == 0 or len(c2) == 0: return True # returns True so that generate_dirty skips this
        if len(c1) > len(c2):
            c1, c2 = c2, c1

        sample_value = c1.sample().to_numpy()[0]
        for e2 in c2:
            if not scoring.smoothed_compatible(sample_value, e2, G.threshold):
                return False

        return True
    
    # This method returns the "opposite" (but not really) of what 'is_compatible_worker' used to return.
    # A value 'u' should only be combined with a value from series 'c' if this method returns 'True'.
    @staticmethod
    def can_combine_to_dirty_column(c: pd.Series, u) -> bool:
        scoring = Scoring(PatternCountCache(G, skip_db_init=True))
        if len(c) == 0: return False

        for v in c:
            if scoring.smoothed_compatible(v, u, G.threshold):
                return False

        return True

    # TODO change to dill, pickle wont pickle
    def save(self, filename: str):
        # Remove redis connections from autodetect as they cannot be pickled
        self.remove_redis_connections()
        with open(f"{filename}.pkl", "wb") as f:
            dill.dump(self, f)

        self.add_redis_connections()

    def remove_redis_connections(self):
        self.cache.redis = None
        self.cache.cmk = None
        self.scoring.cache.redis = None
        self.scoring.cache.cmk = None

        for cache in self.caches.values():
            cache.redis = None
            cache.cmk = None

        for scoring in self.scorings.values():
            scoring.cache.redis = None
            scoring.cache.cmk = None

    def add_redis_connections(self):
        self.cache.redis = Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
        self.cache.cmk = self.cache.redis.cms()
        self.scoring.cache.redis = Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
        self.scoring.cache.cmk = self.scoring.cache.redis.cms()

        for cache in self.caches.values():
            cache.redis = Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
            cache.cmk = cache.redis.cms()

        for scoring in self.scorings.values():
            scoring.cache.redis = Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
            scoring.cache.cmk = scoring.cache.redis.cms()

    @staticmethod
    def create_or_load(filename: str, data_path: str | None, training_set_size: int | None) -> TrainingSet:
        if os.path.isfile(f"{filename}.pkl"):
            with open(f"{filename}.pkl", "rb") as f:
                training_set = dill.load(f)
                training_set.add_redis_connections()
                return training_set

        if not data_path or not training_set_size:
            raise Exception("Incorrect arguments")

        dataframes = generate_df_for_directory(data_path, workers=10)
        training_set = TrainingSet(dataframes)
        training_set.generate_training_set(training_set_size)
        training_set.save(filename)
        return training_set


def generate_dirty_training_set_worker(columns: CleanColumns, size: int, samples_per_iteration: int) -> list[tuple[str, str, Label]]:
    tuples_generated = 0
    result = []

    while tuples_generated < size:
        print(f"Sampling columns for dirty set ... {tuples_generated} of {size}")

        o1 = columns.sample_column()
        C1 = o1.copy().apply(G.convert)
        o2 = columns.sample_column()
        C2 = o2.copy().apply(G.convert)

        if C1.equals(C2):
            continue

        if TrainingSet.is_compatible_worker(C1, C2):
            continue

        v1 = o1.sample(samples_per_iteration, replace=True).to_numpy()
        v2 = o2.sample(samples_per_iteration, replace=True).to_numpy()
        v_all = list(zip(v1, v2, [Label.NEGATIVE for _ in range(samples_per_iteration)]))
        result.extend(v_all)
        tuples_generated += samples_per_iteration

    return result

def generate_language_cache(corpus, language, idx) -> PatternCountCache:
    print("Building cache for language " + str(idx) + " of " + str(len(L)))
    cache = PatternCountCache(language)
    for df in corpus:
        cache.add_data(df)

    cache.redis = None
    cache.cmk = None
    return cache

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield (i, i+n)

def fill_cache_worker(start: int, end: int, languages: list[Language]):
    conn = redis.Redis(password=os.getenv("REDIS_PASSWORD", None))

    for i in range(start, end):
        df_dict = conn.json().get(f"corpus:{i}", "$")[0]
        df = pd.DataFrame.from_dict(df_dict)
        for language in languages:
            PatternCountCache.add_data_for_language(language, df, conn)
        
        if i % 10000:
            print(f"Added {i} of {end} ...")