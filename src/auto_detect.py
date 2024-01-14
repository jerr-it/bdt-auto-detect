import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import redis

from src.stats.language import L, Language
from src.stats.npmi import st_aggregate, Scoring, PatternCountCache


def st_aggregate_worker(min_precision: float, language: Language) -> Language:
    scoring = Scoring(PatternCountCache(language=language, skip_db_init=True))
    redis_conn = redis.Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD", None))
    scoring.cache.cmk = redis_conn.cms()
    
    try:
        language.threshold, language.h_minus, language.h_plus = st_aggregate(redis_conn, scoring, min_precision)
    except SyntaxError:
        return None

    scoring = None
    return language


class AutoDetect:
    def __init__(self, min_precision: float, memory_budget: int = 10e9):
        self.L_reduced = []
        self.memory_budget = memory_budget

        with ProcessPoolExecutor(max_workers=int(os.getenv("WORKERS", 10))) as executor:
            futures = []
            for language in L:
                futures.append(executor.submit(st_aggregate_worker, min_precision, language))

            for future in as_completed(futures):
                language = future.result()
                if language is not None:
                    self.L_reduced.append(language)

        # for language in L:
        #     try:
        #         scoring = trainings_set.scorings[language]
        #         language.threshold, language.h_minus, language.h_plus = st_aggregate(trainings_set.tuples, scoring,
        #                                                                              min_precision)
        #         self.L_reduced.append(language)
        #     except SyntaxError:
        #         "Language not added to reduced set"
        #         continue

        self.best_languages: set[Language] = set()

    def train(self) -> set[Language]:
        print("Training")
        G_select: set[Language] = set()
        curr_size = 0
        Lc: list[Language] = self.L_reduced.copy()

        while len(Lc) != 0:
            Lc_dash = []

            for language in Lc:
                size = PatternCountCache(language=language, skip_db_init=True).memory_usage
                if size + curr_size > self.memory_budget: continue
                Lc_dash.append(language)

            if len(Lc_dash) == 0:
                break

            unionized = set()
            for L_j in G_select:
                unionized = unionized.union(L_j.h_minus)
            subtrahend = len(unionized)

            best_score = float("-inf")
            best_language = None
            for L_i in Lc_dash:
                unionized = set()
                for L_j in G_select:
                    unionized = unionized.union(L_j.h_minus, L_i.h_minus)

                minuend = len(unionized)
                size = PatternCountCache(language=L_i, skip_db_init=True).memory_usage
                score = (minuend - subtrahend) / size
                #print(score)
                if score > best_score:
                    best_score = score
                    best_language = L_i
                    #print(best_language)

            L_star = best_language

            G_select.add(L_star)
            curr_size += PatternCountCache(language=L_star, skip_db_init=True).memory_usage
            Lc.remove(L_star)

        L_k = None
        H_k_minus = float("-inf")
        for L_i in L:
            if PatternCountCache(language=L_i, skip_db_init=True).memory_usage > self.memory_budget: continue

            if (score := len(L_i.h_minus)) > H_k_minus:
                H_k_minus = score
                L_k = L_i

        unionized = set()
        for L_j in G_select:
            unionized = unionized.union(L_j.h_minus)
        total_score = len(unionized)

        if total_score >= H_k_minus:
            self.best_languages = G_select
            print("G_select")
            return G_select
        else:
            self.best_languages = {L_k}
            return {L_k}

    # Throws KeyError in case of unseen value
    def predict(self, v1, v2) -> (bool, float):
        print(self.best_languages)
        best_score = float("-inf")
        key_errors = 0
        for language in self.best_languages:
            try:
                score = Scoring(PatternCountCache(language=language, skip_db_init=True)).smoothed_npmi(language.convert(v1), language.convert(v2))
                print(score, language.threshold, language.convert(v1), language.convert(v2))
                if score > language.threshold and score > best_score:
                    best_score = score
            except KeyError:
                key_errors += 1
                continue

        if key_errors == len(self.best_languages):
            raise KeyError("No matching language found")

        if best_score > float("-inf"):
            return True, best_score

        return False, 0.0

