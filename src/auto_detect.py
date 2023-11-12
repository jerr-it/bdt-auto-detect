from src.stats.language import L, Language
from src.data_gen.training_data_generation import TrainingSet
from src.stats.npmi import st_aggregate

from src.utils.label import Label

class AutoDetect:
    def __init__(self, trainings_set: TrainingSet, min_precision: float, memory_budget: int = 10e9):
        self.L_reduced = L.copy()
        self.trainings_set = trainings_set
        self.memory_budget = memory_budget

        for language in L:
            try:
                scoring = trainings_set.scorings[language]
                language.threshold, language.h_minus, language.h_plus = st_aggregate(trainings_set.tuples, scoring,
                                                                                     min_precision)
                self.L_reduced.append(language)
            except SyntaxError:
                continue

        self.best_languages: set[Language] = set()

    def train(self) -> set[Language]:
        G_select: set[Language] = set()
        curr_size = 0
        Lc: list[Language] = self.L_reduced.copy()

        while len(Lc) != 0:
            Lc_dash = []

            for language in Lc:
                size = self.trainings_set.caches[language].cmk.memory_usage()
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
                size = self.trainings_set.caches[L_i].cmk.memory_usage()
                score = (minuend - subtrahend) / size
                #print(score)
                if score > best_score:
                    best_score = score
                    best_language = L_i
                    #print(best_language)

            L_star = best_language

            G_select.add(L_star)
            curr_size += self.trainings_set.caches[L_star].cmk.memory_usage()
            Lc.remove(L_star)

        L_k = None
        H_k_minus = float("-inf")
        for L_i in L:
            if self.trainings_set.caches[L_i].cmk.memory_usage() > self.memory_budget: continue

            if (score := len(L_i.h_minus)) > H_k_minus:
                H_k_minus = score
                L_k = L_i

        unionized = set()
        for L_j in G_select:
            unionized = unionized.union(L_j.h_minus)
        total_score = len(unionized)

        if total_score >= H_k_minus:
            self.best_languages = G_select
            return G_select
        else:
            self.best_languages = {L_k}
            return {L_k}

    # Throws KeyError in case of unseen value
    def predict_nonsense(self, v1, v2) -> (bool, float):
        print(self.best_languages)
        best_score = float("-inf")
        key_errors = 0
        for language in self.best_languages:
            try:
                score = self.trainings_set.scorings[language].smoothed_npmi(language.convert(v1), language.convert(v2))
                print(score, language.convert(v1), language.convert(v2))
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
