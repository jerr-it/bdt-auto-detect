from src.stats.language import L, Language
from src.data_gen.training_data_generation import TrainingSet
from src.stats.npmi import st_aggregate


class AutoDetect:
    def __init__(self, trainings_set: TrainingSet, size: int, min_precision: float, memory_budget: int = 10e9):
        training_data = trainings_set.generate_training_set(size)
        self.trainings_set = trainings_set
        self.memory_budget = memory_budget

        for language in L:
            scoring = trainings_set.scorings[language]
            language.threshold, language.h_minus, language.h_plus = st_aggregate(training_data, scoring, min_precision)

    def train(self) -> set[Language]:
        G_select: set[Language] = set()
        curr_size = 0
        Lc: list[Language] = L.copy()

        while len(Lc) != 0:
            Lc_dash = []

            for language in Lc:
                size = self.trainings_set.caches[language].cmk.memory_usage()
                if size + curr_size > self.memory_budget: continue
                Lc_dash.append(language)

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
                if score > best_score:
                    best_score = score
                    best_language = L_i

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
            return G_select
        else:
            return {L_k}
