import os

from src.data_gen.auto_gen_tests import generate_df_for_directory
from src.auto_detect import AutoDetect
from src.data_gen.training_data_generation import TrainingSet

path = os.path.join(os.path.curdir, "248", "WDC", "scsv", "248")
dataframes = generate_df_for_directory(path, workers=10)
training_set = TrainingSet(dataframes)
training_set.generate_training_set(size=10000)
autodetect = AutoDetect(training_set, min_precision=0.75, memory_budget=1e9)
autodetect.train()
for l in autodetect.best_languages:
    print(l, "\n------------\n")
