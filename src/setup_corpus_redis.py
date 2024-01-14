import os
import redis
from redis.commands.json.path import Path
import dill

conn = redis.Redis(host="localhost", port=6379, db=0, password=os.getenv("REDIS_PASSWORD",))

with open("df_list.pkl", "rb") as f:
    corpus = dill.load(f)

    conn.set("corpus_length", len(corpus))
    for idx, df in enumerate(corpus):
        #print(f"Index: {idx}, df: {df}")
        for col in df.columns:
            df[col] = df[col].astype(str)
        print(df.to_dict())
        conn.json().set(f"corpus:{idx}", "$", df.to_dict())
    
