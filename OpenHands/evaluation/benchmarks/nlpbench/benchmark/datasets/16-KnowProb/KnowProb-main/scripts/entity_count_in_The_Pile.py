
import pandas as pd
import requests, time
from tqdm import tqdm
from pathlib import Path
import pickle
PERMANENT_PATH = "."
import requests, time

pararel = pd.read_csv(Path(PERMANENT_PATH) / "paper_data/pararel.csv")

subjects = pararel.subject.tolist()

payload = {
    'corpus': 'v4_piletrain_llama',
    'query_type': 'count',
    'query': 'query goes here...',
}

entity_counts = {}
for entity in tqdm(subjects):
    if entity not in entity_counts:
        payload['query'] = entity
        result = requests.post('https://api.infini-gram.io/', json=payload).json()
        time.sleep(0.01)
        entity_counts[entity] = result['count']
with open(Path(PERMANENT_PATH) / "paper_data/entity_counts.pkl", "wb") as fo:
    pickle.dump(entity_counts, fo, protocol=pickle.HIGHEST_PROTOCOL)