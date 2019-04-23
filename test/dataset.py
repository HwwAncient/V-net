import json
import nltk
import pandas as pd
from tqdm import tqdm

from utils import get_fake_answer_json

infile = "data/test.json"
outfile = "data/test_pre.json"

df = pd.read_json(infile, lines=True, encoding="utf-8")
with open(outfile, "w", encoding="utf-8") as f, tqdm(total=(df.size/13)) as pbar:
    for row in df.iterrows():
        pbar.update(1)
        js = {
            "documents": [get_fake_answer_json(p["segmented_paragraphs"][p["most_related_para"]], row[1]["segmented_answers"][0])
                          for p in row[1]["documents"] if p["is_selected"]],
            "query": row[1]["segmented_question"],
            "query_id": row[1]["question_id"],
            "answer": row[1]["segmented_answers"][0],
            "well_formed_answers": row[1]["segmented_answers"][0],
        }
        f.write(json.dumps(js) + "\n")
