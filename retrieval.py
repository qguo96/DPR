import json
import argparse
import tqdm
from pyserini.search import SimpleSearcher

parser = argparse.ArgumentParser()
parser.add_argument("--qa", type=str, required=True)
parser.add_argument("--index", type=str, default="index")
parser.add_argument("--ndoc", type=int, default=1000)
parser.add_argument("--output", type=str, default="retrieval_result.json")
args = parser.parse_args()

qas = open(args.qa, "r").readlines()
questions = []
for qa in qas:
    question = json.loads(qa)["question"]
    questions.append({"question": question})

print("load index")
searcher = SimpleSearcher(args.index)

results = []
for qa in tqdm.tqdm(questions):
    question = qa["question"]
    hits = searcher.search(question, k=1000)
    ctxs = []
    for hit in hits:
        doc = hit.raw
        title = doc.split("\n")[0].strip()
        text = doc[len(title):].strip()
        id = hit.docid
        score = hit.score
        ctxs.append({"id": id, "title": title, "text": text, "score": score})
    results.append({"question": question, "answers": [""], "ctxs": ctxs})

json.dump(results, open(args.output, "w"), indent=4)
