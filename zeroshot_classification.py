import os
import json
from time import time
from ollama import Client
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required = True,
                    help = "JSON file produced by the scraper.")
parser.add_argument("--model", type = str, default = "mistral",
                    help = "Ollama model to use.")
parser.add_argument("--label", type = str, required = True,
                    help = "The name of the label column [topic_tags / category_tag].")
args = parser.parse_args()

# Inizializzazione client di Ollama
with open("ollama_host.txt", "r") as f:
    ip_address = f.read()
OLLAMA_CLIENT = Client(host = ip_address)

if __name__ == "__main__":

    tic = time()

    model_name = args.model.replace(":", ".").split(".")[0]

    if not os.path.exists("scores"):
        os.makedirs("scores")

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("./clean_data/" + args.file, "r") as f:
        pages = json.load(f)

    df = pd.DataFrame.from_records(pages)
    df = df[df[args.label].apply(len) > 0]
    print("Remaining pages:", df.shape[0])

    # La classificazione è multilabel nel caso dei topic_tags, multiclass nel caso dei category_tag
    classification_type = "multilabel" if args.label.endswith("s") else "multiclass"
    filename = f"{classification_type}-{model_name}-zero-shot.json"

    # Lista di tutti i tag possibili per l'attributo scelto
    all_tag = df[args.label].explode().unique().tolist()
    all_tag = list(filter(lambda s: isinstance(s, str), all_tag))
    print(all_tag)

    predictions = []

    # Classificazione zero-shot con LLM
    for doc in tqdm(df['clean_text']):

        if classification_type == "multiclass":
            d = OLLAMA_CLIENT.generate(
                model = args.model,
                stream = False,
                prompt = f"Dato il seguente documento: {doc}. \nQuale categoria è più adatta a descriverlo tra le seguenti? {', '.join(all_tag)}\nRiporta solo la categoria"
            )
        else:
            d = OLLAMA_CLIENT.generate(
                model = args.model,
                stream = False,
                prompt = f"Dato il seguente documento: {doc}. \nQuali categorie sono adatte a descriverlo tra le seguenti? {', '.join(all_tag)}\nRiporta solo le categorie, da un minimo di 1 a un massimo di 3, separate da una virgola"
            )

        predictions.append(d['response'])

    df["predictions"] = list(map(lambda s: s.split(", "), predictions))

    # Calcolo di precision e recall
    if classification_type == "multiclass":
        scores = classification_report(
            df[args.label].apply(lambda L: L[0]).tolist(), 
            df['predictions'].apply(lambda L: L[0]).tolist(),
            output_dict=True
        )
    else:
        df['intersections'] = df.apply(lambda row: len(set(row[args.label]).intersection(set(row['predictions']))), axis = 1)
        scores = {
            "micro-precision": df.apply(lambda row: row['intersections'] / len(row['predictions']), axis = 1).mean(),
            "micro-recall": df.apply(lambda row: row['intersections'] / len(row[args.label]) if len(row[args.label]) else 0, axis = 1).mean(),
            "macro-precision": df['intersections'].sum() / df['predictions'].apply(len).sum(),
            "macro-recall": df['intersections'].sum() / df[args.label].apply(len).sum(),
            "support": df.shape[0]
        }
        scores['micro-f1'] = hmean([scores['micro-precision'], scores['micro-recall']])
        scores['macro-f1'] = hmean([scores['macro-precision'], scores['macro-recall']])

    scores['time'] = time() - tic
    records = df[['name', args.label, 'predictions']].to_json("results/"+filename, orient = "records")

    with open("scores/" + filename, "w") as f:
        json.dump(scores, f)