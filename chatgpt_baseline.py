import os
import json
from time import time
from openai import OpenAI
import pandas as pd
import spacy
from scipy.stats import hmean
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required = True,
                    help = "File JSON prodotto da cleaning.py")
parser.add_argument("--model", type = str, default = "gpt-4o-mini",
                    help = "Modello GPT scelto.")
parser.add_argument("--label", type = str, required = True,
                    help = "Nome dell'attributo da predire [topic_tags / category_tag].")
args = parser.parse_args()

# Inizializzazione client di OpenAI (API key scaricata dalla piattaforma in formato txt)
with open("openaikey.txt", "r") as f:
    api_key = f.read()
OPENAI_CLIENT = OpenAI(api_key = api_key)

# Funzione per pseudo-anonimizzare i testi
def pseudonymize(doc: spacy.tokens.doc.Doc, placeholder: str = "_____"):

    text = doc.text
    for ent in doc.ents:
        text = text.replace(ent, placeholder)

    return text


if __name__ == "__main__":

    tic = time()
    model_name = args.model

    # Download del modello Spacy utilizzato per riconoscere e pseudo-anonimizzare i testi
    spacy.cli.download("it_core_news_lg")
    nlp = spacy.load("it_core_news_lg")

    # Creazione cartelle dei risultati
    if not os.path.exists("scores"):
        os.makedirs("scores")
    if not os.path.exists("results"):
        os.makedirs("results")

    # Caricamento dati
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

    # Pulizia documenti con Spacy
    docs = [pseudonymize(doc) for doc in nlp.pipe(df['clean_text'].tolist())]

    predictions = []

    for doc in tqdm(docs):

        # Interrogazione all'API di ChatGPT
        if classification_type == "multiclass":
            d = OPENAI_CLIENT.chat.completions.create(
                model = args.model,
                messages = [{
                    "role": "user",
                    "content": f"Dato il seguente documento: {doc}. \nQuale categoria è più adatta a descriverlo tra le seguenti? {', '.join(all_tag)}\nRiporta solo la categoria"
                }]
            )
        else:
            d = OPENAI_CLIENT.chat.completions.create(
                model = args.model,
                messages = [{
                    "role": "user",
                    "content": f"Dato il seguente documento: {doc}. \nQuali categorie sono più adatte a descriverlo tra le seguenti? {', '.join(all_tag)}\nRiporta da un minimo di 1 ad un massimo di 3 categorie, separate da una virgola."
                }]
            )

        predictions.append(d.choices[0].message.content)

    # Trasformazione da stringa a lista di stringhe
    df["predictions"] = list(map(lambda s: s.split(", "), predictions))

    # Calcolo precision e recall
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