import os
import json
from ollama import Client
import pandas as pd
from tqdm.auto import tqdm
from time import time
from scipy.stats import hmean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, coverage_error, label_ranking_average_precision_score, label_ranking_loss
from bornrule import BornClassifier
from random import choice
from typing import Iterable
from nltk.corpus import stopwords
import nltk
import argparse

nltk.download("stopwords")

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required = True,
                    help = "JSON file produced by the scraper.")
parser.add_argument("--model", type = str, default = "mistral",
                    help = "Ollama model to use.")
parser.add_argument("--label", type = str, required = True,
                    help = "The name of the label column.")
args = parser.parse_args()

# Inizializzazione client di Ollama
with open("ollama_host.txt", "r") as f:
    ip_address = f.read()
OLLAMA_CLIENT = Client(host = ip_address)
THRESHOLD = 0.25


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

    # La classificazione è multilabel nel caso dei topic_tags, multiclass nel caso dei category_tag
    classification_type = "multilabel" if args.label.endswith("s") else "multiclass"
    filename = f"{classification_type}-{model_name}-data-augmentation.json"

    # Lista di tutti i tag possibili per l'attributo scelto
    all_tag = df[args.label].explode().unique().tolist()
    all_tag = list(filter(lambda s: isinstance(s, str), all_tag))

    new_data = []

    # Generazione di nuovi testi fittizi per ogni tag con LLM
    for _, row in tqdm(df[df[args.label].apply(len) > 0].iterrows()):

        d = OLLAMA_CLIENT.generate(
            model = args.model,
            stream = False,
            prompt = f"Data la seguente pagina a tema {' e '.join(row[args.label])}, genera una nuova pagina con lo stesso tema. Pagina: {row['clean_text']}"
        )

        new_data.append(
            {
                "name": "generated",
                "clean_text": d['response'],
                args.label: row[args.label]
            }
        )

    df = pd.concat([df, pd.DataFrame.from_records(new_data)], axis = 0)
    df = df[df[args.label].apply(len) > 0]
    print("Remaining pages:", df.shape[0])

    sw = stopwords.words("italian")

    # Classificazione supervisionata
    # 1) Calcolo del TFIDF dei testi risultanti e binarizzazione dei tag obiettivo
    tfidf = TfidfVectorizer(lowercase=False, stop_words=sw, max_features = 1000)
    mlb = MultiLabelBinarizer(classes=all_tag)

    joint_tags = df[args.label].apply(lambda L: "-".join(L) if isinstance(L, Iterable) else L)
    df_train, df_test = train_test_split(df, test_size = len(joint_tags.unique()), random_state = 42, stratify = joint_tags.tolist())

    X_train = tfidf.fit_transform(df_train['clean_text'].tolist())
    y_train = mlb.fit_transform(df_train[args.label].tolist())

    X_test = tfidf.transform(df_test['clean_text'].tolist())
    y_test = mlb.transform(df_test[args.label].tolist())

    # 2) Training del modello Born
    born = BornClassifier()
    born.fit(X_train, y_train)
    
    # 3) Predizione
    y_prob = born.predict_proba(X_test)

    # 4) Calcolo precision e recall
    if classification_type == "multiclass":
        y_pred = [mlb.classes_[idx] for idx in y_prob.argmax(axis=1)]
        y_true = df_test[args.label].apply(lambda L: L[0]).tolist()
        scores = classification_report(y_true, y_pred, output_dict=True)
        df_test['predictions'] = y_pred
    else:
        ce = coverage_error(y_test, y_prob)
        lrap = label_ranking_average_precision_score(y_test, y_prob)
        lrl = label_ranking_loss(y_test, y_prob)
        scores = {
            "coverage error": ce,
            "label ranking average precision": lrap,
            "label ranking loss": lrl,
            "support": len(y_test)
        }
        df_test['predictions'] = ["; ".join([c for c,x in zip(mlb.classes_, v) if x >= THRESHOLD]) for v in y_prob]
        df_test['predictions'] = df_test['predictions'].apply(lambda s: s.split("; "))
        
        df_test['intersections'] = df_test.apply(lambda row: len(set(row[args.label]).intersection(set(row['predictions']))), axis = 1)
        scores['micro-precision'] = df_test.apply(lambda row: row['intersections'] / len(row['predictions']), axis = 1).mean()
        scores['micro-recall'] = df_test.apply(lambda row: row['intersections'] / len(row[args.label]) if len(row[args.label]) else 0, axis = 1).mean()
        scores['micro-f1'] = hmean([scores['micro-precision'], scores['micro-recall']])
        scores['macro-precision'] = df_test['intersections'].sum() / df_test['predictions'].apply(len).sum()
        scores['macro-recall'] = df_test['intersections'].sum() / df_test[args.label].apply(len).sum()
        scores['macro-f1'] = hmean([scores['macro-precision'], scores['macro-recall']])
    
    scores['time'] = time() - tic
    records = df_test[['name', args.label, 'predictions']].to_json("results/" + filename, orient = "records")

    with open("scores/" + filename, "w") as f:
        json.dump(scores, f)

