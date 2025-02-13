import os
import json
from time import time
from ollama import Client
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, coverage_error, label_ranking_average_precision_score, label_ranking_loss
from scipy.stats import hmean
from bornrule import BornClassifier
from tqdm.auto import tqdm
from typing import List
import argparse

tqdm.pandas()

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

# Funzione per caricare le features da estrarre dal file Excel di AGID
def get_sheet_names(excel_file_path):
    """
    Extracts sheet names from an Excel file.

    Parameters:
    excel_file_path (str): The path to the Excel file.

    Returns:
    list: A list of sheet names in the Excel file.
    """
    try:
        # Load the Excel file
        excel_file = pd.ExcelFile(excel_file_path)
        
        # Get the sheet names
        sheet_names = excel_file.sheet_names
        return sheet_names
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
# Funzione per estrarre le features di interesse (es: nomi di persona, date, luoghi, orari di ricevimento, ...) dal testo
def extract_features(text: str, features: List[str]):

    d = OLLAMA_CLIENT.generate(
        model = args.model,
        stream = False,
        prompt = f"Trova quali categorie nella lista {features} sono presenti nel seguente documento: {text}\nRitorna solo le categorie, separate da una virgola."
    )

    present_features = d['response'].split(", ")

    feature_vector = [int(feat in present_features) for feat in features]

    return feature_vector


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

    classification_type = "multilabel" if args.label.endswith("s") else "multiclass"
    filename = f"{classification_type}-{model_name}-feature-extraction.json"

    all_tag = df[args.label].explode().unique().tolist()
    all_tag = list(filter(lambda s: isinstance(s, str), all_tag))

    excel_file = "Architettura-informazione-sito-Comuni.xlsx"
    sheet_names = get_sheet_names(excel_file)

    features = []
    for name in sheet_names:
        if name.startswith("Tipologia"):
            info = pd.read_excel(excel_file, sheet_name=name, header=2)
            features.extend(info['Elemento'].tolist())

    print("Total number of features:", len(features))
    print("Features:", features)

    # Classificazione supervisionata
    # 1) Divisione dataset in training set e test set
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 2) Estrazione features dai testi
    X_train = df_train['clean_text'].progress_apply(lambda s: extract_features(s, features)).tolist()
    X_test = df_test['clean_text'].progress_apply(lambda s: extract_features(s, features)).tolist()

    # 3) Trasformazione classi in vettori di 0 e 1 (binarizzazione)
    mlb = MultiLabelBinarizer(classes = all_tag)
    y_train = mlb.fit_transform(df_train[args.label].tolist())
    y_test = mlb.transform(df_test[args.label].tolist())

    # 4) Training del modello Born
    born = BornClassifier()
    born.fit(X_train, y_train)

    # 5) Predizione
    y_prob = born.predict_proba(X_test)

    # 6) Calcolo precision e recall
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

