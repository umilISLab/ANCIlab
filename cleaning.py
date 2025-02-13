import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import os


if __name__ == "__main__":

    # I file vengono letti dalla cartella "raw_data" e alla fine salvati come un unico file JSON nella cartella "clean_data"
    if not os.path.exists("./clean_data"):
        os.makedirs("./clean_data")

    records = []
    for file in os.listdir("./raw_data"):
        print(file[:-5])
        data = pd.read_json(os.path.join("./raw_data",file), orient = "columns")
        records.append(data)

    df = pd.concat(records, axis = 0).reset_index()
    df.rename({"index":"name"}, axis = 1, inplace=True)
    print(df.head(2))

    df = df[df['webpage'].apply(lambda s: isinstance(s, str))]
    df['category_tag'] = df['category_tag'].apply(lambda s: s.strip().strip(";").split(";") if s else [])
    df['topic_tags'] = df['topic_tags'].apply(lambda s: s.strip().strip(";").split(";") if s else [])

    # Metodo euristico: eliminare le parole, bigram e trigram isolati (cioè non in una frase) che appaiono in più del 95% delle pagine
    counter = CountVectorizer(ngram_range=(2,3), max_features= 10000, lowercase = False)
    X = counter.fit_transform(df['webpage']).toarray()
    X[X > 1] = 1
    X = pd.DataFrame(X, columns = counter.get_feature_names_out().tolist())

    ngram_df = X.sum() / X.shape[0]
    blacklist = ngram_df[ngram_df > 0.95].index.tolist()
    blacklist = sorted(blacklist, key = lambda s: len(s.split()), reverse=True)

    subst = "|".join(blacklist)
    df['clean_text'] = df['webpage'].apply(lambda s: re.sub(rf"({subst})", "", s)) \
                                    .apply(lambda s: re.sub(r"\n{2,}", "\n", s)) \
                                    .apply(lambda s: re.sub(r" {2,}", " ", s)) \
                                    .apply(lambda s: re.sub(r"( \w\b){2,}", "", s))
    
    if "name" not in df.columns:
        df['name'] = ["Test"]*df.shape[0]

    df = df[['name','clean_text','category_tag', 'topic_tags']]
        
    df.to_json(f"clean_data/all_data.json", orient="records")