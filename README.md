# ANCIlab

Repository per il progetto di tagging automatico delle pagine web dei Comuni lombardi in collaborazione tra ANCIlab e Università degli Studi di Milano.

Gli script Python presenti in questo repository permettono, in ordine, di:
1. Eseguire lo scraping delle pagine web dei Comuni che abbiano fornito la relativa autorizzazione, estraendo per ogni pagina il testo ivi contenuto, i tag di tipo "Categoria" e i tag di tipo "Argomenti" - `scraper.py`
2. Pulire e raggruppare i dati testuali in input ai modelli di classificazione, eliminando il più possibile parti di header, footer e menù interni alle pagine - `cleaning.py`
3. Produrre una classificazione utilizzando un modello in cloud (ChatGPT) da considerare come "valore obiettivo" della classificazione - `chatgpt_baseline.py`
4. Produrre una classificazione con uno dei tre metodi previsti nella sperimentazione - `zeroshot_classification.py`, `feature_extraction.py`, data_augmentation.py`

I risultati della sperimentazione sono disponibili nella presentazione pptx qui presente, utilizzata nell'evento di restituzione svoltosi in modalità online venerdì 7 febbraio 2025.

In collaborazione con (LASER)[https://security.di.unimi.it/]
