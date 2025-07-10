import nltk
import pandas as pd
import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC


if __name__ == "__main__":
    csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    print(csv_path)
    df = pd.read_csv(csv_path)

    # renaming the ‘Labour (Co-op)’ value
    df["party"] = df["party"].replace({"Labour (Co-op)": "Labour"})
    
    parties = df["party"].value_counts()
    top_4_parties = parties.nlargest(4).index.tolist()
    df = df[df["party"].isin(top_4_parties)]

    if "Speaker" in df.columns:
        df = df.drop(columns=["Speaker"])

    df = df[df['speech_class'] == 'Speech']
    df = df[df["speech"].str.len() >= 1000]
    print(df.shape)

    vectoriser = TfidfVectorizer(
        stop_words="english",
        max_features=3000,       
    )
    X_train = vectoriser.fit_transform(df["speech"])
    y_train = df["party"]

    X_train, X_test, y_train, y_test, feature_names, target_names = train_test(
        X_train, y_train, 
        test_size=0.2,
        random_state=26,
        stratify=y_train
    )
    










