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







