import nltk
import pandas as pd
import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC


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
    X = vectoriser.fit_transform(df["speech"])
    y = df["party"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=26,
        stratify=y
    )

    # Train Random Forest 
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=26    
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # RF Performance evaluation
    print("RandomForest (n=300) macro-F1:"
          , f1_score(y_test, y_pred, average="macro"))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred))
    
    # Train SVM with linear kernel classifier
    svm_model = LinearSVC(
        random_state=26,
    )
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    # Performance evaluation SVM
    print("SVM (linear) macro-F1:",
          f1_score(y_test, y_pred_svm, average="macro"))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred_svm))
    

          

    




    










