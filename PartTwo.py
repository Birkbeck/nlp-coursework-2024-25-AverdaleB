import nltk
import pandas as pd
import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_scoreclassification_report
from sklearn.svm import SVC


if __name__ == "__main__":