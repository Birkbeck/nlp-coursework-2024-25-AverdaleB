
import nltk
import string
import spacy
import pandas as pd 
import pickle
from pathlib import Path


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = nltk.sent_tokenize(text)
    sum_sentences = len(sentences)
    
    tokens = nltk.word_tokenize(text)
    words = [token for token in tokens if token.isalpha()]
    sum_words = len(words)

    sum_syllables = sum(count_syl(word, d) for word in words)
    if len(sentences) > 0 and len(words) > 0:
        # Flesch-Kincaid Grade Level formula =
        # 0.39 * (total words / total sentences) + 11.8 (total syllables / total words) - 15.59
        return (
            0.39 * (sum_words / sum_sentences)
            + 11.8 * (sum_syllables / sum_words)
            - 15.59
        )
    else:
        return 0.0  # to avoid division by zero


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        # count vowel sounds
        return len([phoneme for phoneme in d[word][0] if phoneme[-1].isdigit()])
    else:
        # syllables are estimated by counting vowel clusters
        vowels = "aeiouy"
        word = word.lower()
        count_syl = 0
        prev_was_vowel = False
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    count_syl += 1
                    prev_was_vowel = True
            else:
                prev_was_vowel = False
        return max(count_syl, 1)





def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    #Pandas dataframe to store columns
    df = pd.DataFrame(columns=["text", "title", "author", "year"])
    i = 0
   
    for files in path.glob("*.txt"):
        novels = files.stem.split("-")

        title = "-".join(novels[:-2])
        author = novels[-2]
        year = int(novels[-1])
        text = files.read_text(encoding="utf-8")

        df.loc[i] = [text, title, author, year]
        i += 1
        
    df = df.sort_values("year").reset_index(drop=True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    store_path.mkdir(parents=True, exist_ok=True)
    pickle_path = store_path / out_name

    df_parsed = df.copy()
    parsed_docs = []
    print("Parsing texts using spacy")

    for i, row in df_parsed.iterrows():
        text = row["text"]
        title = row["title"]
        print(f"processing: {title}")
    
        #split text into smaller chunks 
        batch = []
        batch_size = 100000  # You may want to define a batch size
        for start in range(0, len(text), batch_size):
            fragment = text[start:start + batch_size]
            batch.append(nlp(fragment))
        #store list of parsed_docs
        parsed_docs.append(batch)

    df_parsed["parsed"] = parsed_docs

    #serialise parsed DataFrame to a pickle file
    print(f"Parsed dataframe saved to {pickle_path}")
    df_parsed.to_pickle(pickle_path)
    print("Parsing is complete")
    return df_parsed
 

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # type token ratio
    tokens = nltk.word_tokenize(text)
    data_clean = [t.lower() for t in tokens if t.isalpha()]

    if len(data_clean) > 0:
        num_tokens = len(data_clean)
        num_types = len(set(data_clean))
        return num_types / num_tokens
    else:
        return 0.0  #to avoid divsion by zero





def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    
    #NLTK data
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("cmudict")
    
    print(df.head())
    
    df_parsed = parse(df)
    print(df_parsed.head())
    
    #Type Token Ratio.
    ttr_map = get_ttrs(df)
    print("\nType-Token Ratios (TTR):")
    for title, ttr in ttr_map.items():
        print(f"{title}: {ttr:.4f}")

    #Flesch-Kincaid Grade Level Scores.
    fk_map = get_fks(df)
    print("\nFlesch-Kincaid (FK) Reading Grade Scores:")
    for title, fk in fk_map.items():
        print(f"{title}: {fk:.4f}")


    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

