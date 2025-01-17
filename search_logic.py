import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string


def fetch_texts_from_db(database_path="database/sopranos_data.db"):
    """
    Pobiera dane z bazy SQLite i zwraca DataFrame.
    """
    conn = sqlite3.connect(database_path)
    query = "SELECT Line_Number, Text, Season, Episode FROM transcripts"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_data_for_tfidf(df):
    """
    Przygotowuje dane do obliczeń TF-IDF: usuwa brakujące wartości i konwertuje do string.
    """
    df = df.dropna(subset=['Text'])  # Usuń puste wartości
    df['Text'] = df['Text'].astype(str)  # Upewnij się, że wszystkie dane w 'Text' są typu string
    return df


def preprocess_text(text):
    """
    Funkcja do wstępnego przetwarzania tekstu: usuwa interpunkcję, ignoruje wielkość znaków.
    """
    return text.translate(str.maketrans("", "", string.punctuation)).lower()


def calculate_tfidf_matrix(df):
    """
    Oblicza macierz TF-IDF dla kolumny 'Text'.
    """
    # Przetwarzamy tylko dla celów obliczeń, nie zmieniamy oryginalnej kolumny
    df['Processed_Text'] = df['Text'].apply(preprocess_text)

    # Obliczanie macierzy TF-IDF na przetworzonym tekście
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
    return vectorizer, tfidf_matrix


def search_with_cosine_similarity(query, vectorizer, tfidf_matrix, df, top_n=10):
    """
    Wyszukuje teksty najbardziej podobne do zapytania na podstawie Cosine Similarity.
    """
    # Wstępne przetwarzanie zapytania
    query = preprocess_text(query)

    # Obliczanie wektora TF-IDF dla zapytania
    query_vector = vectorizer.transform([query])

    # Obliczanie podobieństwa cosinusowego
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Pobranie indeksów Top N wyników
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Pobranie wyników z DataFrame
    results = df.iloc[top_indices][['Line_Number', 'Season', 'Episode', 'Text']].copy()
    results['Similarity'] = similarities[top_indices]
    return results
