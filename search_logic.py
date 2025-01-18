import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def fetch_texts_from_db(season=None):
    """
    Pobiera teksty z bazy SQLite z opcjonalnym filtrem sezonu.
    """
    conn = sqlite3.connect("database/sopranos_data.db")
    query = "SELECT * FROM transcripts"
    
    # Jeśli podano sezon, dodaj warunek WHERE
    if season is not None:
        query += f" WHERE Season = {season}"

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
    Przetwarza tekst, usuwając interpunkcję, konwertując na małe litery i ignorując stopwords.
    """
    # Usuwanie interpunkcji
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Konwersja na małe litery
    text = text.lower()
    # Usuwanie stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


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


def jaccard_similarity(query, document):
    """
    Oblicza miarę Jaccarda pomiędzy zapytaniem a dokumentem.
    """
    query_set = set(query.split())
    document_set = set(document.split())
    intersection = query_set.intersection(document_set)
    union = query_set.union(document_set)
    return len(intersection) / len(union) if len(union) > 0 else 0


def search_with_similarity(query, vectorizer, tfidf_matrix, df, similarity_metric="cosine", top_n=10, sort_by=None, filter_type=None, contains_profanity=None, contains_italian=None, contains_name=None, contains_food=None):
    """
    Wyszukuje teksty najbardziej podobne do zapytania na podstawie wybranej miary podobieństwa
    oraz stosuje dodatkowe filtry.
    """
    query = preprocess_text(query)

    if similarity_metric == "cosine":
        # Obliczanie wektora TF-IDF dla zapytania
        query_vector = vectorizer.transform([query])

        # Obliczanie podobieństwa cosinusowego
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    elif similarity_metric == "jaccard":
        # Obliczanie miary Jaccarda dla każdego tekstu
        similarities = df['Processed_Text'].apply(lambda doc: jaccard_similarity(query, doc)).values

    # Pobranie indeksów wyników z podobieństwem > 0.0
    valid_indices = [i for i, sim in enumerate(similarities) if sim > 0.0]

    if not valid_indices:
        # Jeśli nie ma wyników z podobieństwem > 0.0, zwracamy pusty DataFrame
        return pd.DataFrame(columns=['Line_Number', 'Season', 'Episode', 'Text', 'Similarity'])

    # Pobranie wyników z DataFrame
    sorted_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_n]
    results = df.iloc[sorted_indices][['Line_Number', 'Season', 'Episode', 'Text', 'Is_Question', 'Is_Exclamation', "Word_Count", "Character_Count", "Contains_Profanity", "Contains_Italian", "Contains_Name", "Contains_Food"]].copy()
    results['Similarity'] = [similarities[i] for i in sorted_indices]

    # Zastosowanie filtra Is_Question lub Is_Exclamation
    if filter_type == "question":
        results = results[results['Is_Question'] == True]
    elif filter_type == "exclamation":
        results = results[results['Is_Exclamation'] == True]

        # Zastosowanie filtra Is_Question lub Is_Exclamation
    if contains_profanity == "only_profanity":
        results = results[results['Contains_Profanity'] == True]
    elif contains_profanity == "no_profanity":
        results = results[results['Contains_Profanity'] == False]

             # Zastosowanie filtra Is_Question lub Is_Exclamation
    if contains_italian == "only_italian":
        results = results[results['Contains_Italian'] == True]
    elif contains_italian == "no_italian":
        results = results[results['Contains_Italian'] == False]

                 # Zastosowanie filtra Is_Question lub Is_Exclamation
    if contains_name == "only_name":
        results = results[results['Contains_Name'] == True]
    elif contains_name == "no_name":
        results = results[results['Contains_Name'] == False]

        # Zastosowanie filtra Is_Question lub Is_Exclamation
    if contains_food == "only_food":
        results = results[results['Contains_Food'] == True]
    elif contains_food == "no_food":
        results = results[results['Contains_Food'] == False]                 

    # Dodanie sortowania wyników według `sort_by`
    if sort_by == "word_count":
        results['Word_Count'] = results['Text'].apply(lambda x: len(x.split()))
        results = results.sort_values(by='Word_Count', ascending=False)
    elif sort_by == "character_count":
        results['Character_Count'] = results['Text'].apply(len)
        results = results.sort_values(by='Character_Count', ascending=False)

    return results