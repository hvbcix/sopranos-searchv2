from flask import Flask, request, render_template
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Połączenie z bazą danych SQLite
DATABASE = "database/sopranos_data.db"

def fetch_texts_from_db():
    """Funkcja pobierająca dane z bazy SQLite."""
    conn = sqlite3.connect(DATABASE)
    query = "SELECT Line_Number, Text, Season, Episode FROM transcripts"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Wczytanie danych z bazy danych SQLite
df = fetch_texts_from_db()

# Wektoryzacja danych (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Text'])  # Tworzenie macierzy TF-IDF

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")  # Pobieranie zapytania z formularza
        if query:
            # Obliczanie wektora TF-IDF dla zapytania
            query_vector = vectorizer.transform([query])

            # Obliczanie podobieństwa cosinusowego
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            # Pobranie najlepszych wyników
            top_indices = similarities.argsort()[-10:][::-1]  # Top 10 wyników
            results = df.iloc[top_indices][['Line_Number', 'Season', 'Episode', 'Text']].to_dict(orient='records')

    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
