from flask import Flask, request, render_template
from search_logic import fetch_texts_from_db, prepare_data_for_tfidf, calculate_tfidf_matrix, search_with_cosine_similarity

app = Flask(__name__)

# Inicjalizacja danych
df = fetch_texts_from_db()
df = prepare_data_for_tfidf(df)
vectorizer, tfidf_matrix = calculate_tfidf_matrix(df)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")  # Pobieranie zapytania z formularza
        if query:
            # Wykonaj wyszukiwanie za pomocą Cosine Similarity
            results = search_with_cosine_similarity(query, vectorizer, tfidf_matrix, df).to_dict(orient="records")

    return render_template("index.html", results=results, query=query)


if __name__ == "__main__":
    app.run(debug=True)
