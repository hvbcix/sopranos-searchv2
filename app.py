from flask import Flask, request, render_template
from search_logic import fetch_texts_from_db, prepare_data_for_tfidf, calculate_tfidf_matrix, search_with_similarity

app = Flask(__name__)

# Przygotowanie danych
df = fetch_texts_from_db()
df = prepare_data_for_tfidf(df)
vectorizer, tfidf_matrix = calculate_tfidf_matrix(df)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""
    similarity_metric = "cosine"  # Domyślna miara podobieństwa
    top_n = 25  # Domyślna liczba wyników

    if request.method == "POST":
        query = request.form.get("query")  # Pobieranie zapytania z formularza
        similarity_metric = request.form.get("similarity_metric", "cosine")  # Pobieranie wybranej miary
        top_n = int(request.form.get("top_n", 25))  # Pobieranie liczby wyników (domyślnie 25)

        if query:
            # Wykonaj wyszukiwanie za pomocą wybranej miary
            results = search_with_similarity(query, vectorizer, tfidf_matrix, df, similarity_metric, top_n).to_dict(orient="records")

    # Sprawdzenie, czy są jakiekolwiek wyniki
    no_results = len(results) == 0

    return render_template("index.html", results=results, query=query, similarity_metric=similarity_metric, no_results=no_results, top_n=top_n)


if __name__ == "__main__":
    app.run(debug=True)
