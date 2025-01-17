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
    selected_season = "all"  # Domyślny wybór (wszystkie sezony)
    sort_by = None  # Domyślne sortowanie

    if request.method == "POST":
        query = request.form.get("query")  # Pobieranie zapytania z formularza
        similarity_metric = request.form.get("similarity_metric", "cosine")  # Pobieranie wybranej miary
        top_n = int(request.form.get("top_n", 25))  # Pobieranie liczby wyników (domyślnie 25)
        selected_season = request.form.get("season", "all")  # Pobieranie wybranego sezonu
        sort_by = request.form.get("sort_by", None)  # Pobranie opcji sortowania (np. word_count, character_count)

        if query:
            # Pobieranie danych z bazy z filtrem sezonu
            if selected_season != "all":
                filtered_df = fetch_texts_from_db(season=int(selected_season))
            else:
                filtered_df = fetch_texts_from_db()  # Bez filtra sezonu

            # Przygotowanie danych i wykonanie wyszukiwania
            filtered_df = prepare_data_for_tfidf(filtered_df)
            vectorizer, tfidf_matrix = calculate_tfidf_matrix(filtered_df)
            results = search_with_similarity(
                query, vectorizer, tfidf_matrix, filtered_df, similarity_metric, top_n, sort_by
            ).to_dict(orient="records")

    # Pobieranie unikalnych sezonów do dropdowna
    seasons = sorted(df['Season'].unique())

    # Sprawdzenie, czy są jakiekolwiek wyniki
    no_results = len(results) == 0

    return render_template(
        "index.html",
        results=results,
        query=query,
        similarity_metric=similarity_metric,
        no_results=no_results,
        top_n=top_n,
        seasons=seasons,
        selected_season=selected_season,  # Przekazanie wybranego sezonu
        sort_by=sort_by  # Przekazanie wybranej opcji sortowania
    )

if __name__ == "__main__":
    app.run(debug=True)
