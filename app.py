from flask import Flask, request, render_template
from search_logic import fetch_texts_from_db, prepare_data_for_tfidf, calculate_tfidf_matrix, search_with_similarity
from statistics_logic import calculate_total_occurrences, plot_occurrences_over_episodes, plot_occurrences_by_season, plot_sentiment_pie_chart, generate_wordcloud_from_db

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
    filter_type = None  # Domyślny brak filtra typu
    sort_by = None
    contains_profanity = None
    contains_italian = None
    contains_name = None
    contains_food = None

    if request.method == "POST":
        query = request.form.get("query")  # Pobieranie zapytania z formularza
        similarity_metric = request.form.get("similarity_metric", "cosine")  # Pobieranie wybranej miary
        top_n = int(request.form.get("top_n", 25))  # Pobieranie liczby wyników (domyślnie 25)
        selected_season = request.form.get("season", "all")  # Pobieranie wybranego sezonu
        filter_type = request.form.get("filter_type", None)  # Pobieranie filtra typu
        sort_by = request.form.get("sort_by", None)  # Pobieranie filtra typu
        contains_profanity = request.form.get("contains_profanity", None)
        contains_italian = request.form.get("contains_italian", None)
        contains_name = request.form.get("contains_name", None)
        contains_food = request.form.get("contains_food", None)

        if "show_statistics" in request.form:  # If 'Show Statistics' button is clicked
            return render_template("statistics.html", query=query)

        if query:
            results = []  # Replace with actual search logic    

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
                query, vectorizer, tfidf_matrix, filtered_df, similarity_metric, top_n, filter_type=filter_type, sort_by=sort_by, contains_profanity=contains_profanity, contains_italian=contains_italian, contains_name=contains_name, contains_food=contains_food
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
        filter_type=filter_type,  # Przekazanie filtra typu
        sort_by=sort_by,
        contains_profanity=contains_profanity,
        contains_italian=contains_italian,
        contains_name=contains_name,
        contains_food=contains_food,
    )




@app.route("/statistics", methods=["GET"])
def statistics():
    query = request.args.get("query", "").strip()
    if not query:
        return render_template(
            "statistics.html",
            query=query,
            total_occurrences=None, 
            chart_url=None,
            bar_chart_url=None,
            sentiment_pie_chart=None)

    # 1. Obliczamy całkowitą liczbę wystąpień
    total_occurrences = calculate_total_occurrences(query)

    # 2. Generujemy wykres (base64)
    chart_url = plot_occurrences_over_episodes(query)

    bar_chart_url = plot_occurrences_by_season(query)

    sentiment_pie_url = plot_sentiment_pie_chart(query)

    wc_url = generate_wordcloud_from_db()


    # 3. Renderujemy template
    return render_template("statistics.html", 
                           query=query, 
                           total_occurrences=total_occurrences,
                           chart_url=chart_url,
                           bar_chart_url=bar_chart_url,
                           sentiment_pie_url=sentiment_pie_url,
                           wordcloud_url=wc_url)


if __name__ == "__main__":
    app.run(debug=True)
