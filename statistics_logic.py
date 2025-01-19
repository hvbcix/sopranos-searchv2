import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend 'Agg' pozwala renderować wykresy bez GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

def calculate_total_occurrences(query):
    """
    Calculate the total number of occurrences of a query word in the database.
    """
    conn = sqlite3.connect("database/sopranos_data.db")
    query = query.lower()  # Ensure the query is case-insensitive
    df = pd.read_sql_query("SELECT Text FROM transcripts", conn)
    conn.close()

    # Ensure all text is lowercase for case-insensitive matching
    df['Text'] = df['Text'].str.lower()

    # Count occurrences of the query in the Text column
    total_occurrences = df['Text'].str.count(rf'\b{query}(?!\w)').sum()
    return total_occurrences


def plot_occurrences_over_episodes(query):

    # 1. Pobranie danych z bazy
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Season, Episode, Text FROM transcripts", conn)
    conn.close()

    # 2. Przetworzenie danych, liczenie wystąpień
    query_lower = query.lower()
    df['Text'] = df['Text'].str.lower()
    df['occurrences'] = df['Text'].str.count(rf'\b{query_lower}(?!\w)')

    # 3. Grupowanie do poziomu (Season, Episode)
    grouped = df.groupby(['Season', 'Episode'], as_index=False)['occurrences'].sum()

    # 4. Posortuj, by zachować kolejność S01E01, S01E02, S02E01, itd.
    grouped = grouped.sort_values(by=['Season', 'Episode'])

    # Opcjonalnie stworzymy kolumnę łączącą Season+Episode w formie np. "S1E1", "S1E2", ...
    # lub numer ciągły (EpisodeID).
    grouped['EpisodeID'] = range(1, len(grouped) + 1)

    # 5. Rysowanie wykresu w Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['EpisodeID'], grouped['occurrences'], marker='o')
    plt.title(f"Wystąpienia słowa '{query}' w kolejnych odcinkach")
    plt.xlabel("Kolejny odcinek (przez sezony)")
    plt.ylabel("Liczba wystąpień")
    plt.grid(True)

    # 6. Konwersja wykresu do base64, by można go było osadzić w HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()  # Ważne, by zwolnić zasoby plotu w Matplotlib

    return plot_base64

def plot_occurrences_by_season(query):
    """
    Zwraca dane wykresu (base64) przedstawiającego łączną liczbę wystąpień
    danego słowa w poszczególnych sezonach.
    """
    # 1. Pobierz dane z bazy
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Season, Text FROM transcripts", conn)
    conn.close()

    # 2. Policz wystąpienia słowa w każdej linii i grupuj po sezonie
    df["Text"] = df["Text"].str.lower()
    query_lower = query.lower()
    df["occurrences"] = df["Text"].str.count(rf'\b{query_lower}(?!\w)')

    # Grupowanie po sezonie
    grouped = df.groupby("Season", as_index=False)["occurrences"].sum()

    # 3. Rysowanie wykresu słupkowego
    plt.figure(figsize=(10, 6))
    plt.bar(grouped["Season"], grouped["occurrences"], color="skyblue")
    plt.title(f"Liczba wystąpień słowa '{query}' w poszczególnych sezonach")
    plt.xlabel("Sezon")
    plt.ylabel("Łączna liczba wystąpień")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Konwersja wykresu do postaci base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    bar_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  # zamykamy rysunek, by zwolnić zasoby

    return bar_chart_base64

def plot_sentiment_pie_chart(query):
    """
    Tworzy wykres kołowy pokazujący udział dokumentów o sentymencie
    pozytywnym (Sentiment_Compound > 0), negatywnym (< 0) i neutralnym (= 0)
    wśród dokumentów, które zawierają dane słowo (query).
    """
    # 1. Pobranie danych z bazy
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Text, Sentiment_Compound FROM transcripts", conn)
    conn.close()

    # 2. Filtrowanie tylko tych linii, w których występuje słowo (case-insensitive)
    df["Text"] = df["Text"].str.lower()
    query_lower = query.lower()
    df["has_query"] = df["Text"].str.count(rf'\b{query_lower}(?!\w)') > 0
    
    df_query = df[df["has_query"] == True].copy()
    if df_query.empty:
        # Jeśli żadna linia nie zawiera słowa, można zwrócić None albo "pusty" wykres
        return None

    # 3. Podział na kategorie sentymentu (powyżej 0 = pozytywny, poniżej 0 = negatywny, równo 0 = neutralny)
    def classify_sentiment(score):
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    df_query["sentiment_label"] = df_query["Sentiment_Compound"].apply(classify_sentiment)

    # 4. Zliczamy, ile mamy linii w każdej kategorii
    counts = df_query["sentiment_label"].value_counts()

    # 5. Konfiguracja kolorów (według życzenia)
    color_map = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }
    # Tworzymy listę kolorów w kolejności odpowiadającej labels z value_counts()
    labels = counts.index.tolist()
    colors = [color_map[label] for label in labels]

    # 6. Rysowanie wykresu kołowego
    plt.figure(figsize=(10, 6))
    plt.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',  # wyświetla procenty z 1 miejscem po przecinku
        startangle=140
    )
    plt.title(f"Sentiment distribution (word='{query}')")

    # 7. Konwersja do base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pie_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return pie_chart_base64

def generate_wordcloud_from_db():
    """
    Pobiera cały tekst z bazy danych i generuje chmurę słów (WordCloud).
    Zwraca obrazek w formacie base64, aby można go było wyświetlić w HTML.
    """

    # 1. Połączenie z bazą i pobranie wszystkich linii tekstu
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Text FROM transcripts", conn)
    conn.close()

    # 2. Łączymy wszystko w jeden duży string
    #    i konwertujemy na lowercase dla spójności
    all_text = " ".join(df["Text"].astype(str)).lower()

    nltk.download('stopwords')  # upewniamy się, że mamy pobrane
    english_stopwords = set(stopwords.words('english'))
    english_stopwords.update(["i'm", "get", "got", "i'll"])  # w razie potrzeby


    # 3. Generowanie chmury słów
    #    Możesz przekazać dodatkowe parametry np. stopwords, szerokość, wysokość...
    wordcloud = WordCloud(
        width=800, 
        height=600,
        background_color='white',
        max_words=200,  # maksymalna liczba słów w chmurze
        stopwords=english_stopwords
    ).generate(all_text)

    # 4. Konwersja do base64
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # wyłączenie osi
    # Zapisujemy do bufora
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    wc_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return wc_base64