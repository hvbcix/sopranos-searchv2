# statistics_logic.py
import sqlite3
import pandas as pd

# Dodaj te importy, aby rysować i konwertować wykres na base64:
import matplotlib
matplotlib.use('Agg')  # backend 'Agg' pozwala renderować wykresy bez GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
    """
    Zwraca dane wykresu (base64-encoded PNG) pokazującego
    liczbę wystąpień danego słowa w kolejnych odcinkach.
    """
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
