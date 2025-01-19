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
    query = query.lower()  
    df = pd.read_sql_query("SELECT Text FROM transcripts", conn)
    conn.close()

    df['Text'] = df['Text'].str.lower()


    total_occurrences = df['Text'].str.count(rf'\b{query}(?!\w)').sum()
    return total_occurrences


def plot_occurrences_over_episodes(query):

    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Season, Episode, Text FROM transcripts", conn)
    conn.close()

    query_lower = query.lower()
    df['Text'] = df['Text'].str.lower()
    df['occurrences'] = df['Text'].str.count(rf'\b{query_lower}(?!\w)')

    grouped = df.groupby(['Season', 'Episode'], as_index=False)['occurrences'].sum()

    grouped = grouped.sort_values(by=['Season', 'Episode'])

    grouped['EpisodeID'] = range(1, len(grouped) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(grouped['EpisodeID'], grouped['occurrences'], marker='o')
    plt.title(f"Wystąpienia słowa '{query}' w kolejnych odcinkach")
    plt.xlabel("Kolejny odcinek (przez sezony)")
    plt.ylabel("Liczba wystąpień")
    plt.grid(True)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close() 

    return plot_base64

def plot_occurrences_by_season(query):

    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Season, Text FROM transcripts", conn)
    conn.close()

    df["Text"] = df["Text"].str.lower()
    query_lower = query.lower()
    df["occurrences"] = df["Text"].str.count(rf'\b{query_lower}(?!\w)')

    grouped = df.groupby("Season", as_index=False)["occurrences"].sum()

    plt.figure(figsize=(10, 6))
    plt.bar(grouped["Season"], grouped["occurrences"], color="skyblue")
    plt.title(f"Liczba wystąpień słowa '{query}' w poszczególnych sezonach")
    plt.xlabel("Sezon")
    plt.ylabel("Łączna liczba wystąpień")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    bar_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  

    return bar_chart_base64

def plot_sentiment_pie_chart(query):
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Text, Sentiment_Compound FROM transcripts", conn)
    conn.close()

    df["Text"] = df["Text"].str.lower()
    query_lower = query.lower()
    df["has_query"] = df["Text"].str.count(rf'\b{query_lower}(?!\w)') > 0
    
    df_query = df[df["has_query"] == True].copy()
    if df_query.empty:
        return None

    def classify_sentiment(score):
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    df_query["sentiment_label"] = df_query["Sentiment_Compound"].apply(classify_sentiment)

    counts = df_query["sentiment_label"].value_counts()

    color_map = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }
    labels = counts.index.tolist()
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(10, 6))
    plt.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%', 
        startangle=140
    )
    plt.title(f"Sentiment distribution (word='{query}')")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pie_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return pie_chart_base64

def generate_wordcloud_from_db():
    conn = sqlite3.connect("database/sopranos_data.db")
    df = pd.read_sql_query("SELECT Text FROM transcripts", conn)
    conn.close()

    all_text = " ".join(df["Text"].astype(str)).lower()

    nltk.download('stopwords')  
    english_stopwords = set(stopwords.words('english'))
    english_stopwords.update(["i'm", "get", "got", "i'll"])  

    wordcloud = WordCloud(
        width=800, 
        height=600,
        background_color='white',
        max_words=200,  
        stopwords=english_stopwords
    ).generate(all_text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off") 
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    wc_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return wc_base64