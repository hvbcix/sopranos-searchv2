import pandas as pd
import string
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Wczytaj listę przekleństw z pliku
with open("data/badwords.txt", "r", encoding="utf-8") as f:
    profanities = set(f.read().splitlines()) 

# Wczytaj listę włoskich słów z pliku
with open("data/ita.txt", "r", encoding="utf-8") as f:
    italian_words = set(f.read().splitlines()) 

with open("data/food.txt", "r", encoding="utf-8") as f:
    food_words = set(f.read().splitlines())

# Wczytanie listy imion z pliku names.txt
with open("data/names.txt", "r", encoding="utf-8") as f:
    names = set(name.strip().lower() for name in f.readlines())  # Ignorujemy wielkość liter

#Pobieranie lexicona do analizy sentymentu
nltk.download('vader_lexicon')

# Wczytaj dane z CSV
input_file = "data/sopranos_transcripts_full.csv"
output_file = "data/sopranos_transcripts_enriched.csv"
df = pd.read_csv(input_file)

sia = SentimentIntensityAnalyzer()

# Przypisanie nowego Line_Number (od 1 do n)
df["Line_Number"] = range(1, len(df) + 1)

# Funkcje pomocnicze
def word_count(text):
    return len(text.split())

def contains_italian_word(text):
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    words = text_cleaned.split()  # Podziel tekst na słowa
    for word in words:
        if word.lower() in italian_words:
            return True
    return False

def contains_profanity(text):
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    return any(word.lower() in profanities for word in text_cleaned.split())

def contains_name(text):
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    return any(word.lower() in names for word in text_cleaned.split())

def contains_food_reference(text):
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    return any(word.lower() in food_words for word in text_cleaned.split())

def character_count(text):
    return len(text)

def is_question(text):
    return text.strip().endswith("?")

def is_exclamation(text):
    return text.strip().endswith("!")

def analyze_sentiment_nltk(text):
    return sia.polarity_scores(text)

def remove_character_prefix(text):
    return re.sub(r"^[A-Z]+:\s*", "", text)

df["Text"] = df["Text"].apply(remove_character_prefix)

sentiment_results = df['Text'].apply(analyze_sentiment_nltk)

# Dodanie kolumn
df["Word_Count"] = df["Text"].apply(word_count)
df["Contains_Profanity"] = df["Text"].apply(contains_profanity)
df["Contains_Italian"] = df["Text"].apply(contains_italian_word)
df["Character_Count"] = df["Text"].apply(character_count)
df["Is_Question"] = df["Text"].apply(is_question)
df["Is_Exclamation"] = df["Text"].apply(is_exclamation)
df["Contains_Name"] = df["Text"].apply(contains_name)
df["Contains_Food"] = df["Text"].apply(contains_food_reference)
df['Sentiment_Compound'] = [result['compound'] for result in sentiment_results]

# Sprawdź liczbę brakujących wartości w kolumnie "Text"
df = df.dropna(subset=['Text'])
missing_values_count = df['Text'].isnull().sum()

if missing_values_count > 0:
    print(f"Kolumna 'Text' zawiera {missing_values_count} brakujących wartości.")
else:
    print("Kolumna 'Text' nie zawiera brakujących wartości.")

# Zapis do nowego pliku CSV
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Zapisano wzbogacone dane do pliku {output_file}")
