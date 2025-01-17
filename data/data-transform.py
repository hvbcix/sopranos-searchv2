import pandas as pd
import string
import re

# Wczytaj listę przekleństw z pliku
with open("data/badwords.txt", "r", encoding="utf-8") as f:
    profanities = set(f.read().splitlines())  # Tworzymy zestaw słów

with open("data/ita.txt", "r", encoding="utf-8") as f:
    italian_words = set(f.read().splitlines())  # Tworzymy zestaw słów

with open("data/food.txt", "r", encoding="utf-8") as f:
    food_words = set(f.read().splitlines())

# Wczytanie listy imion z pliku names.txt
with open("data/names.txt", "r", encoding="utf-8") as f:
    names = set(name.strip().lower() for name in f.readlines())  # Ignorujemy wielkość liter


# Wczytaj dane z CSV
input_file = "data/sopranos_transcripts_full.csv"
output_file = "data/sopranos_transcripts_enriched.csv"
df = pd.read_csv(input_file)

# Przypisanie nowego Line_Number (od 1 do n)
df["Line_Number"] = range(1, len(df) + 1)

# Funkcje pomocnicze
def word_count(text):
    return len(text.split())

def contains_italian_word(text):
    # Usuń interpunkcję z tekstu
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    words = text_cleaned.split()  # Podziel tekst na słowa
    for word in words:
        if word.lower() in italian_words:  # Sprawdź, czy słowo znajduje się w zestawie włoskich słów
            return True
    return False

def contains_profanity(text):
    # Usuń interpunkcję z tekstu
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    # Sprawdź, czy jakiekolwiek słowo znajduje się na liście przekleństw
    return any(word.lower() in profanities for word in text_cleaned.split())

def contains_name(text):
    # Usuwamy interpunkcję z tekstu
    text_cleaned = text.translate(str.maketrans("", "", string.punctuation))
    # Dzielimy tekst na słowa i sprawdzamy każde słowo
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

def remove_character_prefix(text):
    # Usuwamy prefiks w formacie "IMIĘ:" z początku linii
    return re.sub(r"^[A-Z]+:\s*", "", text)

df["Text"] = df["Text"].apply(remove_character_prefix)


# Dodanie kolumn
df["Word_Count"] = df["Text"].apply(word_count)
df["Contains_Profanity"] = df["Text"].apply(contains_profanity)
df["Contains_Italian"] = df["Text"].apply(contains_italian_word)
df["Character_Count"] = df["Text"].apply(character_count)
df["Is_Question"] = df["Text"].apply(is_question)
df["Is_Exclamation"] = df["Text"].apply(is_exclamation)
df["Contains_Name"] = df["Text"].apply(contains_name)
df["Contains_Food"] = df["Text"].apply(contains_food_reference)

# Usunięcie wierszy z pustymi wartościami w kolumnie "Text"
df = df.dropna(subset=['Text'])

# Sprawdź liczbę brakujących wartości w kolumnie "Text"
missing_values_count = df['Text'].isnull().sum()

if missing_values_count > 0:
    print(f"Kolumna 'Text' zawiera {missing_values_count} brakujących wartości.")
else:
    print("Kolumna 'Text' nie zawiera brakujących wartości.")

# Zapis do nowego pliku CSV
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Zapisano wzbogacone dane do pliku {output_file}")
