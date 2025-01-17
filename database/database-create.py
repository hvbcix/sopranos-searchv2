import sqlite3
import pandas as pd

# Ścieżka do pliku SQLite w folderze "database"
database_path = "database/sopranos_data.db"  # Upewnij się, że folder "database" istnieje

# Ścieżka do pliku CSV
csv_file_path = "data/sopranos_transcripts_enriched.csv"  # Plik CSV musi znajdować się w folderze "database"

# Wczytanie danych z pliku CSV
df = pd.read_csv(csv_file_path)

# Połączenie z SQLite
conn = sqlite3.connect(database_path)

# Zapisanie danych z CSV do bazy danych SQLite
table_name = "transcripts"
df.to_sql(table_name, conn, if_exists="replace", index=False)

# Sprawdzenie, czy dane zostały poprawnie zapisane
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

query = "SELECT * FROM transcripts LIMIT 50;"
df = pd.read_sql_query(query, conn)

# Wyświetlenie wyników
print(df)

# Zamknięcie połączenia
conn.close()

print(f"Baza danych została utworzona w lokalizacji: {database_path}")
print(f"Dostępne tabele w bazie: {tables}")
