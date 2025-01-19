import sqlite3
import pandas as pd

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
    # Use a less strict regex to match words surrounded by punctuation
    total_occurrences = df['Text'].str.count(rf'\b{query}(?!\w)').sum()

    return total_occurrences