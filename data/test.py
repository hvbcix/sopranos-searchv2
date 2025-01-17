# Lista postaci z The Sopranos
characters = [
    "Tony Soprano",
    "Jennifer Melfi",
    "Carmela Soprano",
    "Christopher Moltisanti",
    "Corrado 'Junior' Soprano",
    "Salvatore 'Big Pussy' Bonpensiero",
    "Silvio Dante",
    "Paulie Gualtieri",
    "Anthony 'A.J.' Soprano Jr.",
    "Meadow Soprano",
    "Livia Soprano",
    "Adriana La Cerva",
    "Richie Aprile",
    "Janice Soprano",
    "Artie Bucco",
    "Furio Giunta",
    "Robert 'Bobby Bacala' Baccalieri",
    "Eugene Pontecorvo",
    "Charmaine Bucco",
    "Ralph Cifaretto",
    "John 'Johnny Sack' Sacrimoni",
    "Anthony 'Tony B' Blundetto",
    "Vito Spatafore",
    "Rosalie Aprile",
    "Patsy Parisi",
    "Carmine 'Little Carmine' Lupertazzi",
    "Phil Leotardo",
    "Angie Bonpensiero",
    "Butch DeConcini",
    "Benny Fazio",
    "Paul 'Little Paulie' Germani",
    "Carlo Gervasi",
    "Gabriella Dante",
    "Giovanni 'Johnny Boy' Soprano",
    "Richard 'Dickie' Moltisanti"
]

# Tworzenie pliku tekstowego
output_file = "sopranos_characters_split.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for character in characters:
        parts = character.replace("'", "").replace('"', "").split()  # Usunięcie apostrofów i podział
        for part in parts:
            f.write(part + "\n")  # Zapisanie każdej części w osobnej linii
        f.write("\n")  # Dodanie pustej linii między postaciami

print(f"Plik '{output_file}' został utworzony.")
