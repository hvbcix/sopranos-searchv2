import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# URL głównej strony serialu
base_url = "https://subslikescript.com/series/The_Sopranos-141842"

# Pobierz zawartość strony
response = requests.get(base_url)
soup = BeautifulSoup(response.content, "html.parser")

# Znajdź wszystkie linki do odcinków
season_links = soup.find_all("a", href=True)
episode_links = [
    "https://subslikescript.com" + link["href"]
    for link in season_links
    if "/series/The_Sopranos-141842/season-" in link["href"]
]

# Funkcja do scrapowania jednego odcinka
def scrape_episode(episode_url):
    response = requests.get(episode_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Znajdź sekcję z transkryptem
    transcript_section = soup.find("div", class_="full-script")
    if not transcript_section:
        print(f"Brak transkryptu dla: {episode_url}")
        return []
    
    # Wzorzec do wyciągnięcia sezonu, odcinka i nazwy odcinka
    pattern = r"season-(\d+)/episode-(\d+)-(.+)"
    match = re.search(pattern, episode_url)
    
    if not match:
        raise ValueError(f"URL nie pasuje do wzorca: {episode_url}")
    
    # Pobranie wartości z grup w regexie
    season_number = int(match.group(1))  # Sezon
    episode_number = int(match.group(2))  # Odcinek
    episode_name = match.group(3).replace("_", " ")  # Nazwa odcinka z zamianą "_" na spacje


    # Podziel transkrypt na linie
    lines = transcript_section.get_text(strip=True, separator="\n").split("\n")

    # Utwórz dane dla każdej linii
    data = []
    for line_number, line in enumerate(lines, start=1):
        cleaned_line = line.lstrip("-").strip()  # Usuwa "-" z początku i nadmiarowe spacje        
        data.append({
            "Line Number": line_number,
            "Text": cleaned_line,
            "Season": season_number,
            "Episode": episode_number,
            "Episode Name": episode_name
        })
    return data

# Główna funkcja do scrapowania wszystkich odcinków
def scrape_all_episodes(links):
    all_data = []
    for link in links:
        print(f"Scrapowanie: {link}")
        episode_data = scrape_episode(link)
        all_data.extend(episode_data)
    return all_data

# Scrapuj wszystkie odcinki
all_transcripts = scrape_all_episodes(episode_links)

# Zapisz dane do pliku CSV
df = pd.DataFrame(all_transcripts)
output_file = "sopranos_transcripts_full.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Transkrypty zapisane do pliku {output_file}")