import pandas as pd
import ast
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import spotify_api
import time
import csv
from requests.exceptions import HTTPError


def prepare_data() -> None:
    november2018 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesNov2018.csv")
    april2019 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesApril2019.csv")
    total_saf = pd.concat([november2018, april2019], ignore_index = True)
    total_saf = total_saf.drop_duplicates(subset=['track_id'])

    # Zmień wartości w kolumnie artist_name na listy stringów
    def convert_to_list(artist_name):
        try:
            # Jeśli wartość jest już listą (np. "[artysta1, artysta2]"), zamień na listę
            return ast.literal_eval(artist_name)
        except (ValueError, SyntaxError):
            # Jeśli wartość to pojedynczy string, zamień na listę jednoelementową
            return [artist_name]

    # Zastosuj funkcję do kolumny artist_name
    total_saf['artist_name'] = total_saf['artist_name'].apply(convert_to_list)
    total_saf = total_saf[['track_id', 'track_name', 'artist_name', 'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
        'speechiness', 'tempo', 'time_signature', 'valence']]
    total_saf.to_csv("../../data/Spotify Audio Features/saf_data.csv")



def add_spotify_info(resume_from_track_id=None) -> None:
    # Wczytaj dane z pliku CSV
    total_saf = pd.read_csv("../../data/Spotify Audio Features/saf_data.csv")

    # Upewnij się, że token jest ważny
    spotify_api.ensure_token_valid()

    # Filtruj puste track_id
    total_saf = total_saf.dropna(subset=['track_id'])  # Usuń wiersze z pustym track_id
    total_saf = total_saf[total_saf['track_id'].str.len() == 22]  # Zostaw tylko poprawne track_id

    # Lista track_id
    track_ids = total_saf['track_id'].tolist()

    # Jeśli podano resume_from_track_id, znajdź indeks, od którego należy zacząć
    if resume_from_track_id:
        try:
            start_index = track_ids.index(resume_from_track_id)
            print(f"Wznawianie przetwarzania od track_id: {resume_from_track_id} (indeks: {start_index})")
            track_ids = track_ids[start_index:]  # Przetwarzaj od tego miejsca
        except ValueError:
            print(f"Track_id {resume_from_track_id} nie znaleziono na liście. Rozpoczynanie od początku.")
            start_index = 0
    else:
        start_index = 0

    # Podziel listę track_id na mniejsze partie (Spotify API ma limit 50 tracków na żądanie)
    batch_size = 50
    batches = [track_ids[i:i + batch_size] for i in range(0, len(track_ids), batch_size)]

    # Licznik żądań i przetworzonych utworów
    request_count = 0
    processed_tracks = start_index  # Uwzględnij już przetworzone utwory

    # Otwórz plik CSV do zapisu (tryb append, aby nie nadpisywać istniejących danych)
    with open("../../data/Spotify Audio Features/new_data.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Jeśli plik jest pusty, dodaj nagłówki
        if file.tell() == 0:
            writer.writerow(["track_id", "artist_ids", "album_id", "album_name", "explicit"])

        # Pobierz informacje o utworach z API Spotify
        for batch in batches:
            # Odnów token po 40 000 przetworzonych utworów
            if processed_tracks % 40000 == 0 and processed_tracks > 0:
                print(f"Przetworzono {processed_tracks} utworów. Odnawianie tokenu...")
                spotify_api.ensure_token_valid()

            # Konwertuj listę track_id na ciąg znaków oddzielonych przecinkami
            #track_ids_str = ','.join(batch)

            # Debugowanie: Wyświetl track_ids_str przed wysłaniem żądania
            #print(f"Przetwarzanie partii: {track_ids_str}")

            # Wykonaj żądanie do API Spotify
            while True:  # Pętla do ponawiania żądań w przypadku błędu 429
                try:
                    tracks_info = spotify_api.spotify.tracks(batch)
                    request_count += 1

                    # Przetwórz wyniki i zapisz do pliku CSV
                    for track_info in tracks_info['tracks']:
                        if track_info:  # Sprawdź, czy wynik nie jest pusty
                            track_id = track_info['id']
                            # Pobierz listę ID wszystkich artystów
                            artist_ids = ','.join([artist['id'] for artist in track_info['artists']])
                            album_id = track_info['album']['id']
                            album_name = track_info['album']['name']
                            explicit = track_info['explicit']

                            # Zapisz dane do pliku CSV
                            writer.writerow([track_id, artist_ids, album_id, album_name, explicit])
                            processed_tracks += 1

                            # Printuj postęp co 100 utworów
                            if processed_tracks % 100 == 0:
                                print(f"Przetworzono {processed_tracks} utworów...")

                    # Dodaj opóźnienie po każdym żądaniu (np. 1 sekunda)
                    time.sleep(1)  # Opóźnienie 1 sekunda między żądaniami

                    # Jeśli wykonano 50 żądań, dodaj dłuższe opóźnienie (np. 10 sekund)
                    if request_count % 50 == 0:
                        print(f"Wykonano {request_count} żądań. Dodaję dłuższe opóźnienie...")
                        time.sleep(10)  # Dłuższe opóźnienie co 50 żądań

                    break  # Wyjdź z pętli while, jeśli żądanie zakończyło się sukcesem

                except HTTPError as e:
                    if e.response.status_code == 429:
                        # Błąd 429: Too Many Requests
                        retry_after = int(e.response.headers.get('Retry-After', 10))  # Pobierz czas oczekiwania z nagłówka
                        print(f"Błąd 429: Przekroczono limit żądań. Ponawiam próbę za {retry_after} sekund...")
                        time.sleep(retry_after)  # Poczekaj określony czas
                    else:
                        # Inny błąd HTTP
                        print(f"Błąd HTTP: {e}")
                        break  # Wyjdź z pętli while w przypadku innych błędów

                except Exception as e:
                    # Inne błędy
                    print(f"Błąd podczas pobierania danych: {e}")
                    break  # Wyjdź z pętli while w przypadku innych błędów

    print(f"Zakończono! Przetworzono łącznie {processed_tracks} utworów.")
    print("Dane zostały zapisane do pliku new_data.csv.")