import pandas as pd
import ast

def prepare_data() -> None:
    november2018 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesNov2018.csv")
    april2019 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesApril2019.csv")
    total_saf = pd.concat([november2018, april2019], ignore_index = True)
    total_saf.drop_duplicates(subset=['track_id'])

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
    total_saf = total_saf[['track_id', 'track_name', 'artist_name', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
        'speechiness', 'tempo', 'time_signature', 'valence', 'popularity']]
    total_saf.to_csv("../../data/Spotify Audio Features/saf_data.csv")

def add_Spotify_info() -> None:
    pass