import pandas as pd

# Wczytaj dane
audio_features = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/audio_features.csv")
r_track_artist = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/r_track_artist.csv")
artists = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/artists.csv")
tracks = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/tracks.csv")
r_albums_tracks = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/r_albums_tracks.csv")
albums = pd.read_csv("../../data/8+ M. Spotify Tracks, Genre, Audio Features/original/albums.csv")

# Usuń niepotrzebne kolumny
audio_features.drop(columns=['analysis_url'], inplace=True)

# Połącz audio_features z r_track_artist
total = pd.merge(audio_features, r_track_artist, left_on='id', right_on='track_id', how='left')

# Połącz z artists
total = pd.merge(total, artists[['id', 'name']], left_on='artist_id', right_on='id', how='left')
total.rename(columns={'name': 'artist_name'}, inplace=True)
total.drop(columns=['id_x', 'id_y'], inplace=True)

# Zmień nazwę kolumny duration
total.rename(columns={'duration': 'duration_ms'}, inplace=True)

# Zmień typy danych, aby zaoszczędzić pamięć
total = total.astype({
    'acousticness': 'float32',
    'danceability': 'float32',
    'duration_ms': 'int32',
    'energy': 'float32',
    'instrumentalness': 'float32',
    'key': 'int8',
    'liveness': 'float32',
    'loudness': 'float32',
    'mode': 'int8',
    'speechiness': 'float32',
    'tempo': 'float32',
    'time_signature': 'int8',
    'valence': 'float32',
    'artist_name': 'category'
})

# Przetwarzanie w partiach
batch_size = 1_000_000  # Przetwarzaj 1 mln wierszy na raz
total_rows = len(total)
processed_data = []

for start in range(0, total_rows, batch_size):
    end = start + batch_size
    batch = total[start:end].copy()  # Kopiuj partię danych

    # Grupowanie i agregacja
    grouped_batch = batch.groupby('track_id').agg({
        'acousticness': 'first',
        'danceability': 'first',
        'duration_ms': 'first',
        'energy': 'first',
        'instrumentalness': 'first',
        'key': 'first',
        'liveness': 'first',
        'loudness': 'first',
        'mode': 'first',
        'speechiness': 'first',
        'tempo': 'first',
        'time_signature': 'first',
        'valence': 'first',
        'artist_id': 'first',
        'artist_name': lambda x: list(x.unique())  # Lista unikalnych nazw artystów
    }).reset_index()

    processed_data.append(grouped_batch)

# Połącz przetworzone partie
grouped = pd.concat(processed_data)

# Połącz z tracks
grouped = pd.merge(grouped, tracks[['id', 'name', 'popularity', 'explicit']], left_on='track_id', right_on='id', how='left')
grouped.rename(columns={'name': 'track_name'}, inplace=True)
grouped.drop(columns='id', inplace=True)

# Połącz z r_albums_tracks i albums
grouped = pd.merge(grouped, r_albums_tracks[['track_id', 'album_id']], on='track_id', how='left')
grouped = pd.merge(grouped, albums[['id', 'name']], left_on='album_id', right_on='id', how='left')
grouped.rename(columns={'album_id': 'release_id', 'name': 'release_name'}, inplace=True)
grouped.drop(columns='id', inplace=True)

# Wybierz końcowe kolumny
final_columns = [
    'track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name', 
    'explicit', 'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 
    'time_signature', 'valence'
]
final_data = grouped[final_columns]

final_data = final_data.astype({
    'popularity': 'int8',  # Zmień na int8
    'explicit': 'bool'     # Zmień na bool
})

# Zapisz wynik
final_data.to_csv('../../data/8+ M. Spotify Tracks, Genre, Audio Features/8m_data.csv', index=False)

# Podsumowanie
print(final_data.columns)
print(final_data.shape[0])