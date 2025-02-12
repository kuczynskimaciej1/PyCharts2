import pandas as pd

total = pd.DataFrame()

audio_features = pd.read_csv("../../data/10+ M. Beatport Tracks/original/audio_features.csv")
audio_features = audio_features.drop(columns='updated_on')
sp_track = pd.read_csv("../../data/10+ M. Beatport Tracks/original/sp_track.csv")
sp_artist_track = pd.read_csv("../../data/10+ M. Beatport Tracks/original/sp_artist_track.csv")
sp_artist = pd.read_csv("../../data/10+ M. Beatport Tracks/original/sp_artist.csv")
sp_release = pd.read_csv("../../data/10+ M. Beatport Tracks/original/sp_release.csv")

total = pd.merge(audio_features, sp_track[['track_id', 'track_title', 'isrc', 'release_id', 'explicit']], left_on='isrc', right_on='isrc', how='left')
total = pd.merge(total, sp_artist_track[['track_id', 'artist_id']], left_on='track_id', right_on='track_id', how='left')
total = pd.merge(total, sp_artist[['artist_id', 'artist_name']], left_on='artist_id', right_on='artist_id', how='left')
total = pd.merge(total, sp_release[['release_id', 'popularity', 'release_title']], left_on='release_id', right_on='release_id', how='left')

total = total.drop(columns=['isrc'])

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
    'artist_name': 'category',
    'popularity': 'int8',
    'explicit': 'bool',
})

total = total.groupby('track_id').agg({
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
    'track_title': 'first',
    'popularity': 'first',
    'explicit': 'first',
    'release_id': 'first',
    'release_title': 'first',
    'artist_id': 'first',
    'artist_name': lambda x: list(x.unique())  # Lista unikalnych nazw artystów
}).reset_index()

total = total.rename(columns={'track_title': 'track_name'})
total = total.rename(columns={'release_title': 'release_name'})
total = total[['track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name', 'explicit', 
               'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
               'speechiness', 'tempo', 'time_signature', 'valence']]

print(total.columns)
print(total.shape[0])

total.to_csv('../../data/10+ M. Beatport Tracks/10m_data.csv', index=False)