import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Wczytanie danych
data = pd.read_csv("../../data/full_training_data.csv")
data = data.drop(columns=['Unnamed: 0'])
data['explicit'] = data['explicit'].astype(int)

# Kodowanie cech kategorycznych
categorical_columns = ['track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Zamiana na liczby
    label_encoders[col] = le

# Normalizacja cech liczbowych
numerical_columns = ['acousticness', 'danceability', 'duration_ms', 'energy',
                     'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                     'speechiness', 'tempo', 'time_signature', 'valence']

scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Przygotowanie danych dla biblioteki Surprise
# Dodajemy więcej cech poprzez sumowanie ich wpływu do "popularity"
data['combined_feature'] = (
    data['track_id'] * 0.2 +
    data['artist_id'] * 0.2 +
    data['release_id'] * 0.1 +
    data['danceability'] * 0.1 +
    data['energy'] * 0.1 +
    data['valence'] * 0.1 +
    data['tempo'] * 0.1
)

reader = Reader(rating_scale=(data['combined_feature'].min(), data['combined_feature'].max()))
data_svd = Dataset.load_from_df(data[['track_id', 'artist_id', 'combined_feature']], reader)

# Podział na zbiór treningowy i testowy
trainset, testset = train_test_split(data_svd, test_size=0.2)

# Trenowanie modelu SVD
model_svd = SVD()
model_svd.fit(trainset)

# Walidacja krzyżowa
cv_results = cross_validate(model_svd, data_svd, cv=5, verbose=True)

# Zapis modelu
with open("svd_model_full.pkl", "wb") as f:
    pickle.dump(model_svd, f)

# Wykres błędu RMSE dla każdej walidacji
plt.plot(range(1, 6), cv_results['test_rmse'], marker='o', label='RMSE')
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("Wyniki walidacji krzyżowej SVD (wszystkie cechy)")
plt.legend()
plt.show()

# Funkcja rekomendacji na podstawie podanych utworów
def generate_playlist(input_tracks, model, data, label_encoders, top_n=10):
    unique_tracks = data[['track_id', 'track_name']].drop_duplicates()
    
    recommendations = []
    for track in input_tracks:
        encoded_track = label_encoders['track_id'].transform([track])[0]
        for candidate_track in unique_tracks['track_id'].values:
            if candidate_track != encoded_track:
                score = model.predict(encoded_track, candidate_track).est
                recommendations.append((candidate_track, score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_tracks = [track_id for track_id, _ in recommendations[:top_n]]
    
    return unique_tracks[unique_tracks['track_id'].isin(recommended_tracks)]['track_name'].values

# Przykładowa playlista
example_tracks = ['0009Q7nGlWjFzSjQIo9PmK', '000EFWe0HYAaXzwGbEU3rG']
recommended_playlist = generate_playlist(example_tracks, model_svd, data, label_encoders)
print("Recommended Playlist:", recommended_playlist)
