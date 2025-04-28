from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dot, Subtract
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.engine import data_adapter
from collections import defaultdict

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Wczytanie danych
data = pd.read_csv("../../data/shortened_data.csv")
data = data.drop(columns=['track_name', 'artist_name', 'release_name'])
data['explicit'] = data['explicit'].astype(int)
print("Wczytanie danych")

# Przygotowanie danych
artist_ids = data['artist_id'].astype('category').cat.codes.values
track_ids = data['track_id'].astype('category').cat.codes.values
release_ids = data['release_id'].astype('category').cat.codes.values
print("Przygotowanie danych")

# Create mappings between original IDs and encoded IDs
track_id_map = dict(zip(data['track_id'], track_ids))
artist_id_map = dict(zip(data['artist_id'], artist_ids))
release_id_map = dict(zip(data['release_id'], release_ids))
print("Create mappings between original IDs and encoded IDs")

# Liczba unikalnych wartości dla każdej kolumny kategorycznej
n_tracks = len(data['track_id'].unique())
n_artists = len(data['artist_id'].unique())
n_releases = len(data['release_id'].unique())
print("Liczba unikalnych wartości dla każdej kolumny kategorycznej")

# Wymiar embeddingów
embedding_dim = 21

# Create positive pairs (user-item interactions)
# Here we'll treat artists as "users" and tracks as "items"
positive_pairs = data[['artist_id', 'track_id']].drop_duplicates()
positive_pairs['rating'] = 1  # Positive interaction
print("Create positive pairs (user-item interactions)")

# Create negative pairs (negative sampling)
np.random.seed(42)
negative_samples = []
artist_track_map = defaultdict(set)
print("Create negative pairs (negative sampling)")

# Create a map of artist to their tracks
for _, row in positive_pairs.iterrows():
    artist_track_map[row['artist_id']].add(row['track_id'])
print("Create a map of artist to their tracks")

# Generate negative samples
for artist in artist_track_map:
    artist_tracks = artist_track_map[artist]
    all_tracks = set(data['track_id'].unique())
    negative_tracks = list(all_tracks - artist_tracks)
    
    # Sample as many negative tracks as positive ones
    if negative_tracks:
        sampled_negatives = np.random.choice(negative_tracks, size=min(len(artist_tracks), len(negative_tracks)), replace=False)
        for track in sampled_negatives:
            negative_samples.append({'artist_id': artist, 'track_id': track, 'rating': 0})

negative_pairs = pd.DataFrame(negative_samples)
print("Generate negative samples")

# Combine positive and negative pairs
all_pairs = pd.concat([positive_pairs, negative_pairs])

# Add numerical features for each track
track_features = data.drop(columns=['artist_id', 'release_id', 'popularity']).drop_duplicates('track_id')
all_pairs = all_pairs.merge(track_features, on='track_id')
print("Add numerical features for each track")

# Prepare data for model
X = all_pairs.drop(columns=['rating'])
y = all_pairs['rating']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare categorical features
artist_ids_train = X_train['artist_id'].map(artist_id_map).values
track_ids_train = X_train['track_id'].map(track_id_map).values
release_ids_train = X_train['release_id'].map(release_id_map).values

artist_ids_test = X_test['artist_id'].map(artist_id_map).values
track_ids_test = X_test['track_id'].map(track_id_map).values
release_ids_test = X_test['release_id'].map(release_id_map).values
print("Prepare categorical features")

# Normalize numerical features
numeric_features = X_train.drop(columns=['artist_id', 'track_id', 'release_id'])
scaler = StandardScaler()
numeric_features_train = scaler.fit_transform(numeric_features)
numeric_features_test = scaler.transform(X_test.drop(columns=['artist_id', 'track_id', 'release_id']))
print("Normalize numerical features")

# Warstwy wejściowe
artist_input = Input(shape=(1,), name='artist_input')
track_input = Input(shape=(1,), name='track_input')
release_input = Input(shape=(1,), name='release_input')
numeric_input = Input(shape=(numeric_features_train.shape[1],), name='numeric_input')

# Warstwy embeddingowe
artist_embedding = Embedding(n_artists, embedding_dim, name='artist_embedding')(artist_input)
track_embedding = Embedding(n_tracks, embedding_dim, name='track_embedding')(track_input)
release_embedding = Embedding(n_releases, embedding_dim, name='release_embedding')(release_input)

# Spłaszczenie embeddingów
artist_vec = Flatten()(artist_embedding)
track_vec = Flatten()(track_embedding)
release_vec = Flatten()(release_embedding)
print("Spłaszczenie embeddingów")

# Połączenie wszystkich cech
concat = Concatenate()([artist_vec, track_vec, release_vec, numeric_input])
print("Spłaszczenie embeddingów")

# Warstwy gęste
dense_1 = Dense(64, activation='relu')(concat)
dense_2 = Dense(32, activation='relu')(dense_1)
output_layer = Dense(1, activation='sigmoid')(dense_2)

# Budowa modelu
model = Model(inputs=[artist_input, track_input, release_input, numeric_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback do zapisywania modelu
checkpoint = ModelCheckpoint('recommender_model_best.h5', monitor='val_loss', 
                           save_best_only=True, save_weights_only=False, mode='min')

# Trening modelu
history = model.fit(
    [artist_ids_train, track_ids_train, release_ids_train, numeric_features_train],
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint],
    verbose=1
)

# Save history
with open('recommender_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
print("Save history")

# Wykres dokładności
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Recommender Model Training')
plt.legend()
plt.savefig('recommender_training.png')

# Recommendation function
def recommend_tracks(artist_id, top_n=10):
    """Recommend tracks for a given artist"""
    # Get all unique tracks
    all_tracks = data[['track_id', 'release_id']].drop_duplicates('track_id')
    
    # Create input data for prediction
    artist_ids = np.array([artist_id_map[artist_id]] * len(all_tracks))
    track_ids = np.array([track_id_map[t] for t in all_tracks['track_id']])
    release_ids = np.array([release_id_map[r] for r in all_tracks['release_id']])
    
    # Get numerical features for all tracks
    numeric_features = all_tracks.merge(track_features, on='track_id')
    numeric_features = numeric_features.drop(columns=['track_id', 'release_id'])
    numeric_features = scaler.transform(numeric_features)
    
    # Predict probabilities
    predictions = model.predict([artist_ids, track_ids, release_ids, numeric_features])
    
    # Get top recommendations
    all_tracks['score'] = predictions
    recommendations = all_tracks.sort_values('score', ascending=False).head(top_n)
    
    return recommendations

# Example recommendation
example_artist = data['artist_id'].iloc[0]  # First artist in dataset
recommendations = recommend_tracks(example_artist)
print(f"Top recommendations for artist {example_artist}:")
print(recommendations)