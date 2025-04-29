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

# Load data
data = pd.read_csv("../../data/shortened_data.csv")
data = data.drop(columns=['track_name', 'artist_name', 'release_name'])
data['explicit'] = data['explicit'].astype(int)
print("Data loaded")

# Prepare data
artist_ids = data['artist_id'].astype('category').cat.codes.values
track_ids = data['track_id'].astype('category').cat.codes.values
release_ids = data['release_id'].astype('category').cat.codes.values
print("Data prepared")

# Create mappings between original IDs and encoded IDs
track_id_map = dict(zip(data['track_id'], track_ids))
artist_id_map = dict(zip(data['artist_id'], artist_ids))
release_id_map = dict(zip(data['release_id'], release_ids))
print("Mappings created")

# Number of unique values for each categorical column
n_tracks = len(data['track_id'].unique())
n_artists = len(data['artist_id'].unique())
n_releases = len(data['release_id'].unique())
print(f"Unique values - Tracks: {n_tracks}, Artists: {n_artists}, Releases: {n_releases}")

# Embedding dimension
embedding_dim = 21

# Create positive pairs (artist-track interactions)
positive_pairs = data[['artist_id', 'track_id', 'release_id']].drop_duplicates()
positive_pairs['rating'] = 1  # Positive interaction
print("Positive pairs created")

# Create negative pairs (negative sampling)
np.random.seed(42)
negative_samples = []
artist_track_map = defaultdict(set)

# Create a map of artist to their tracks and releases
for _, row in data.iterrows():
    artist_track_map[(row['artist_id'], row['track_id'])] = row['release_id']
print("Artist-track-release map created")

# Generate negative samples
all_tracks = data[['track_id', 'release_id']].drop_duplicates()
all_tracks_list = all_tracks.to_dict('records')

for artist in data['artist_id'].unique():
    # Get artist's actual tracks
    artist_tracks = [k[1] for k in artist_track_map.keys() if k[0] == artist]
    
    # Sample negative tracks
    possible_negatives = [t for t in all_tracks_list if t['track_id'] not in artist_tracks]
    
    if possible_negatives:
        sampled_negatives = np.random.choice(
            possible_negatives, 
            size=min(len(artist_tracks), len(possible_negatives)), 
            replace=False
        )
        
        for neg in sampled_negatives:
            negative_samples.append({
                'artist_id': artist,
                'track_id': neg['track_id'],
                'release_id': neg['release_id'],
                'rating': 0
            })

negative_pairs = pd.DataFrame(negative_samples)
print("Negative samples generated")

# Combine positive and negative pairs
all_pairs = pd.concat([positive_pairs, negative_pairs])
print(f"Total pairs: {len(all_pairs)} (Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)})")

# Add numerical features for each track
# First ensure we keep release_id in track_features
track_features = data.drop(columns=['artist_id', 'popularity']).drop_duplicates('track_id')
print("Track features prepared")

# Verify release_id exists before merging
assert 'release_id' in track_features.columns, "release_id missing from track_features"

# Merge with all_pairs
all_pairs = all_pairs.merge(
    track_features.drop(columns=['release_id']),  # We already have release_id from the pairs
    on='track_id',
    how='left'
)
print("Numerical features added")

# Prepare data for model
X = all_pairs.drop(columns=['rating'])
y = all_pairs['rating']

# Verify release_id exists before splitting
assert 'release_id' in X.columns, "release_id missing before train-test split"

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train and test")

# Prepare categorical features
artist_ids_train = X_train['artist_id'].map(artist_id_map).values
track_ids_train = X_train['track_id'].map(track_id_map).values
release_ids_train = X_train['release_id'].map(release_id_map).values

artist_ids_test = X_test['artist_id'].map(artist_id_map).values
track_ids_test = X_test['track_id'].map(track_id_map).values
release_ids_test = X_test['release_id'].map(release_id_map).values
print("Categorical features prepared")

# Normalize numerical features
numeric_cols = X_train.columns.difference(['artist_id', 'track_id', 'release_id'])
scaler = StandardScaler()
numeric_features_train = scaler.fit_transform(X_train[numeric_cols])
numeric_features_test = scaler.transform(X_test[numeric_cols])
print("Numerical features normalized")

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

def recommend_tracks(artist_id, top_n=10):
    """Recommend tracks for a given artist"""
    # Get all unique tracks
    all_tracks = data[['track_id', 'release_id']].drop_duplicates('track_id')
    
    # Create input data for prediction
    artist_ids = np.array([artist_id_map[artist_id]] * len(all_tracks))
    track_ids = np.array([track_id_map[t] for t in all_tracks['track_id']])
    release_ids = np.array([release_id_map[r] for r in all_tracks['release_id']])
    
    # Get numeric features for these tracks
    numeric_features = all_tracks.merge(track_features, on='track_id')
    
    # Remove the ID columns before scaling
    cols_to_drop = ['track_id', 'release_id_x', 'release_id_y']  # Include both possible release_id columns
    cols_to_drop = [col for col in cols_to_drop if col in numeric_features.columns]
    numeric_features = numeric_features.drop(columns=cols_to_drop)
    
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