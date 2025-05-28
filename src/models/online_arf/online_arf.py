from river import ensemble, metrics, preprocessing, stream, evaluate
from river.tree import HoeffdingTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import defaultdict
import time

class RiverStreamRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = preprocessing.MinMaxScaler()
        self.track_features = None
        self.artist_track_map = defaultdict(set)
        self.track_id_map = {}
        self.artist_id_map = {}
        self.metric = metrics.Accuracy()

    def preprocess_data(self, file_path, sample_size=5000):
        """Load and preprocess initial data for streaming simulation."""
        print("Loading and preprocessing data...")
        data = pd.read_csv(file_path)
        data = data.drop(columns=['track_name', 'artist_name', 'release_name', 'Unnamed: 0'])
        data['explicit'] = data['explicit'].astype(int)

        # Sample data if too large
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        # Create positive interactions (artist-track pairs)
        positive_pairs = data[['artist_id', 'track_id']].drop_duplicates()
        positive_pairs['interaction'] = 1

        # Create negative samples
        negative_samples = []
        all_tracks = set(data['track_id'].unique())

        # Create artist-track map
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])

        # Generate negative samples
        for artist in self.artist_track_map:
            artist_tracks = self.artist_track_map[artist]
            negative_tracks = list(all_tracks - artist_tracks)
            
            if negative_tracks:
                sampled_negatives = np.random.choice(
                    negative_tracks,
                    size=min(len(artist_tracks), len(negative_tracks)),
                    replace=False
                )
                for track in sampled_negatives:
                    negative_samples.append({
                        'artist_id': artist,
                        'track_id': track,
                        'interaction': 0
                    })

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs])

        # Add track features
        self.track_features = data.drop(columns=['artist_id', 'release_id', 'popularity']).drop_duplicates('track_id')
        all_pairs = all_pairs.merge(self.track_features, on='track_id')

        # Encode categorical features
        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])

        # Create mappings
        self.track_id_map = dict(zip(all_pairs['track_id'], all_pairs['track_id'].astype('category').cat.codes))
        self.artist_id_map = dict(zip(all_pairs['artist_id'], all_pairs['artist_id'].astype('category').cat.codes))

        # Scale numerical features
        numeric_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                        'speechiness', 'tempo', 'time_signature', 'valence', 'explicit']
        
        all_pairs[numeric_cols] = self.scaler.learn_many(all_pairs[numeric_cols]).transform_many(all_pairs[numeric_cols])

        return all_pairs

    def initialize_model(self):
        """Initialize Adaptive Random Forest (ARF) model."""
        self.model = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,  # Number of trees in the forest
            drift_detector=True,  # Enable concept drift detection
            warning_detector=True,  # Enable warning detection
            seed=42
        )

    def train_on_stream(self, data, stream_size=3000):
        """Train the model on a simulated data stream."""
        print(f"Training on stream (size: {stream_size})...")
        
        # Convert to river-compatible stream
        features = ['artist_id', 'track_id', 'danceability', 'energy', 'valence']
        X = data[features]
        y = data['interaction']
        
        dataset = stream.iter_pandas(X, y)
        
        start_time = time.time()
        accuracy = metrics.Accuracy()
        
        for i, (x, y_true) in enumerate(dataset, 1):
            # Scale numerical features on the fly
            x_scaled = self.scaler.transform_one(x)
            
            # Predict
            y_pred = self.model.predict_one(x_scaled)
            
            # Update model
            self.model.learn_one(x_scaled, y_true)
            
            # Update accuracy metric
            accuracy.update(y_true, y_pred)
            
            if i % 500 == 0:
                print(f"Processed {i} samples | Accuracy: {accuracy.get():.4f}")
            
            if i >= stream_size:
                break
        
        end_time = time.time()
        print(f"Training completed. Total samples processed: {i}")
        print(f"Final accuracy: {accuracy.get():.4f}")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")

    def recommend_tracks(self, artist_id, top_n=5):
        """Recommend tracks for a given artist."""
        if artist_id not in self.artist_id_map:
            print(f"Artist {artist_id} not seen before. Returning popular tracks.")
            return self.get_popular_tracks(top_n)
        
        # Get all tracks
        all_tracks = self.track_features['track_id'].unique()
        recommendations = []
        
        for track_id in all_tracks:
            if track_id in self.track_id_map:
                track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
                
                # Prepare feature vector
                x = {
                    'artist_id': artist_id,
                    'track_id': track_id,
                    'danceability': track_data['danceability'],
                    'energy': track_data['energy'],
                    'valence': track_data['valence']
                }
                
                # Predict interaction probability
                x_scaled = self.scaler.transform_one(x)
                y_proba = self.model.predict_proba_one(x_scaled)
                score = y_proba.get(1, 0.0)  # Probability of positive interaction
                
                recommendations.append({
                    'track_id': track_id,
                    'score': score,
                    'danceability': track_data['danceability'],
                    'energy': track_data['energy'],
                    'valence': track_data['valence']
                })
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

    def get_popular_tracks(self, top_n=5):
        """Fallback method to get popular tracks."""
        if self.track_features is None:
            return []
        
        popular_tracks = self.track_features.sort_values(
            by=['danceability', 'energy'], 
            ascending=False
        ).head(top_n)
        
        return popular_tracks.to_dict('records')

# Example Usage
recommender = RiverStreamRecommender()

# Step 1: Preprocess data
data = recommender.preprocess_data("../../data/full_training_data.csv", sample_size=5000)

# Step 2: Initialize ARF model
recommender.initialize_model()

# Step 3: Train on stream
recommender.train_on_stream(data, stream_size=3000)

# Step 4: Get recommendations
example_artist = data['artist_id'].iloc[0]
print(f"\nGetting recommendations for artist {example_artist}...")
recommendations = recommender.recommend_tracks(example_artist, top_n=5)

print("\nTop recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. Track ID: {rec['track_id']} | Score: {rec['score']:.3f} | "
            f"Danceability: {rec['danceability']:.2f} | Energy: {rec['energy']:.2f}")