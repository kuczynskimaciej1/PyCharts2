from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import DataStream
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
import time

class MusicStreamRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.track_features = None
        self.artist_track_map = defaultdict(set)
        self.track_id_map = {}
        self.artist_id_map = {}
    
    def preprocess_data(self, file_path, sample_size=10000):
        """Load and preprocess the initial batch of data"""
        print("Loading and preprocessing initial data...")
        data = pd.read_csv(file_path)
        data = data.drop(columns=['track_name', 'artist_name', 'release_name', 'Unnamed: 0'])
        data['explicit'] = data['explicit'].astype(int)
        
        # Sample data if too large for initial processing
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
        
        all_pairs[numeric_cols] = self.scaler.fit_transform(all_pairs[numeric_cols])
        
        return all_pairs
    
    def initialize_model(self):
        """Initialize the Hoeffding Tree model"""
        self.model = HoeffdingTree(
            nominal_attributes=[0, 1],  # artist_id and track_id are nominal
            grace_period=200,
            split_confidence=1e-7,
            tie_threshold=0.05,
            binary_split=False,
            stop_mem_management=False,
            remove_poor_atts=False,
            leaf_prediction='mc'  # majority class
        )
    
    def simulate_data_stream(self, data, stream_size=1000, batch_size=100):
        """Simulate a data stream from the prepared data"""
        print(f"Simulating data stream with {stream_size} samples...")
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        
        # Convert to numpy arrays
        X_np = X[['artist_id', 'track_id', 'acousticness', 'danceability', 'energy']].values
        y_np = y.values
        
        # Create data stream
        stream = DataStream(X_np, y_np)
        
        # Train on the stream
        self.train_on_stream(stream, stream_size, batch_size)
    
    def train_on_stream(self, stream, stream_size, batch_size):
        """Train the model on the data stream"""
        print(f"Training on stream (batch size: {batch_size})...")
        n_samples = 0
        correct = 0
        start_time = time.time()
        
        while n_samples < stream_size and stream.has_more_samples():
            X_batch, y_batch = stream.next_sample(batch_size)
            
            # Partial fit
            self.model.partial_fit(X_batch, y_batch)
            
            # Predict and calculate accuracy
            y_pred = self.model.predict(X_batch)
            batch_correct = np.sum(y_pred == y_batch)
            correct += batch_correct
            n_samples += batch_size
            
            # Print progress
            if n_samples % (batch_size * 10) == 0:
                batch_acc = batch_correct / batch_size
                overall_acc = correct / n_samples
                print(f"Processed {n_samples} samples - Batch accuracy: {batch_acc:.2f}, Overall accuracy: {overall_acc:.2f}")
        
        end_time = time.time()
        print(f"Training completed. Total samples processed: {n_samples}")
        print(f"Final accuracy: {correct/n_samples:.2f}")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Recommend tracks for a given artist using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call initialize_model() and train_on_stream() first.")
        
        if artist_id not in self.artist_id_map:
            print(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self.get_popular_tracks(top_n)
        
        # Get all tracks
        all_tracks = self.track_features['track_id'].unique()
        recommendations = []
        
        # Prepare batch for prediction
        batch_size = 1000
        for i in range(0, len(all_tracks), batch_size):
            batch_tracks = all_tracks[i:i+batch_size]
            
            # Create feature matrix
            X_batch = []
            for track_id in batch_tracks:
                if track_id in self.track_id_map:
                    track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
                    X_batch.append([
                        self.artist_id_map[artist_id],
                        self.track_id_map[track_id],
                        track_data['acousticness'],
                        track_data['danceability'],
                        track_data['energy']
                    ])
            
            if X_batch:
                X_batch = np.array(X_batch)
                # Predict probabilities (using predict_proba if available)
                try:
                    y_proba = self.model.predict_proba(X_batch)
                    pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                except:
                    # Fallback to predict if predict_proba not available
                    y_pred = self.model.predict(X_batch)
                    pos_proba = y_pred
                
                # Add to recommendations
                for j, track_id in enumerate(batch_tracks):
                    if track_id in self.track_id_map:
                        recommendations.append({
                            'track_id': track_id,
                            'score': pos_proba[j],
                            'acousticness': X_batch[j][2],
                            'danceability': X_batch[j][3],
                            'energy': X_batch[j][4]
                        })
        
        # Sort and get top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def get_popular_tracks(self, top_n=5):
        """Fallback method to get popular tracks"""
        if self.track_features is None:
            return []
        
        # Use danceability and energy as proxy for popularity
        popular_tracks = self.track_features.sort_values(
            by=['danceability', 'energy'], 
            ascending=False
        ).head(top_n)
        
        return popular_tracks.to_dict('records')

# Example usage
recommender = MusicStreamRecommender()

# Step 1: Preprocess initial data
data = recommender.preprocess_data("../../data/full_training_data.csv", sample_size=5000)

# Step 2: Initialize the Hoeffding Tree model
recommender.initialize_model()

# Step 3: Simulate and train on data stream
recommender.simulate_data_stream(data, stream_size=2000, batch_size=100)

# Step 4: Get recommendations for an artist
example_artist = data['artist_id'].iloc[0]
print(f"\nGetting recommendations for artist {example_artist}...")
recommendations = recommender.recommend_tracks(example_artist, top_n=5)

print("\nTop recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. Track ID: {rec['track_id']} | Score: {rec['score']:.3f} | "
            f"Danceability: {rec['danceability']:.2f} | Energy: {rec['energy']:.2f}")