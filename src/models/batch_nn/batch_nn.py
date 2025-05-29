from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, log_loss)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
import psutil
import time
import seaborn as sns
import os
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import warnings
import logging
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Input, Embedding, Flatten, Dense, 
                                    Concatenate, Dropout, BatchNormalization,
                                    LeakyReLU)
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.metrics import AUC, Precision, Recall
from tensorflow.python.keras.losses import BinaryCrossentropy

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class DeepLearningRecommender:
    def __init__(self, experiment_name="dl_rec"):
        self._setup_directories(experiment_name)
        self._setup_logging(experiment_name)
        
        # Initialize model attributes
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.track_features = None
        self.artist_track_map = defaultdict(set)
        self.track_id_map = {}
        self.artist_id_map = {}
        self.genre_map = {}
        self.cluster_labels = None
        self.best_hyperparams = None
        self.experiment_name = experiment_name
        self.artist_encoder = None
        self.track_encoder = None
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': [],
            'train_auc': [],
            'val_auc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'training_time': []
        }
        
    def _setup_directories(self, experiment_name):
        """Create directories for outputs"""
        self.output_dir = f"outputs/{experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        
    def _save_visualization(self, fig, name):
        """Save visualization to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.output_dir}/plots/{name}_{timestamp}.png"
        
        try:
            if hasattr(fig, 'savefig'):
                fig.savefig(path, bbox_inches='tight', dpi=300)
            else:
                plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved visualization: {path}")
        except Exception as e:
            self.logger.error(f"Error saving visualization {name}: {str(e)}")
        finally:
            if 'fig' in locals():
                plt.close(fig)
        
    def _save_model(self):
        """Save model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.output_dir}/models/model_{timestamp}.h5"
        metadata_path = f"{self.output_dir}/models/metadata_{timestamp}.joblib"
        
        # Save Keras model
        self.model.save(model_path)
        
        # Save metadata
        save_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'track_features': self.track_features,
            'mappings': {
                'artist_track': dict(self.artist_track_map),
                'track_id': self.track_id_map,
                'artist_id': self.artist_id_map,
                'genre': self.genre_map
            },
            'cluster_labels': self.cluster_labels,
            'hyperparams': self.best_hyperparams,
            'metrics': self.metrics_history
        }
        
        joblib.dump(save_data, metadata_path)
        self.logger.info(f"Saved model: {model_path}")
        self.logger.info(f"Saved metadata: {metadata_path}")
        return model_path
        
    def _load_model(self, model_path, metadata_path):
        """Load saved model and metadata"""
        self.model = tf.keras.models.load_model(model_path)
        data = joblib.load(metadata_path)
        
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.track_features = data['track_features']
        self.artist_track_map = defaultdict(set, data['mappings']['artist_track'])
        self.track_id_map = data['mappings']['track_id']
        self.artist_id_map = data['mappings']['artist_id']
        self.genre_map = data['mappings']['genre']
        self.cluster_labels = data['cluster_labels']
        self.best_hyperparams = data['hyperparams']
        self.metrics_history = data.get('metrics', {})
        self.logger.info(f"Loaded model from: {model_path}")
        
    def _calculate_metrics(self, y_true, y_pred, y_proba=None, threshold=0.5):
        """Calculate comprehensive performance metrics"""
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = (y_true >= threshold).astype(int)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_squared_error(y_true, y_pred),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }
        
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true_binary, y_proba),
                'pr_auc': average_precision_score(y_true_binary, y_proba)
            })
        
        # Log classification report
        self.logger.info("\nClassification Report:\n" + classification_report(y_true_binary, y_pred_binary))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        self._save_visualization(fig, "confusion_matrix")
        
        return metrics
    
    def _plot_training_history(self, history):
        """Plot training history metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # RMSE
        axes[0, 1].plot(history.history['root_mean_squared_error'], label='Train RMSE')
        axes[0, 1].plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(history.history['auc'], label='Train AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('AUC')
        axes[1, 0].legend()
        
        # Precision-Recall
        axes[1, 1].plot(history.history['precision'], label='Train Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        self._save_visualization(fig, "training_history")
        
    def _plot_feature_importance(self):
        """Plot feature importance from the model"""
        try:
            # Get the embeddings for artists and tracks
            artist_embedding_layer = self.model.get_layer('artist_embedding')
            track_embedding_layer = self.model.get_layer('track_embedding')
            
            # Get weights
            artist_weights = artist_embedding_layer.get_weights()[0]
            track_weights = track_embedding_layer.get_weights()[0]
            
            # Calculate feature importance as variance
            artist_importance = np.var(artist_weights, axis=0)
            track_importance = np.var(track_weights, axis=0)
            
            # Plot artist embedding importance
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=artist_importance, y=[f"AE_{i}" for i in range(len(artist_importance))], ax=ax1)
            ax1.set_title('Artist Embedding Dimensions Importance')
            self._save_visualization(fig1, "artist_embedding_importance")
            
            # Plot track embedding importance
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=track_importance, y=[f"TE_{i}" for i in range(len(track_importance))], ax=ax2)
            ax2.set_title('Track Embedding Dimensions Importance')
            self._save_visualization(fig2, "track_embedding_importance")
            
        except Exception as e:
            self.logger.warning(f"Could not plot feature importance: {str(e)}")
            
    def preprocess_data(self, file_path, sample_size=10000):
        """Preprocessing with genre clustering and feature engineering"""
        self.logger.info("Loading and preprocessing data with hierarchical clustering...")
        data = pd.read_csv(file_path)
        
        # Check if 'interaction' column exists, if not create it
        if 'interaction' not in data.columns:
            self.logger.info("'Interaction' column not found - creating positive interactions")
            data['interaction'] = 1  # Assume all existing records are positive interactions
            self.logger.info("'Interaction' column created")

        # Create genre proxy from artist_id
        data['genre'] = data['artist_id'].apply(lambda x: hash(x) % 5)  # Simulate 5 genres
        self.logger.info("'Genre' column created")
        
        # Feature selection strategy
        feature_strategy = self._get_feature_strategy(strategy='content')
        cols_to_keep = feature_strategy + ['artist_id', 'track_id', 'genre', 'interaction']
        
        # Only keep columns that actually exist in the data
        cols_to_keep = [col for col in cols_to_keep if col in data.columns]
        data = data[cols_to_keep]
        self.logger.info("Columns to keep filtered")

        # Create positive pairs
        positive_pairs = data[['artist_id', 'track_id', 'genre']].drop_duplicates()
        positive_pairs['interaction'] = 1
        
        # Create artist-track-genre map
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])
            self.genre_map[row['artist_id']] = row['genre']
        self.logger.info("Artist-track-genre map created")
        
        # Generate negative samples
        negative_samples = []
        all_tracks = set(data['track_id'].unique())

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
                        'genre': self.genre_map[artist],
                        'interaction': 0
                    })

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs])
        self.logger.info(f"Created {len(negative_pairs)} negative samples")
        self.logger.info(f"Total pairs: {len(all_pairs)}")

        # Add track features
        self.track_features = data.drop(columns=['artist_id', 'interaction'], errors='ignore').drop_duplicates('track_id')
        all_pairs = all_pairs.merge(self.track_features, on='track_id')
        self.logger.info("Added track features")

        # Hierarchical clustering
        self._apply_hierarchical_clustering(all_pairs)
        self.logger.info("Applied hierarchical clustering")
        
        # Encode categorical features
        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])
        self.logger.info("Encoded categorical features")
        
        # Create mappings
        self.track_id_map = dict(zip(all_pairs['track_id'], all_pairs['track_id'].astype('category').cat.codes))
        self.artist_id_map = dict(zip(all_pairs['artist_id'], all_pairs['artist_id'].astype('category').cat.codes))
        self.logger.info("Created ID mappings")
        
        # Scale numerical features
        numeric_cols = [col for col in all_pairs.columns if col not in ['artist_id', 'track_id', 'genre', 'interaction', 'cluster']]
        if numeric_cols:
            all_pairs[numeric_cols] = self.scaler.fit_transform(all_pairs[numeric_cols])
            self.logger.info("Scaled numerical features")
        
        return all_pairs
    
    def _get_feature_strategy(self, strategy='content'):
        """Functional feature selection strategy"""
        if strategy == 'content':
            return ['acousticness', 'danceability', 'energy', 'valence', 'tempo']
        elif strategy == 'extended':
            return ['acousticness', 'danceability', 'duration_ms', 'energy',
                   'instrumentalness', 'liveness', 'loudness', 'speechiness',
                   'tempo', 'valence']
        else:
            return ['danceability', 'energy', 'valence']
    
    def _apply_hierarchical_clustering(self, data):
        """Apply hierarchical clustering to tracks"""
        cluster_features = data[['danceability', 'energy', 'valence']].values
        Z = linkage(cluster_features, method='ward')
        self.cluster_labels = fcluster(Z, t=3, criterion='maxclust')
        data['cluster'] = self.cluster_labels
        
        # Visualize with Seaborn
        cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
        grid = sns.pairplot(cluster_data, hue='cluster', palette='viridis')
        self._save_visualization(grid, "hierarchical_clustering")
        
    def _build_model(self, num_artists, num_tracks, feature_dim, hyperparams):
        """Build deep learning recommendation model"""
        # Input layers
        artist_input = Input(shape=(1,), name='artist_input')
        track_input = Input(shape=(1,), name='track_input')
        feature_input = Input(shape=(feature_dim,), name='feature_input')
        
        # Embedding layers
        artist_embedding = Embedding(
            input_dim=num_artists,
            output_dim=hyperparams['embedding_dim'],
            embeddings_regularizer=l2(hyperparams['l2_reg']),
            name='artist_embedding'
        )(artist_input)
        artist_embedding = Flatten()(artist_embedding)
        
        track_embedding = Embedding(
            input_dim=num_tracks,
            output_dim=hyperparams['embedding_dim'],
            embeddings_regularizer=l2(hyperparams['l2_reg']),
            name='track_embedding'
        )(track_input)
        track_embedding = Flatten()(track_embedding)
        
        # Concatenate all features
        merged = Concatenate()([artist_embedding, track_embedding, feature_input])
        
        # Add hidden layers
        x = Dense(hyperparams['hidden_units'][0], kernel_regularizer=l2(hyperparams['l2_reg']))(merged)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        for units in hyperparams['hidden_units'][1:]:
            x = Dense(units, kernel_regularizer=l2(hyperparams['l2_reg']))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(
            inputs=[artist_input, track_input, feature_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=hyperparams['learning_rate']),
            loss=BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
                tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
                AUC(name='auc'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )
        
        # Plot model architecture
        plot_model(model, to_file=f"{self.output_dir}/plots/model_architecture.png", show_shapes=True)
        
        return model
    
    def initialize_model(self, data, hyperparams=None):
        """Initialize deep learning model"""
        if hyperparams is None:
            hyperparams = {
                'embedding_dim': 32,
                'hidden_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'l2_reg': 0.001,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 50,
                'patience': 5
            }
        
        self.best_hyperparams = hyperparams
        
        # Get dimensions
        num_artists = len(self.label_encoders['artist_id'].classes_)
        num_tracks = len(self.label_encoders['track_id'].classes_)
        feature_cols = [col for col in data.columns if col not in ['artist_id', 'track_id', 'interaction']]
        feature_dim = len(feature_cols)
        
        self.logger.info(f"Initializing model with {num_artists} artists, {num_tracks} tracks, and {feature_dim} features")
        
        # Build model
        self.model = self._build_model(num_artists, num_tracks, feature_dim, hyperparams)
        self.model.summary(print_fn=self.logger.info)
        
    def train_model(self, data):
        """Train the deep learning model"""
        self.logger.info("\nStarting model training...")
        start_time = time.time()
        
        # Prepare data
        feature_cols = [col for col in data.columns if col not in ['artist_id', 'track_id', 'interaction']]
        X_artist = self.label_encoders['artist_id'].transform(data['artist_id'])
        X_track = self.label_encoders['track_id'].transform(data['track_id'])
        X_features = data[feature_cols].values
        y = data['interaction'].values
        
        # Split data
        X_artist_train, X_artist_val, X_track_train, X_track_val, X_features_train, X_features_val, y_train, y_val = train_test_split(
            X_artist, X_track, X_features, y, test_size=0.2, random_state=42
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.best_hyperparams['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Train model
        history = self.model.fit(
            x=[X_artist_train, X_track_train, X_features_train],
            y=y_train,
            validation_data=([X_artist_val, X_track_val, X_features_val], y_val),
            batch_size=self.best_hyperparams['batch_size'],
            epochs=self.best_hyperparams['epochs'],
            callbacks=callbacks,
            verbose=2
        )
        
        # Training time
        training_time = time.time() - start_time
        self.metrics_history['training_time'].append(training_time)
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Update metrics history
        for metric in history.history:
            if 'val_' in metric:
                self.metrics_history[f'val_{metric}'].append(np.min(history.history[metric]))
            else:
                self.metrics_history[f'train_{metric}'].append(np.min(history.history[metric]))
        
        # Plot training history
        self._plot_training_history(history)
        
        # Evaluate on full data
        self.logger.info("\nEvaluating on full dataset...")
        y_pred = self.model.predict([X_artist, X_track, X_features]).flatten()
        metrics = self._calculate_metrics(y, y_pred, y_proba=y_pred)
        
        # Plot feature importance
        self._plot_feature_importance()
        
        # Save model
        model_path = self._save_model()
        
        return model_path
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Generate recommendations for given artist"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.label_encoders['artist_id'].classes_:
            self.logger.info(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self._get_popular_tracks(top_n)
        
        # Get all tracks in same cluster
        artist_tracks = list(self.artist_track_map[artist_id])
        if len(artist_tracks) > 0:
            sample_track = artist_tracks[0]
            cluster = self.track_features[self.track_features['track_id'] == sample_track]['cluster'].values[0]
            candidate_tracks = self.track_features[self.track_features['cluster'] == cluster]['track_id'].unique()
        else:
            candidate_tracks = self.track_features['track_id'].unique()
        
        # Prepare data for prediction
        artist_ids = np.array([artist_id] * len(candidate_tracks))
        track_ids = np.array(candidate_tracks)
        
        # Filter out tracks not in our training data
        valid_tracks = [t for t in track_ids if t in self.label_encoders['track_id'].classes_]
        if not valid_tracks:
            return self._get_popular_tracks(top_n)
            
        artist_ids = np.array([artist_id] * len(valid_tracks))
        track_ids = np.array(valid_tracks)
        
        # Get features for valid tracks
        track_features = []
        for t in valid_tracks:
            features = self.track_features[self.track_features['track_id'] == t]
            feature_cols = [col for col in features.columns if col not in ['artist_id', 'track_id', 'interaction']]
            track_features.append(features[feature_cols].values[0])
        
        track_features = np.array(track_features)
        
        # Encode IDs
        encoded_artist = self.label_encoders['artist_id'].transform(artist_ids)
        encoded_track = self.label_encoders['track_id'].transform(track_ids)
        
        # Generate predictions
        scores = self.model.predict([encoded_artist, encoded_track, track_features]).flatten()
        
        # Prepare results
        recommendations = []
        for i, track_id in enumerate(valid_tracks):
            track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
            
            recommendations.append({
                'track_id': track_id,
                'score': scores[i],
                'danceability': track_data['danceability'],
                'energy': track_data['energy'],
                'valence': track_data['valence'],
                'cluster': cluster
            })
        
        # Sort and get top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def _get_popular_tracks(self, top_n=5):
        """Fallback method using cluster information"""
        if self.track_features is None:
            return []
        
        popular_tracks = self.track_features.sort_values(
            by=['danceability', 'energy'], 
            ascending=False
        ).head(top_n)
        
        return popular_tracks.to_dict('records')
    
    def _setup_logging(self, experiment_name):
        """Configure logging to file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.output_dir}/training.log"),
                logging.StreamHandler(),
                logging.FileHandler(f"{self.output_dir}/metrics.log")
            ]
        )
        self.logger = logging.getLogger(experiment_name)

# Example usage
if __name__ == "__main__":
    recommender = DeepLearningRecommender("dl_rec_experiment1")
    
    try:
        # Step 1: Preprocess data with hierarchical clustering
        data = recommender.preprocess_data("../../../data/ten_thousand.csv", sample_size=1000)
        
        # Step 2: Initialize model with default hyperparameters
        recommender.initialize_model(data)
        
        # Step 3: Train model
        model_path = recommender.train_model(data)
        recommender.logger.info(f"\nModel trained and saved to: {model_path}")
        
        # Step 4: Get recommendations
        example_artist = data['artist_id'].iloc[0]
        recommender.logger.info(f"\nRecommendations for artist {example_artist}:")
        recs = recommender.recommend_tracks(example_artist, top_n=5)
        
        recommender.logger.info("\nTop recommendations:")
        for i, rec in enumerate(recs, 1):
            recommender.logger.info(
                f"{i}. Track ID: {rec['track_id']} | Score: {rec['score']:.3f} | "
                f"Dance: {rec['danceability']:.2f} | Energy: {rec['energy']:.2f} | "
                f"Cluster: {rec['cluster']}"
            )
            
    except Exception as e:
        recommender.logger.error(f"Experiment failed: {str(e)}", exc_info=True)