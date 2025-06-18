from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster
import time
import seaborn as sns
import os
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
import logging
import tensorflow as tf
import random
import json
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Flatten, Dense, 
                                    Concatenate, Dropout, BatchNormalization,
                                    LeakyReLU)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import BinaryCrossentropy

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class DeepLearningRecommender:

    NUMERIC_FEATURES = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
    'tempo', 'time_signature', 'valence']

    OTHER_FEATURES = ['explicit', 'artist_id', 'release_id']

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
        def __init__(self, experiment_name="dl_rec"):
            self._setup_directories(experiment_name)
            self._setup_logging(experiment_name)
            
            # Initialize ALL instance attributes here
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
            
            # Initialize metrics_history with all expected keys
            self._init_metrics_history()
            
    def _init_metrics_history(self):
        """Initialize metrics tracking dictionary"""
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
            # Handle Seaborn PairGrid objects
            if hasattr(fig, 'savefig'):
                fig.savefig(path, bbox_inches='tight', dpi=300)
            elif hasattr(fig, 'fig'):  # For Seaborn PairGrid
                fig.fig.savefig(path, bbox_inches='tight', dpi=300)
            else:
                plt.savefig(path, bbox_inches='tight', dpi=300)
                
            # Only close if it's a matplotlib Figure
            if isinstance(fig, plt.Figure):
                plt.close(fig)
            elif hasattr(fig, 'fig'):  # Close PairGrid's figure
                plt.close(fig.fig)
        except Exception as e:
            self.logger.error(f"Error saving visualization {name}: {str(e)}")

    def plot_feature_distributions(self, data):
        """
        Rysuje histogramy rozkładów cech z self.feature_cols
        """
        if data is None:
            self.logger.error("plot_feature_distributions: data is None!")
            return
        feat_cols = self.feature_cols
        if not feat_cols:
            self.logger.error("plot_feature_distributions: feature_cols is empty!")
            return
        fig, axes = plt.subplots(1, min(3, len(feat_cols)), figsize=(15, 4))
        if len(feat_cols) == 1:
            axes = [axes]
        for ax, feat in zip(axes, feat_cols[:3]):
            if feat not in data.columns:
                self.logger.warning(f"plot_feature_distributions: {feat} not found in data.columns")
                continue
            sns.histplot(data[feat], ax=ax, kde=True)
            ax.set_title(f'Distribution of {feat}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/feature_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close(fig)
        
    def _save_model(self):
        """Save model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.output_dir}/models/model_{timestamp}.h5"
        metadata_path = f"{self.output_dir}/models/metadata_{timestamp}.joblib"
        
        # Save Keras model
        self.model.save(model_path)
        print("Input data")
        print([inp.shape for inp in self.model.inputs])
        print("Summary")
        self.model.summary()

        
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

    def plot_elbow_and_silhouette(self, data, max_clusters=10):
        features = data[self.feature_cols].fillna(0).values

        inertias = []
        silhouettes = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(features, labels))
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(range(2, max_clusters + 1), inertias, marker='o')
        ax[0].set_title('Elbow curve (inertia)')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('Inertia')

        ax[1].plot(range(2, max_clusters + 1), silhouettes, marker='o', color='green')
        ax[1].set_title('Silhouette score')
        ax[1].set_xlabel('Number of clusters')
        ax[1].set_ylabel('Score')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/elbow_silhouette_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close(fig)
        print('Elbow/silhouette plot saved!')
    
    def _plot_training_history(self, history):
        """Plot training history metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # RMSE - Updated to use 'rmse' instead of 'root_mean_squared_error'
        axes[0, 1].plot(history.history['rmse'], label='Train RMSE')
        axes[0, 1].plot(history.history['val_rmse'], label='Validation RMSE')
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
            
    def preprocess_data(
        self, 
        file_path, 
        sample_size=None, 
        n_clusters=6, 
        elbow_max_clusters=12
    ):
        """
        Preprocessing: Wczytanie, czyszczenie, feat.eng, kodowanie explict, kodowanie id, 
        generowanie all_pairs, clustering, zapisywanie wykresów jakości klasteryzacji, 
        przygotowanie gotowych tablic do trenowania i rekomendacji.
        """
        ### 1. Wczytanie danych
        self.logger.info("Loading CSV data...")
        data = pd.read_csv(file_path)
        if sample_size:
            data = data.sample(sample_size, random_state=42)

        # 2. Zamień explicit na 0/1
        if 'explicit' in data.columns:
            data['explicit'] = data['explicit'].astype(int)

        # 3. Kodowanie artist_id i release_id (przyjmujemy JEDEN artysta, JEDEN album per track)
        for col in ['artist_id', 'release_id']:
            if col in data.columns:
                # Jeśli lista w stringu: zamień na pierwszy element listy
                data[col] = data[col].apply(
                    lambda x: eval(x)[0] if (isinstance(x, str) and x.startswith('[')) else x
                )
                le = LabelEncoder()
                data[col + '_le'] = le.fit_transform(data[col])
                self.label_encoders[col] = le

        # 4. Zdefiniuj CECHY NAUCZANIA
        NUMERIC_FEATURES = [
            'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
            'tempo', 'time_signature', 'valence'
        ]
        ID_FEATURES = ['explicit', 'artist_id_le', 'release_id_le']
        ALL_FEATURES = NUMERIC_FEATURES + ID_FEATURES
        self.feature_cols = [col for col in ALL_FEATURES if col in data.columns]
        self.logger.info(f"feature_cols: {self.feature_cols}")

        # 5. Sklejenie podstawowej tablicy track_features
        self.track_features = data[['track_id'] + self.feature_cols].drop_duplicates('track_id')
        self.logger.info(f"track_features shape: {self.track_features.shape}")

        # 6. Generuj positive pairs (wszystkie unikalne tracki/artist relacje)
        positive_pairs = data[['artist_id', 'track_id']].drop_duplicates().copy()
        positive_pairs['interaction'] = 1

        # 7. Negative sampling (po prostu losowo dla par artist-track, które nie istnieją)
        all_artists = data['artist_id'].unique()
        all_tracks = data['track_id'].unique()
        negative_samples = []

        artist_track_positive = set(zip(positive_pairs['artist_id'], positive_pairs['track_id']))
        np.random.seed(42)
        for artist in all_artists:
            pos_tracks = set(data[data['artist_id'] == artist]['track_id'])
            neg_tracks = list(set(all_tracks) - pos_tracks)
            if not neg_tracks or not pos_tracks:
                continue
            sampled_negatives = np.random.choice(
                neg_tracks, size=min(len(pos_tracks), len(neg_tracks)), replace=False
            )
            for track in sampled_negatives:
                negative_samples.append({'artist_id': artist, 'track_id': track, 'interaction': 0})

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        self.logger.info(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")

        # 8. Uzupełnij cechy trackowe do par (join po track_id)
        all_pairs = all_pairs.merge(self.track_features, how='left', on='track_id')
        self.logger.info(f"all_pairs shape after merge: {all_pairs.shape}")

        # 9. Uzupełnij encoder dla artistów jeśli używasz w modelu
        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])

        # 10. Scaling numerycznych cech
        scaler = MinMaxScaler()
        all_pairs[self.feature_cols] = scaler.fit_transform(all_pairs[self.feature_cols].fillna(0))
        self.scaler = scaler

        # 11. ELBOW & SILHOUETTE (zapisywanie wykresu) — przed klasteryzacją!
        self.plot_elbow_and_silhouette(all_pairs, max_clusters=elbow_max_clusters)

        # 12. HIERARCHICZNE KLASTRY ZAPISANE JAKO 'cluster'
        all_pairs = self._apply_hierarchical_clustering(all_pairs, n_clusters)
        self.logger.info(f"Applied hierarchical clustering with n_clusters={n_clusters}")

        # 13. Wykresy rozkładów cech
        self.plot_feature_distributions(all_pairs)

        # 14. Uaktualnij też self.track_features żeby zawierała 'cluster'
        self.track_features = all_pairs[['track_id'] + self.feature_cols + ['cluster']].drop_duplicates('track_id')

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
    
    def log_model_stats(self):
        """Log and save general model statistics and structure."""
        try:
            stats = {}
            stats['num_trainable_params'] = self.model.count_params()
            stats['model_size_MB'] = self.model.count_params() * 4 / (1024 ** 2)  # float32 = 4B
            # Save model summary to text file
            summary_lines = []
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            summary_path = f"{self.output_dir}/models/model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_path, "w") as f:
                f.write('\n'.join(summary_lines))
            
            self.logger.info(f"Model trainable params: {stats['num_trainable_params']}")
            self.logger.info(f"Model size (float32, MB): {stats['model_size_MB']:.2f}")
            self.logger.info(f"Saved model summary: {summary_path}")

            # Dump to JSON if needed
            import json
            stats_path = f"{self.output_dir}/models/model_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            
            return stats
        except Exception as e:
            self.logger.error(f"Could not log model stats: {e}")
            return {}
    
    def _apply_hierarchical_clustering(self, data, n_clusters = 6):
        cluster_features = data[self.feature_cols].fillna(0).values
        Z = linkage(cluster_features, method='ward')
        self.cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        data['cluster'] = self.cluster_labels
        # Visualize with Seaborn
        try:
            cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
            # KLUCZOWA ZMIANA: s=15, alpha=0.3 (mniej nachodzą!):
            grid = sns.pairplot(
                cluster_data,
                hue='cluster',
                palette='viridis',
                plot_kws={'s': 15, 'alpha': 0.3},   # <-- OTO TE PARAMETRY
                diag_kws={'alpha': 0.3}
            )
            self._save_visualization(grid.fig, "hierarchical_clustering")
        except Exception as e:
            self.logger.error(f"Could not visualize clusters: {str(e)}")
        return data
        
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
# In your _build_model or initialize_model method:
        model.compile(
            optimizer=Adam(learning_rate=hyperparams['learning_rate']),
            loss=BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),  # Changed from 'root_mean_squared_error'
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
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
        self.feature_cols = [col for col in data.columns if col not in ['artist_id', 'track_id', 'interaction']]
        feature_dim = len(self.feature_cols)
        
        self.logger.info(f"Initializing model with {num_artists} artists, {num_tracks} tracks, and {feature_dim} features")
        
        # Build model
        self.model = self._build_model(num_artists, num_tracks, feature_dim, hyperparams)
        self.model.summary(print_fn=self.logger.info)
        
    def train_model(self, data):
        """Train the deep learning model"""
            # Defensive check - ensure metrics_history exists
        if not hasattr(self, 'metrics_history'):
            self.logger.warning("metrics_history not found - initializing")
            self._init_metrics_history()

        self.logger.info("\nStarting model training...")
        start_time = time.time()
        
        # Prepare data
        self.feature_cols = [col for col in data.columns if col not in ['artist_id', 'track_id', 'interaction']]
        X_artist = self.label_encoders['artist_id'].transform(data['artist_id'])
        X_track = self.label_encoders['track_id'].transform(data['track_id'])
        X_features = data[self.feature_cols].values
        y = data['interaction'].values
        
        X_artist = np.array(X_artist)
        X_track = np.array(X_track)
        X_features = np.array(X_features)
        y = np.array(y)

        print(type(X_artist), type(X_track), type(X_features), type(y))
        print(train_test_split)


        split = train_test_split(X_artist, X_track, X_features, y, test_size=0.2, random_state=42)
        print(f"Split produced {len(split)} outputs, lengths: {[len(x) for x in split]}")
        (X_artist_train, X_artist_val, 
        X_track_train, X_track_val,
        X_features_train, X_features_val,
        y_train, y_val) = split
        
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
        
        # Rest of your training code...
        try:
        # Your existing training code
            training_time = time.time() - start_time
            
            # Safe metric recording
            if hasattr(self, 'metrics_history'):
                self.metrics_history['training_time'].append(training_time)
            else:
                self.logger.error("metrics_history still not available!")
                
            # Process history metrics
            if hasattr(self, 'metrics_history') and history:
        
                # Update metrics history
                # With this more robust version:
                for metric in history.history:
                    # Handle both cases where metric might be 'mae' or 'mean_absolute_error'
                    metric_name = metric.replace('val_', '').replace('train_', '')
                    
                    if metric.startswith('val_'):
                        target_dict = self.metrics_history
                        prefix = 'val_'
                    else:
                        target_dict = self.metrics_history
                        prefix = 'train_'
                    
                    # Standardize metric names
                    if metric_name == 'mean_absolute_error':
                        metric_key = f'{prefix}mae'
                    elif metric_name == 'root_mean_squared_error':
                        metric_key = f'{prefix}rmse'
                    else:
                        metric_key = f'{prefix}{metric_name}'
                    
                    # Create key if it doesn't exist
                    if metric_key not in target_dict:
                        target_dict[metric_key] = []
                    
                    target_dict[metric_key].append(np.min(history.history[metric]))
                
                # Plot training history
                self._plot_training_history(history)
                
                # Evaluate on full data
                self.logger.info("\nEvaluating on full dataset...")
                y_pred = self.model.predict([X_artist, X_track, X_features]).flatten()
                metrics = self._calculate_metrics(y, y_pred, y_proba=y_pred)
                
                # Plot feature importance
                self._plot_feature_importance()
                self.plot_shap(data)
                
                # Save model
                model_path = self._save_model()
                print("FEATURE COLS used during training:", self.feature_cols)
                print("train_model X_features shape:", X_features.shape)
                
                return model_path
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Generate recommendations for given artist"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.label_encoders['artist_id'].classes_:
            self.logger.info(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self._get_popular_tracks(top_n)
        
        # Get all tracks in same cluster
        artist_tracks = list(self.artist_track_map.get(artist_id, []))
        if not artist_tracks:
            return self._get_popular_tracks(top_n)

        # Initialize variables with default values
        valid_tracks = []
        cluster = None
        
        # Get cluster for this artist (using first track as representative)
        try:
            sample_track = artist_tracks[0]
            track_data = self.track_features[self.track_features['track_id'] == sample_track]
            
            if track_data.empty or 'cluster' not in track_data.columns:
                self.logger.warning("No cluster information available - using all tracks")
                candidate_tracks = self.track_features['track_id'].unique()
            else:
                cluster = track_data['cluster'].values[0]
                candidate_tracks = self.track_features[self.track_features['cluster'] == cluster]['track_id'].unique()
            
            # Filter out tracks not in our training data
            for t in valid_tracks:
                row = self.track_features[self.track_features['track_id'] == t]
                track_features.append(row[self.feature_cols].values[0])  # <-- zawsze te kolumny!            
        except Exception as e:
            self.logger.error(f"Error getting cluster: {str(e)}")
            candidate_tracks = self.track_features['track_id'].unique()
            valid_tracks = [t for t in candidate_tracks if t in self.label_encoders['track_id'].classes_]

        if not valid_tracks:
            return self._get_popular_tracks(top_n)
        
        # Prepare data for prediction
        artist_ids = np.array([artist_id] * len(valid_tracks))
        track_ids = np.array(valid_tracks)
        
        # Get features for valid tracks
        track_features = []
        for t in valid_tracks:
            features = self.track_features[self.track_features['track_id'] == t]
            self.feature_cols = [col for col in features.columns if col not in ['artist_id', 'track_id', 'interaction']]
            track_features.append(features[self.feature_cols].values[0])
        
        track_features = np.array(track_features)
        
        # Encode IDs
        encoded_artist = self.label_encoders['artist_id'].transform(artist_ids)
        encoded_track = self.label_encoders['track_id'].transform(track_ids)
        
        # Generate predictions
        print("FEATURE COLS at recommend:", self.feature_cols)
        print("self.track_features.columns at recommend:", self.track_features.columns)
        print("FEATURE SHAPE:", track_features.shape)
        print("EXPECTED FEATURE COLS:", self.feature_cols)
        print("Track features colnames at recommend:", features.columns)
        print("X for model shapes:", encoded_artist.shape, encoded_track.shape, track_features.shape)
        print("FIRST ROW FEATURE VECTOR SIZE:", track_features[0].shape)
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
                'cluster': cluster if cluster is not None else -1  # Default value if no cluster
            })
        
        # Sort and get top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def recommend_similar_tracks(self, base_track_id, top_n=5):
        # znajdź features bazowego utworu
        origin = self.track_features[self.track_features['track_id'] == base_track_id]
        if origin.empty:
            self.logger.warning('Base track not found!')
            return []
        base_vec = origin[self.feature_cols].iloc[0].values.astype(float)

        # wyklucz bazowy utwór z puli
        candidates = self.track_features[self.track_features['track_id'] != base_track_id]
        candidate_vecs = candidates[self.feature_cols].values.astype(float)

        # podobieństwo kosinusowe do base_vec (wszystkie kawałki)
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([base_vec], candidate_vecs)[0]
        # wybierz top_n najbardziej podobnych (INDEKSY)
        top_idx = np.argsort(scores)[::-1][:top_n]

        # pobierz track_id i score dla top wyników
        recs = []
        for idx in top_idx:
            tr = candidates.iloc[idx]
            recs.append({
                'track_id': tr['track_id'],
                'score': scores[idx],
                'track_name': tr.get('track_name', ''),
                'artist_id': tr.get('artist_id', ''),
                'danceability': tr.get('danceability', np.nan),
                'energy': tr.get('energy', np.nan),
                'valence': tr.get('valence', np.nan),
                'cluster': tr.get('cluster', '-')
            })
        return recs
    
    def compute_similarity_metrics(self, base_track_id, recs, metric='cosine'):
        """
        Porównuje każdy utwór z recs do bazowego i zwraca statystyki podobieństw
        :param base_track_id: track_id utworu bazowego
        :param recs: lista rekomendacji (dictów)
        :param metric: 'cosine' lub 'euclidean'
        :return: dict ze statystykami dla topn rekomendacji
        """
        # Wektor bazowy
        base_vec = self.track_features[self.track_features['track_id'] == base_track_id][self.feature_cols].values
        if base_vec.shape[0] == 0:
            print("Base track not found!")
            return {}
        rec_vecs = []
        for r in recs:
            vec = self.track_features[self.track_features['track_id'] == r['track_id']][self.feature_cols].values
            if vec.shape[0] > 0:
                rec_vecs.append(vec[0])

        rec_vecs = np.array(rec_vecs)
        if rec_vecs.shape[0] == 0:
            print("No vectors for recommended tracks!")
            return {}

        # Oblicz metryki
        if metric == 'cosine':
            sims = cosine_similarity(base_vec, rec_vecs)[0]
            # Blisko 1 = bardzo podobne
            stat = {
                'mean_cosine_similarity': np.mean(sims),
                'median_cosine_similarity': np.median(sims),
                'min_cosine_similarity': np.min(sims),
                'max_cosine_similarity': np.max(sims)
            }
        else:
            dists = euclidean_distances(base_vec, rec_vecs)[0]
            stat = {
                'mean_euclidean_distance': np.mean(dists),
                'median_euclidean_distance': np.median(dists),
                'min_euclidean_distance': np.min(dists),
                'max_euclidean_distance': np.max(dists)
            }
        return stat
    
    def _get_popular_tracks(self, top_n=5):
        """Fallback method using cluster information"""
        if self.track_features is None:
            return []
        
        popular_tracks = self.track_features.sort_values(
            by=['danceability', 'energy'], 
            ascending=False
        ).head(top_n)
        
        # Dodaj score=None
        result = []
        for _, row in popular_tracks.iterrows():
            d = row.to_dict()
            d['score'] = None  # lub np. d['score'] = float('nan')
            result.append(d)
        return result
    
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

    def compute_diversity(self, recs):
        """Compute average pairwise distance in feature space among recommended tracks - im wyżej tym większa różnorodność."""
        try:
            feature_matrix = []
            for rec in recs:
                row = self.track_features[self.track_features['track_id'] == rec['track_id']]
                if not row.empty:
                    feature_matrix.append(row[self.feature_cols].values[0])
            X = np.array(feature_matrix)
            if X.shape[0] < 2:
                return np.nan
            from scipy.spatial.distance import pdist
            avg_dist = np.mean(pdist(X, metric='euclidean'))
            self.logger.info(f"Diversity (avg. Euclidean distance): {avg_dist:.4f}")
            return avg_dist
        except Exception as e:
            self.logger.error(f"Failed to compute diversity: {e}")
            return np.nan
    
    def plot_shap(self, data, nsamples=200):
        try:
            # Bierzemy losową próbkę, żeby nie zamulić!
            sample = data.sample(nsamples, random_state=1)
            X_artist = self.label_encoders['artist_id'].transform(sample['artist_id'])
            X_track = self.label_encoders['track_id'].transform(sample['track_id'])
            X_features = sample[self.feature_cols].values

            # SHAP potrzebuje wejścia jako 2D array (tu wszystkie x razem, potem się wyciąga)
            def keras_input(X):
                # Helper do explainerów, konwertuje np. array do listy wejść do modelu
                return [X_artist, X_track, X_features]

            explainer = shap.KernelExplainer(
                (lambda x: self.model.predict([x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1), x[:, 2:]])),
                np.c_[X_artist, X_track, X_features]
            )
            shap_values = explainer.shap_values(np.c_[X_artist, X_track, X_features], nsamples=100)
            # Nazwy cech: embeddingi, a potem cechy oryginalne
            feature_names = (
                ['artist_id', 'track_id'] + list(self.feature_cols)
            )
            fig = shap.summary_plot(
                shap_values, np.c_[X_artist, X_track, X_features],
                feature_names=feature_names, 
                plot_type="bar",
                show=False
            )
            plt.title("SHAP feature importances: NN inputs")
            self._save_visualization(plt.gcf(), "shap_summary")
        except Exception as e:
            self.logger.warning(f"Could not plot SHAP: {e}")
    
    def compute_coverage(self, all_recommendations):
        """Procent unikalnych tracków poleconych spośród wszystkich w zbiorze."""
        unique_tracks = set()
        for recs in all_recommendations:
            for rec in recs:
                unique_tracks.add(rec['track_id'])
        coverage = len(unique_tracks) / len(self.track_features)
        self.logger.info(f"Coverage: {coverage:.3%}")
        return coverage
    
    def jaccard_set_similarity(self, recs_model1, recs_model2):
        set1 = set(rec['track_id'] for rec in recs_model1)
        set2 = set(rec['track_id'] for rec in recs_model2)
        if len(set1 | set2) == 0:
            return np.nan
        return len(set1 & set2) / len(set1 | set2)
    
    def measure_inference_time(self, artist_id, n_trials=20):
        import time
        times = []
        for _ in range(n_trials):
            t0 = time.time()
            try:
                _ = self.recommend_tracks(artist_id)
            except Exception:
                continue
            t1 = time.time()
            times.append(t1 - t0)
        mean_time = np.mean(times)
        self.logger.info(f"Average inference time ({n_trials} trials): {mean_time:.4f} s")
        return mean_time
    
    def compute_stability(self, list_of_metric_dicts, metric='val_rmse'):
        vals = [metrics[metric] for metrics in list_of_metric_dicts if metric in metrics]
        stdev = np.std(vals)
        self.logger.info(f"Stability for {metric}: std={stdev:.4f}, mean={np.mean(vals):.4f}")
        return stdev

# Example usage
if __name__ == "__main__":
    DATA_PATH = "../../../data/ten_thousand.csv"
    EXPERIMENT_NAME = "dl_rec_experiment1"
    N_TEST_ARTISTS = 10    # liczba artystów do testowania rekomendacji

    np.random.seed(42)
    random.seed(42)

    recommender = DeepLearningRecommender(EXPERIMENT_NAME)
    try:
        # Step 1: Preprocess data with hierarchical clustering
        data = recommender.preprocess_data(DATA_PATH)

        # Step 2: Initialize and train model
        recommender.initialize_model(data)
        model_path = recommender.train_model(data)
        recommender.logger.info(f"\nModel trained and saved to: {model_path}")
        model_stats = recommender.log_model_stats()

        # Step 3: Ewaluacja na kilku artystach
        test_artists = data['artist_id'].drop_duplicates().sample(min(N_TEST_ARTISTS, data['artist_id'].nunique()), random_state=42).tolist()
        all_recs = []
        diversity_scores = []
        sim_metrics_scores = []

        for artist_id in test_artists:
            recommender.logger.info(f"\nRecommendations for artist {artist_id}:")
            recs = recommender.recommend_tracks(artist_id, top_n=5)
            all_recs.append(recs)
            for i, rec in enumerate(recs, 1):
                recommender.logger.info(
                    f"{i}. Track ID: {rec['track_id']} "
                    f"Dance: {rec['danceability']:.2f} | Energy: {rec['energy']:.2f} | "
                    f"Cluster: {rec['cluster']}"
                )

            # Diversity
            diversity = recommender.compute_diversity(recs)
            diversity_scores.append(diversity)

            # Similarity metrics względem top 1 tracku z rekomendacji
            if recs:
                base_track = recs[0]['track_id']
                sim_metrics = recommender.compute_similarity_metrics(base_track, recs, metric='cosine')
                sim_metrics_scores.append(sim_metrics)
                recommender.logger.info(f"Artist {artist_id} - Similarity metrics for top recs: {sim_metrics}")

        # Step 4: Coverage
        coverage = recommender.compute_coverage(all_recs)
        recommender.logger.info(f"Model coverage: {coverage:.3%}")

        # Step 5: Rysowanie wykresów boxplot diversity/similarity
        plt.figure(figsize=(8, 4))
        sns.boxplot(diversity_scores)
        plt.title("Rekomendacje: różnorodność (diversity, dystans euklidesowy)")
        plt.ylabel("Dystans euklidesowy")
        plt.tight_layout()
        plot_path_div = f"{recommender.output_dir}/plots/diversity_boxplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path_div)
        plt.close()

        mean_cosine_list = [d.get('mean_cosine_similarity', np.nan) for d in sim_metrics_scores]
        plt.figure(figsize=(8, 4))
        sns.boxplot(mean_cosine_list)
        plt.title("Rekomendacje: średnie podobieństwo kosinusowe do bazowego utworu")
        plt.ylabel("Mean cosine similarity")
        plt.tight_layout()
        plot_path_sim = f"{recommender.output_dir}/plots/cosine_boxplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path_sim)
        plt.close()

        # Step 6: Pomiar czasu odpowiedzi dla inference
        mean_infer_time = recommender.measure_inference_time(test_artists[0], n_trials=10)

        # Step 7: Raport
        report = {
            "model_stats": model_stats,
            "coverage": coverage,
            "diversity_median": float(np.median([d for d in diversity_scores if not np.isnan(d)])),
            "diversity_mean": float(np.mean([d for d in diversity_scores if not np.isnan(d)])),
            "mean_cosine_similarity_median": float(np.median([m for m in mean_cosine_list if not np.isnan(m)])),
            "mean_cosine_similarity_mean": float(np.mean([m for m in mean_cosine_list if not np.isnan(m)])),
            "mean_inference_time_s": mean_infer_time
        }
        with open(f"{recommender.output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2)
        recommender.logger.info(f"Summary report: {report}")

        print("FINISHED!")

    except Exception as e:
        recommender.logger.error(f"Experiment failed: {str(e)}", exc_info=True)