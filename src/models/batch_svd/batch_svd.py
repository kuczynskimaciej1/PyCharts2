from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, log_loss)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from scipy import stats
import psutil
import time
import seaborn as sns
import pingouin as pg
import os
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

class SVDBasedRecommender:
    def __init__(self, experiment_name="svd_rec"):
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
        
        # Metrics tracking
        self.metrics_history = {
            'rmse': [],
            'mae': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'pr_auc': [],
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
        model_path = f"{self.output_dir}/models/model_{timestamp}.joblib"
        
        save_data = {
            'model': self.model,
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
        
        joblib.dump(save_data, model_path)
        self.logger.info(f"Saved model and metadata: {model_path}")
        return model_path
        
    def _load_model(self, model_path):
        """Load saved model and metadata"""
        data = joblib.load(model_path)
        self.model = data['model']
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
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary)
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
    
    def _calculate_additional_metrics(self, y_true, y_pred, y_proba):
        """Calculate additional metrics"""
        metrics = {}
        y_true_binary = (y_true >= 0.5).astype(int)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # Log Loss
        if len(np.unique(y_true_binary)) > 1:
            metrics['log_loss'] = log_loss(y_true_binary, y_proba)
        
        # Cosine Similarity
        pred_dist = np.bincount(y_pred_binary) / len(y_pred_binary)
        true_dist = np.bincount(y_true_binary) / len(y_true_binary)
        max_classes = max(len(pred_dist), len(true_dist))
        pred_dist.resize(max_classes)
        true_dist.resize(max_classes)
        metrics['cosine_sim'] = 1 - cosine(pred_dist, true_dist)
        
        return metrics
    
    def _plot_metrics_history(self):
        """Plot training metrics over time"""
        if not self.metrics_history:
            return
            
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # Error metrics
        ax1 = fig.add_subplot(gs[0, 0])
        for metric in ['rmse', 'mae']:
            if metric in self.metrics_history and self.metrics_history[metric]:
                ax1.plot(self.metrics_history[metric], label=metric)
        ax1.set_title('Error Metrics')
        ax1.legend()
        
        # Classification metrics
        ax2 = fig.add_subplot(gs[0, 1])
        for metric in ['precision', 'recall', 'f1']:
            if metric in self.metrics_history and self.metrics_history[metric]:
                ax2.plot(self.metrics_history[metric], label=metric)
        ax2.set_title('Classification Metrics')
        ax2.legend()
        
        # Probability metrics
        ax3 = fig.add_subplot(gs[1, 0])
        for metric in ['roc_auc', 'pr_auc']:
            if metric in self.metrics_history and self.metrics_history[metric]:
                ax3.plot(self.metrics_history[metric], label=metric)
        ax3.set_title('Probability Metrics')
        ax3.legend()
        
        # Training time
        ax4 = fig.add_subplot(gs[1, 1])
        if 'training_time' in self.metrics_history:
            ax4.plot(self.metrics_history['training_time'], label='Training Time (s)')
        ax4.set_title('Training Time')
        ax4.legend()
        
        plt.tight_layout()
        self._save_visualization(fig, "full_metrics_history")
        
    def _plot_feature_importance(self):
        """Plot latent feature importance"""
        try:
            if hasattr(self.model, 'pu') and hasattr(self.model, 'qi'):
                # Get user and item factors
                user_factors = self.model.pu
                item_factors = self.model.qi
                
                # Calculate feature importance as variance explained
                user_importance = np.var(user_factors, axis=0)
                item_importance = np.var(item_factors, axis=0)
                
                # Plot user factors importance
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=user_importance, y=[f"UF_{i}" for i in range(len(user_importance))], ax=ax1)
                ax1.set_title('User Latent Factors Importance')
                self._save_visualization(fig1, "user_factors_importance")
                
                # Plot item factors importance
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=item_importance, y=[f"IF_{i}" for i in range(len(item_importance))], ax=ax2)
                ax2.set_title('Item Latent Factors Importance')
                self._save_visualization(fig2, "item_factors_importance")
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
        self.logger.info("Feature strategy chosen as content")
        cols_to_keep = feature_strategy + ['artist_id', 'track_id', 'genre', 'interaction']
        self.logger.info("Columns to keep chosen")
        
        # Only keep columns that actually exist in the data
        cols_to_keep = [col for col in cols_to_keep if col in data.columns]
        data = data[cols_to_keep]
        self.logger.info("Columns to keep filtered")

        # Create positive pairs
        positive_pairs = data[['artist_id', 'track_id', 'genre']].drop_duplicates()
        self.logger.info("Duplicates deleted")
        positive_pairs['interaction'] = 1
        
        # Create artist-track-genre map
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])
            self.genre_map[row['artist_id']] = row['genre']
        self.logger.info("Artist-track-genre map created")
        
        # Generate negative samples
        negative_samples = []
        all_tracks = set(data['track_id'].unique())
        self.logger.info("Generating negative samples...")

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
        
    def initialize_model(self, hyperparams=None):
        """Initialize SVD model with optional hyperparameters"""
        if hyperparams:
            self.model = SVD(
                n_factors=hyperparams['n_factors'],
                n_epochs=hyperparams['n_epochs'],
                lr_all=hyperparams['lr_all'],
                reg_all=hyperparams['reg_all'],
                random_state=42
            )
            self.best_hyperparams = hyperparams
        else:
            self.model = SVD(
                n_factors=50,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                random_state=42
            )
    
    def tune_hyperparameters(self, data, max_evals=30):
        """Hyperparameter tuning with GridSearchCV"""
        self.logger.info("\nStarting hyperparameter tuning...")
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(0, 1))
        surprise_data = Dataset.load_from_df(data[['artist_id', 'track_id', 'interaction']], reader)
        
        param_grid = {
            'n_factors': [20, 50, 100],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.01, 0.02, 0.04]
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
        gs.fit(surprise_data)
        
        # Get best params
        best_params = {
            'n_factors': gs.best_params['rmse']['n_factors'],
            'n_epochs': gs.best_params['rmse']['n_epochs'],
            'lr_all': gs.best_params['rmse']['lr_all'],
            'reg_all': gs.best_params['rmse']['reg_all']
        }
        
        self.logger.info(f"Best RMSE: {gs.best_score['rmse']}")
        self.logger.info(f"Best MAE: {gs.best_score['mae']}")
        self.logger.info(f"Best params: {best_params}")
        
        # Initialize model with best params
        self.initialize_model(best_params)
        
        return best_params
    
    def train_model(self, data):
        """Train SVD model with full dataset"""
        self.logger.info("\nStarting model training...")
        start_time = time.time()
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(0, 1))
        surprise_data = Dataset.load_from_df(data[['artist_id', 'track_id', 'interaction']], reader)
        
        # Build full trainset
        trainset = surprise_data.build_full_trainset()
        
        # Train model
        self.model.fit(trainset)
        
        # Training time
        training_time = time.time() - start_time
        self.metrics_history['training_time'].append(training_time)
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Cross-validation
        self.logger.info("\nRunning cross-validation...")
        cv_results = cross_validate(
            self.model, 
            surprise_data, 
            measures=['rmse', 'mae'], 
            cv=5, 
            n_jobs=-1,
            verbose=True
        )
        
        # Log CV results
        self.logger.info("\nCross-validation results:")
        for metric in ['rmse', 'mae']:
            self.logger.info(f"{metric}: {np.mean(cv_results[f'test_{metric}']):.4f} (±{np.std(cv_results[f'test_{metric}']):.4f})")
            self.metrics_history[metric].append(np.mean(cv_results[f'test_{metric}']))
        
        # Generate predictions for all data
        testset = trainset.build_anti_testset()
        predictions = self.model.test(testset)
        
        # Extract true and predicted values
        y_true = np.array([pred.r_ui for pred in predictions])
        y_pred = np.array([pred.est for pred in predictions])
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba=y_pred)
        additional_metrics = self._calculate_additional_metrics(y_true, y_pred, y_proba=y_pred)
        
        # Update metrics history
        for metric, value in {**metrics, **additional_metrics}.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
        
        # Plot metrics and feature importance
        self._plot_metrics_history()
        self._plot_feature_importance()
        
        # Save model
        model_path = self._save_model()
        
        return model_path
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Generate recommendations for given artist"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.artist_id_map:
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
        
        # Generate predictions
        predictions = []
        for track_id in candidate_tracks:
            if track_id in self.track_id_map:
                pred = self.model.predict(artist_id, track_id)
                track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
                
                predictions.append({
                    'track_id': track_id,
                    'score': pred.est,
                    'danceability': track_data['danceability'],
                    'energy': track_data['energy'],
                    'valence': track_data['valence'],
                    'cluster': cluster
                })
        
        # Sort and get top recommendations
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:top_n]
    
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
    recommender = SVDBasedRecommender("svd_rec_experiment1")
    
    try:
        # Step 1: Preprocess data with hierarchical clustering
        data = recommender.preprocess_data("../../../data/ten_thousand.csv", sample_size=1000)
        
        # Step 2: Hyperparameter tuning
        best_params = recommender.tune_hyperparameters(data, max_evals=30)
        recommender.logger.info(f"\nBest hyperparameters: {best_params}")
        
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