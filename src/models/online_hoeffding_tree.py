import pandas as pd
import numpy as np
from collections import defaultdict
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import DataStream
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import pingouin as pg
from functools import partial
import time
import logging
import os
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedMusicRecommender:
    def __init__(self, experiment_name="music_rec"):
        self._setup_logging(experiment_name)
        self._setup_directories(experiment_name)
        
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
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'pr_auc': [],
            'batch_loss': []
        }
        
    def _setup_logging(self, experiment_name):
        """Configure logging to file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{experiment_name}_log.txt"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
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
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        self.logger.info(f"Saved visualization: {path}")
        
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
        
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive performance metrics"""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'pr_auc': average_precision_score(y_true, y_proba)
            })
        
        # Log classification report
        self.logger.info("\nClassification Report:\n" + classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        self._save_visualization(fig, "confusion_matrix")
        
        return metrics
    
    def _plot_metrics_history(self):
        """Plot training metrics over time"""
        if not self.metrics_history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(self.metrics_history.items()):
            if not values:
                continue
                
            ax = axes[i]
            ax.plot(values, marker='o')
            ax.set_title(metric.upper())
            ax.set_xlabel('Batch/Iteration')
            ax.set_ylabel('Score')
            ax.grid(True)
            
        plt.tight_layout()
        self._save_visualization(fig, "training_metrics")
        
    def _plot_feature_importance(self):
        """Plot feature importance if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                features = self.track_features.columns.tolist()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=importances, y=features, ax=ax)
                ax.set_title('Feature Importances')
                self._save_visualization(fig, "feature_importance")
        except Exception as e:
            self.logger.warning(f"Could not plot feature importance: {str(e)}")
            
    def _analyze_regularization(self):
        """Analyze regularization effects"""
        if not self.best_hyperparams:
            return
            
        self.logger.info("\nRegularization Analysis:")
        self.logger.info(f"Grace Period: {self.best_hyperparams['grace_period']} (Controls tree growth)")
        self.logger.info(f"Split Confidence: {self.best_hyperparams['split_confidence']:.2e} (Prevents over-splitting)")
        self.logger.info(f"Tie Threshold: {self.best_hyperparams['tie_threshold']:.3f} (Controls node splitting)")
    
    def preprocess_data(self, file_path, sample_size=10000):
        """Enhanced preprocessing with genre clustering and feature engineering"""
        self.logger.info("Loading and preprocessing data with hierarchical clustering...")
        data = pd.read_csv(file_path)
        
        # Create genre proxy from artist_id (since genre isn't in the dataset)
        data['genre'] = data['artist_id'].apply(lambda x: hash(x) % 5)  # Simulate 5 genres
        
        # Feature selection strategy
        feature_strategy = self._get_feature_strategy(strategy='content')
        cols_to_keep = feature_strategy + ['artist_id', 'track_id', 'genre', 'interaction']
        data = data[cols_to_keep]
        
        # Create positive interactions
        positive_pairs = data[['artist_id', 'track_id', 'genre']].drop_duplicates()
        positive_pairs['interaction'] = 1
        
        # Create artist-track-genre map
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])
            self.genre_map[row['artist_id']] = row['genre']
        
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
        
        # Add track features
        self.track_features = data.drop(columns=['artist_id', 'interaction']).drop_duplicates('track_id')
        all_pairs = all_pairs.merge(self.track_features, on='track_id')
        
        # Hierarchical clustering
        self._apply_hierarchical_clustering(all_pairs)
        
        # Encode categorical features
        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])
        
        # Create mappings
        self.track_id_map = dict(zip(all_pairs['track_id'], all_pairs['track_id'].astype('category').cat.codes))
        self.artist_id_map = dict(zip(all_pairs['artist_id'], all_pairs['artist_id'].astype('category').cat.codes))
        
        # Scale numerical features
        numeric_cols = [col for col in all_pairs.columns if col not in ['artist_id', 'track_id', 'genre', 'interaction', 'cluster']]
        all_pairs[numeric_cols] = self.scaler.fit_transform(all_pairs[numeric_cols])
        
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
        fig = sns.pairplot(cluster_data, hue='cluster', palette='viridis')
        self._save_visualization(fig, "hierarchical_clustering")
        
    def initialize_model(self, hyperparams=None):
        """Initialize model with optional hyperparameter tuning"""
        if hyperparams:
            self.model = HoeffdingTree(
                nominal_attributes=[0, 1],  # artist_id and track_id
                grace_period=hyperparams['grace_period'],
                split_confidence=hyperparams['split_confidence'],
                tie_threshold=hyperparams['tie_threshold'],
                leaf_prediction='mc'
            )
            self.best_hyperparams = hyperparams
        else:
            self.model = HoeffdingTree(
                nominal_attributes=[0, 1],
                grace_period=200,
                split_confidence=1e-7,
                tie_threshold=0.05,
                leaf_prediction='mc'
            )
    
    def tune_hyperparameters(self, data, max_evals=50):
        """Hyperparameter tuning with Hyperopt and TPE"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        
        space = {
            'grace_period': hp.quniform('grace_period', 50, 500, 50),
            'split_confidence': hp.loguniform('split_confidence', np.log(1e-9), np.log(1e-3)),
            'tie_threshold': hp.uniform('tie_threshold', 0.01, 0.1)
        }
        
        def objective(params):
            model = HoeffdingTree(
                nominal_attributes=[0, 1],
                grace_period=int(params['grace_period']),
                split_confidence=params['split_confidence'],
                tie_threshold=params['tie_threshold'],
                leaf_prediction='mc'
            )
            
            # K-Fold Cross Validation
            kf = KFold(n_splits=3, shuffle=True)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Convert to numpy arrays
                X_train_np = X_train.values
                y_train_np = y_train.values
                X_val_np = X_val.values
                y_val_np = y_val.values
                
                # Train and evaluate
                model.partial_fit(X_train_np, y_train_np)
                y_pred = model.predict(X_val_np)
                scores.append(np.mean(y_pred == y_val_np))
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        
        # Initialize model with best params
        self.initialize_model({
            'grace_period': int(best['grace_period']),
            'split_confidence': best['split_confidence'],
            'tie_threshold': best['tie_threshold']
        })
        
        return best
    
    def train_with_cross_validation(self, data, n_splits=5):
        """Train with KFold cross-validation and group analysis"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_results = []
        group_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Convert to numpy arrays
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_val_np = X_val.values
            y_val_np = y_val.values
            
            # Train
            self.model.partial_fit(X_train_np, y_train_np)
            
            # Evaluate
            y_pred = self.model.predict(X_val_np)
            fold_acc = np.mean(y_pred == y_val_np)
            fold_results.append(fold_acc)
            
            # Group analysis by genre and cluster
            val_data = X_val.copy()
            val_data['true'] = y_val_np
            val_data['pred'] = y_pred
            
            # Genre analysis
            genre_loss = val_data.groupby('genre').apply(
                lambda x: mean_squared_error(x['true'], x['pred'])
            )
            
            # Cluster analysis
            cluster_loss = val_data.groupby('cluster').apply(
                lambda x: mean_squared_error(x['true'], x['pred'])
            )
            
            group_losses.append({
                'fold': fold,
                'genre_loss': genre_loss,
                'cluster_loss': cluster_loss
            })
        
        # Statistical analysis with pingouin
        loss_df = pd.DataFrame([{
            'fold': gl['fold'],
            'genre_loss_mean': gl['genre_loss'].mean(),
            'cluster_loss_mean': gl['cluster_loss'].mean()
        } for gl in group_losses])
        
        self.logger.info("\nStatistical Analysis of Group Losses:")
        self.logger.info(pg.anova(data=loss_df.melt(id_vars='fold'), dv='value', between='variable'))
        
        return np.mean(fold_results), group_losses
    
    def train_on_stream(self, data, stream_size=2000, batch_size=100):
        """Enhanced streaming training with metrics tracking"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        
        # Convert to numpy arrays
        X_np = X.values
        y_np = y.values
        
        # Create data stream
        stream = DataStream(X_np, y_np)
        
        self.logger.info(f"\nStarting stream training with {stream_size} samples (batch size: {batch_size})")
        
        n_samples = 0
        correct = 0
        start_time = time.time()
        last_improvement = 0
        best_loss = float('inf')
        
        while n_samples < stream_size and stream.has_more_samples():
            X_batch, y_batch = stream.next_sample(batch_size)
            
            # Partial fit
            self.model.partial_fit(X_batch, y_batch)
            
            # Predict and calculate metrics
            y_pred = self.model.predict(X_batch)
            y_proba = self.model.predict_proba(X_batch)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            batch_metrics = self._calculate_metrics(y_batch, y_pred, y_proba)
            
            # Update metrics history
            for metric, value in batch_metrics.items():
                if metric in self.metrics_history:
                    self.metrics_history[metric].append(value)
            
            batch_correct = np.sum(y_pred == y_batch)
            correct += batch_correct
            n_samples += batch_size
            
            # Adaptive learning
            batch_loss = 1 - (batch_correct / batch_size)
            self.metrics_history['batch_loss'].append(batch_loss)
            
            if batch_loss < best_loss:
                best_loss = batch_loss
                last_improvement = n_samples
            elif n_samples - last_improvement > 500:
                self.logger.info("Reducing split confidence due to plateau...")
                self.model.split_confidence *= 0.5
                last_improvement = n_samples
            
            # Log progress
            if n_samples % (batch_size * 10) == 0:
                self.logger.info(
                    f"Processed {n_samples}/{stream_size} | "
                    f"Accuracy: {batch_metrics['accuracy']:.3f} | "
                    f"F1: {batch_metrics['f1']:.3f} | "
                    f"AUC: {batch_metrics.get('roc_auc', 0):.3f}"
                )
        
        # Final evaluation and saving
        end_time = time.time()
        final_metrics = self._calculate_metrics(y_np[:n_samples], self.model.predict(X_np[:n_samples]))
        
        self.logger.info("\nTraining Completed:")
        self.logger.info(f"Total samples processed: {n_samples}")
        self.logger.info(f"Time elapsed: {end_time - start_time:.2f} seconds")
        self.logger.info(f"Final Accuracy: {final_metrics['accuracy']:.3f}")
        self.logger.info(f"Final F1 Score: {final_metrics['f1']:.3f}")
        
        # Generate and save visualizations
        self._plot_metrics_history()
        self._plot_feature_importance()
        self._analyze_regularization()
        
        # Save model
        model_path = self._save_model()
        return model_path
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Enhanced recommendation with cluster awareness"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.artist_id_map:
            self.logger.info(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self._get_popular_tracks(top_n)
        
        # Get artist's genre and cluster
        artist_genre = self.genre_map.get(artist_id, 0)
        artist_tracks = list(self.artist_track_map[artist_id])
        
        # Get all tracks in same cluster
        if len(artist_tracks) > 0:
            sample_track = artist_tracks[0]
            cluster = self.track_features[self.track_features['track_id'] == sample_track]['cluster'].values[0]
            candidate_tracks = self.track_features[self.track_features['cluster'] == cluster]['track_id'].unique()
        else:
            candidate_tracks = self.track_features['track_id'].unique()
        
        recommendations = []
        batch_size = 1000
        
        for i in range(0, len(candidate_tracks), batch_size):
            batch_tracks = candidate_tracks[i:i+batch_size]
            
            # Create feature matrix
            X_batch = []
            for track_id in batch_tracks:
                if track_id in self.track_id_map:
                    track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
                    X_batch.append([
                        self.artist_id_map[artist_id],
                        self.track_id_map[track_id],
                        track_data['danceability'],
                        track_data['energy'],
                        track_data['valence']
                    ])
            
            if X_batch:
                X_batch = np.array(X_batch)
                try:
                    y_proba = self.model.predict_proba(X_batch)
                    pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                except:
                    y_pred = self.model.predict(X_batch)
                    pos_proba = y_pred
                
                for j, track_id in enumerate(batch_tracks):
                    if track_id in self.track_id_map:
                        recommendations.append({
                            'track_id': track_id,
                            'score': pos_proba[j],
                            'danceability': X_batch[j][2],
                            'energy': X_batch[j][3],
                            'valence': X_batch[j][4],
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

# Example usage
if __name__ == "__main__":
    recommender = EnhancedMusicRecommender("music_rec_experiment1")
    
    try:
        # Step 1: Preprocess data with hierarchical clustering
        data = recommender.preprocess_data("your_dataset.csv", sample_size=5000)
        
        # Step 2: Hyperparameter tuning
        recommender.logger.info("\nStarting hyperparameter tuning...")
        best_params = recommender.tune_hyperparameters(data, max_evals=30)
        recommender.logger.info(f"\nBest hyperparameters: {best_params}")
        
        # Step 3: Train with cross-validation and group analysis
        recommender.logger.info("\nTraining with cross-validation...")
        avg_acc, group_losses = recommender.train_with_cross_validation(data)
        recommender.logger.info(f"\nAverage cross-validation accuracy: {avg_acc:.2f}")
        
        # Step 4: Stream training with adaptive learning
        recommender.logger.info("\nTraining on data stream...")
        model_path = recommender.train_on_stream(data, stream_size=2000, batch_size=100)
        
        # Step 5: Get recommendations
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