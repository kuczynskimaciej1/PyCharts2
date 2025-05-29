from importlib import import_module
import pandas as pd
import numpy as np
from collections import defaultdict
from river import tree, compose, preprocessing, metrics, evaluate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, log_loss)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from scipy import stats
import psutil
import time
import seaborn as sns
import pingouin as pg
from functools import partial
import time
import logging
import os
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedMusicRecommender:
    def __init__(self, experiment_name="music_rec"):
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
        self.negative_samples_file = f"{self.output_dir}/negative_samples_checkpoint.pkl"
        self.hyperopt_trials_file = f"{self.output_dir}/hyperopt_trials.pkl"
        
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
        
    def _setup_directories(self, experiment_name):
        """Create directories for outputs"""
        self.output_dir = f"outputs/{experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)
        
    def _save_checkpoint(self, checkpoint_type, data):
        """Save checkpoint data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{self.output_dir}/checkpoints/{checkpoint_type}_{timestamp}.pkl"
        
        try:
            joblib.dump(data, checkpoint_path)
            self.logger.info(f"Saved {checkpoint_type} checkpoint to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            self.logger.error(f"Error saving {checkpoint_type} checkpoint: {str(e)}")
            return None
            
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint data"""
        try:
            if os.path.exists(checkpoint_path):
                data = joblib.load(checkpoint_path)
                self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
                return data
            return None
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            return None
    
    def _save_negative_samples(self, negative_samples, processed_artists):
        """Save negative samples generation progress"""
        checkpoint = {
            'negative_samples': negative_samples,
            'processed_artists': processed_artists,
            'artist_track_map': dict(self.artist_track_map),
            'genre_map': self.genre_map
        }
        self._save_checkpoint("negative_samples", checkpoint)
        
    def _load_negative_samples(self):
        """Load negative samples generation progress"""
        # Try to find the latest checkpoint
        checkpoint_files = [f for f in os.listdir(f"{self.output_dir}/checkpoints") 
                          if f.startswith("negative_samples")]
        
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint_path = f"{self.output_dir}/checkpoints/{latest_checkpoint}"
            data = self._load_checkpoint(checkpoint_path)
            
            if data:
                # Restore state
                negative_samples = data['negative_samples']
                processed_artists = data['processed_artists']
                self.artist_track_map = defaultdict(set, data['artist_track_map'])
                self.genre_map = data['genre_map']
                
                return negative_samples, processed_artists
                
        return [], set()
    
    def _save_hyperopt_trials(self, trials):
        """Save Hyperopt trials object"""
        joblib.dump(trials, self.hyperopt_trials_file)
        self.logger.info(f"Saved Hyperopt trials to {self.hyperopt_trials_file}")
        
    def _load_hyperopt_trials(self):
        """Load Hyperopt trials object if exists"""
        if os.path.exists(self.hyperopt_trials_file):
            trials = joblib.load(self.hyperopt_trials_file)
            self.logger.info(f"Loaded Hyperopt trials from {self.hyperopt_trials_file}")
            return trials
        return None
        
    def _save_visualization(self, fig, name):
        """Save visualization to file - now handles both matplotlib and seaborn figures"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.output_dir}/plots/{name}_{timestamp}.png"
        
        try:
            # Handle both matplotlib figures and seaborn grid objects
            if hasattr(fig, 'savefig'):
                fig.savefig(path, bbox_inches='tight', dpi=300)
            else:
                # For seaborn grid objects that don't directly have savefig
                plt.savefig(path, bbox_inches='tight', dpi=300)
            
            plt.close()  # Close the current figure
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
        
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive performance metrics with error handling"""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                # Only calculate ROC AUC if both classes are present
                if len(np.unique(y_true)) >= 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                else:
                    metrics['roc_auc'] = np.nan
                    metrics['pr_auc'] = np.nan
                    self.logger.warning("Only one class present - skipping ROC AUC and PR AUC calculation")
            except Exception as e:
                metrics['roc_auc'] = np.nan
                metrics['pr_auc'] = np.nan
                self.logger.warning(f"Error calculating probability metrics: {str(e)}")
        
        # Log classification report (only if both classes present)
        if len(np.unique(y_true)) >= 2:
            self.logger.info("\nClassification Report:\n" + classification_report(y_true, y_pred))
        else:
            self.logger.info("\nClassification Report: Only one class present")
        
        # Plot confusion matrix (handle single class case)
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            self._save_visualization(fig, "confusion_matrix")
        except Exception as e:
            self.logger.warning(f"Could not plot confusion matrix: {str(e)}")
        
        return metrics
    
    def _calculate_additional_metrics(self, y_true, y_pred, y_proba):
        """Calculate additional metrics"""
        metrics = {}
        
        # Log Loss
        if len(np.unique(y_true)) > 1:  # Only calculate if both classes present
            metrics['log_loss'] = log_loss(y_true, y_proba)
        
        # Cosine Similarity (between predicted and true class distributions)
        pred_dist = np.bincount(y_pred) / len(y_pred)
        true_dist = np.bincount(y_true) / len(y_true)
        max_classes = max(len(pred_dist), len(true_dist))
        pred_dist.resize(max_classes)
        true_dist.resize(max_classes)
        metrics['cosine_sim'] = 1 - cosine(pred_dist, true_dist)
        
        return metrics
    
    def _calculate_tree_complexity(self):
        """Calculate Gini index and entropy for the tree"""
        if not hasattr(self.model, 'n_nodes'):
            return {}
            
        # Placeholder for tree analysis - would need actual tree traversal
        return {
            'gini_index': 0.5,  # Placeholder
            'entropy': 0.3      # Placeholder
        }
    
    def _measure_inference_time(self, X, n_samples=100):
        """Measure inference time"""
        start = time.time()
        for _ in range(n_samples):
            xi = X.iloc[np.random.randint(0, len(X))].values
            self.model.predict_one(xi)
        return (time.time() - start) / n_samples
    
    def _perform_statistical_tests(self, data):
        """Perform t-tests and normality tests"""
        results = {}
        
        # Between genres
        genres = data['genre'].unique()
        if len(genres) >= 2:
            genre1 = data[data['genre'] == genres[0]]['interaction']
            genre2 = data[data['genre'] == genres[1]]['interaction']
            t_stat, p_val = stats.ttest_ind(genre1, genre2)
            results['t_test_genre'] = {'statistic': t_stat, 'p_value': p_val}
        
        # Normality tests for features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Test first 3 numeric columns
            stat, p = stats.shapiro(data[col].sample(min(5000, len(data))))
            results[f'normality_{col}'] = {'statistic': stat, 'p_value': p}
        
        return results
    
    def _plot_metrics_history(self):
        """Plot training metrics over time"""
        if not self.metrics_history:
            return
            
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(4, 2)
        
        # Classification metrics
        ax1 = fig.add_subplot(gs[0, 0])
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in self.metrics_history and self.metrics_history[metric]:
                ax1.plot(self.metrics_history[metric], label=metric)
        ax1.set_title('Classification Metrics')
        ax1.legend()
        
        # Probability metrics
        ax2 = fig.add_subplot(gs[0, 1])
        for metric in ['roc_auc', 'pr_auc', 'log_loss']:
            if metric in self.metrics_history and self.metrics_history[metric]:
                ax2.plot(self.metrics_history[metric], label=metric)
        ax2.set_title('Probability Metrics')
        ax2.legend()
        
        # Resource usage
        ax3 = fig.add_subplot(gs[1, 0])
        if 'batch_loss' in self.metrics_history:
            ax3.plot(self.metrics_history['batch_loss'], label='Batch Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        
        # System metrics
        ax4 = fig.add_subplot(gs[1, 1])
        if 'peak_memory_mb' in self.metrics_history:
            ax4.axhline(self.metrics_history['peak_memory_mb'], color='r', label='Peak Memory (MB)')
        if 'training_time' in self.metrics_history:
            ax4.axhline(self.metrics_history['training_time'], color='g', label='Training Time (s)')
        ax4.set_title('System Metrics')
        ax4.legend()
        
        # ... add more subplots as needed
    
        plt.tight_layout()
        self._save_visualization(fig, "full_metrics_history")
        
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
    
    def preprocess_data(self, file_path, sample_size=100, resume_negative_samples=True):
        """Enhanced preprocessing with checkpointing for negative samples generation"""
        self.logger.info("Loading and preprocessing data with hierarchical clustering...")
        data = pd.read_csv(file_path)
        
        # Check if 'interaction' column exists, if not create it
        if 'interaction' not in data.columns:
            self.logger.info("'Interaction' column not found - creating positive interactions")
            data['interaction'] = 1  # Assume all existing records are positive interactions
            self.logger.info("'Interaction' column created")

        # Create genre proxy from artist_id (since genre isn't in the dataset)
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

        # Rest of the preprocessing remains the same...
        positive_pairs = data[['artist_id', 'track_id', 'genre']].drop_duplicates()
        self.logger.info("Duplicates deleted")
        positive_pairs['interaction'] = 1
        
        # Create artist-track-genre map
        debug_counter = 0
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])
            self.genre_map[row['artist_id']] = row['genre']
            self.logger.info(f"Creating map: {debug_counter}")
            debug_counter += 1
        self.logger.info(f"Creating map finished")
        
        # Generate negative samples - with checkpointing
        negative_samples = []
        processed_artists = set()
        
        if resume_negative_samples:
            loaded_data = self._load_negative_samples()
            if loaded_data:
                negative_samples, processed_artists = loaded_data
                self.logger.info(f"Resumed negative samples generation with {len(processed_artists)} processed artists")
        
        all_tracks = set(data['track_id'].unique())
        self.logger.info(f"Generating negative samples - setting track_id unique")

        artists_to_process = [a for a in self.artist_track_map if a not in processed_artists]
        total_artists = len(artists_to_process)
        
        for i, artist in enumerate(artists_to_process):
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
            
            processed_artists.add(artist)
            
            # Save checkpoint every 100 artists or at the end
            if i % 100 == 0 or i == total_artists - 1:
                self._save_negative_samples(negative_samples, processed_artists)
                self.logger.info(f"Processed {i+1}/{total_artists} artists for negative samples")
            
            self.logger.info(f"Generating negative samples - artist done")

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs])
        self.logger.info(f"All pairs concatenated")

        # Add track features
        self.track_features = data.drop(columns=['artist_id', 'interaction'], errors='ignore').drop_duplicates('track_id')
        all_pairs = all_pairs.merge(self.track_features, on='track_id')
        self.logger.info(f"Added track features")

        # Hierarchical clustering
        self._apply_hierarchical_clustering(all_pairs)
        self.logger.info(f"Applied hierarchical clustering")
        
        # Encode categorical features
        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.logger.info(f"Encoded labels for artist")
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])
        self.logger.info(f"Encoded labels for track")
        
        # Create mappings
        self.track_id_map = dict(zip(all_pairs['track_id'], all_pairs['track_id'].astype('category').cat.codes))
        self.artist_id_map = dict(zip(all_pairs['artist_id'], all_pairs['artist_id'].astype('category').cat.codes))
        self.logger.info(f"Created mappings")
        
        # Scale numerical features
        numeric_cols = [col for col in all_pairs.columns if col not in ['artist_id', 'track_id', 'genre', 'interaction', 'cluster']]
        all_pairs[numeric_cols] = self.scaler.fit_transform(all_pairs[numeric_cols])
        self.logger.info(f"Numerical features scaled")
        
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
        
        # Visualize with Seaborn - modified to return matplotlib figure
        cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
        grid = sns.pairplot(cluster_data, hue='cluster', palette='viridis')
        
        # Save the figure from the PairGrid
        fig = grid.figure
        self._save_visualization(fig, "hierarchical_clustering")
        
    def initialize_model(self, hyperparams=None):
        """Initialize model with optional hyperparameter tuning"""
        if hyperparams:
            self.model = tree.HoeffdingTreeClassifier(
                grace_period=int(hyperparams['grace_period']),
                delta=hyperparams['split_confidence'],
                tau=hyperparams['tie_threshold'],
                leaf_prediction='mc'
            )
            self.best_hyperparams = hyperparams
        else:
            self.model = tree.HoeffdingTreeClassifier(
                grace_period=200,
                delta=1e-7,
                tau=0.05,
                leaf_prediction='mc'
            )
    
    def tune_hyperparameters(self, data, max_evals=50, resume=True):
        """Hyperparameter tuning with checkpointing"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        
        # Get feature names for dictionary conversion
        feature_names = X.columns.tolist()
        
        space = {
            'grace_period': hp.quniform('grace_period', 50, 500, 50),
            'split_confidence': hp.loguniform('split_confidence', np.log(1e-9), np.log(1e-3)),
            'tie_threshold': hp.uniform('tie_threshold', 0.01, 0.1)
        }
        
        def objective(params):
            model = tree.HoeffdingTreeClassifier(
                grace_period=int(params['grace_period']),
                delta=params['split_confidence'],
                tau=params['tie_threshold'],
                leaf_prediction='mc'
            )
            
            # K-Fold Cross Validation
            kf = KFold(n_splits=3, shuffle=True)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Convert to dictionary format for River
                train_data = [
                    (dict(zip(feature_names, x)), y) 
                    for x, y in zip(X_train.values, y_train.values)
                ]
                val_data = [
                    (dict(zip(feature_names, x)), y) 
                    for x, y in zip(X_val.values, y_val.values)
                ]
                
                # Train
                for xi, yi in train_data:
                    model.learn_one(xi, yi)
                
                # Evaluate
                correct = 0
                for xi, yi in val_data:
                    y_pred = model.predict_one(xi)
                    if y_pred == yi:
                        correct += 1
                
                scores.append(correct / len(X_val))
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        # Load previous trials if resuming
        trials = self._load_hyperopt_trials() if resume else Trials()
        if trials is None:
            trials = Trials()
            
        # Calculate how many more evaluations we need
        completed_evals = len(trials.trials)
        remaining_evals = max(0, max_evals - completed_evals)
        
        if remaining_evals > 0:
            self.logger.info(f"Resuming hyperparameter tuning from {completed_evals} completed evaluations")
            self.logger.info(f"Running {remaining_evals} additional evaluations")
            
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=completed_evals + remaining_evals,
                trials=trials,
                points_to_evaluate=[t['misc']['vals'] for t in trials.trials],
                # This prevents re-evaluating already tried points
            )
            
            # Save trials after each evaluation
            self._save_hyperopt_trials(trials)
        else:
            self.logger.info(f"Hyperparameter tuning already completed {max_evals} evaluations")
            best = trials.argmin
        
        # Initialize model with best params
        self.initialize_model({
            'grace_period': int(best['grace_period']),
            'split_confidence': best['split_confidence'],
            'tie_threshold': best['tie_threshold']
        })
        
        # Save the best parameters
        self._save_checkpoint("best_hyperparams", best)
        
        return best
    
    def train_with_cross_validation(self, data, n_splits=5):
        """Train with KFold cross-validation and group analysis"""
        # Ensure required columns exist
        required_columns = ['interaction']
        if 'genre' not in data.columns:
            self.logger.warning("'genre' column not found - skipping genre analysis")
        
        if 'cluster' not in data.columns:
            self.logger.warning("'cluster' column not found - skipping cluster analysis")
        
        X = data.drop(columns=['interaction'], errors='ignore')
        y = data['interaction']
        feature_names = X.columns.tolist()
        
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_results = []
        group_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Convert to River-compatible format
            train_data = [
                (dict(zip(feature_names, x)), y)
                for x, y in zip(X_train.values, y_train.values)
            ]
            val_data = [
                (dict(zip(feature_names, x)), y)
                for x, y in zip(X_val.values, y_val.values)
            ]
            
            # Train
            for xi, yi in train_data:
                self.model.learn_one(xi, yi)
            
            # Evaluate
            correct = 0
            y_preds = []
            y_trues = []
            
            for xi, yi in val_data:
                y_pred = self.model.predict_one(xi)
                y_preds.append(y_pred)
                y_trues.append(yi)
                if y_pred == yi:
                    correct += 1
            
            fold_acc = correct / len(X_val)
            fold_results.append(fold_acc)
            
            # Group analysis by available features
            val_data_df = X_val.copy()
            val_data_df['true'] = y_trues
            val_data_df['pred'] = y_preds
            
            current_group_loss = {'fold': fold}
            
            # Genre analysis (if column exists)
            if 'genre' in val_data_df.columns:
                try:
                    genre_loss = val_data_df.groupby('genre').apply(
                        lambda x: mean_squared_error(x['true'], x['pred'])
                    )
                    current_group_loss['genre_loss'] = genre_loss
                except Exception as e:
                    self.logger.warning(f"Genre analysis failed: {str(e)}")
            
            # Cluster analysis (if column exists)
            if 'cluster' in val_data_df.columns:
                try:
                    cluster_loss = val_data_df.groupby('cluster').apply(
                        lambda x: mean_squared_error(x['true'], x['pred'])
                    )
                    current_group_loss['cluster_loss'] = cluster_loss
                except Exception as e:
                    self.logger.warning(f"Cluster analysis failed: {str(e)}")
            
            group_losses.append(current_group_loss)
        
        # Statistical analysis with pingouin (only if we have group losses)
        if any('genre_loss' in gl for gl in group_losses) or any('cluster_loss' in gl for gl in group_losses):
            loss_df = pd.DataFrame([{
                'fold': gl['fold'],
                'genre_loss_mean': gl.get('genre_loss', {}).mean() if 'genre_loss' in gl else np.nan,
                'cluster_loss_mean': gl.get('cluster_loss', {}).mean() if 'cluster_loss' in gl else np.nan
            } for gl in group_losses])
            
            self.logger.info("\nStatistical Analysis of Group Losses:")
            
            try:
                # Melt the dataframe for ANOVA
                melted_df = loss_df.melt(id_vars='fold', value_vars=['genre_loss_mean', 'cluster_loss_mean'],
                                        var_name='group_type', value_name='loss')
                
                # Remove NaN values
                melted_df = melted_df.dropna()
                
                if not melted_df.empty:
                    anova_results = pg.anova(data=melted_df, dv='loss', between='group_type')
                    self.logger.info(anova_results)
                else:
                    self.logger.info("No valid group loss data for statistical analysis")
            except Exception as e:
                self.logger.error(f"Statistical analysis failed: {str(e)}")
        
        return np.mean(fold_results), group_losses
    
    def train_on_stream(self, data, stream_size=100, batch_size=10):
        """Enhanced streaming training with metrics tracking"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        feature_names = X.columns.tolist()
        
        self.logger.info(f"\nStarting stream training with {stream_size} samples (batch size: {batch_size})")
        
        n_samples = 0
        correct = 0
        start_time = time.time()
        last_improvement = 0
        best_loss = float('inf')
        
        metric = metrics.ClassificationReport()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        peak_mem = start_mem
        
        while n_samples < stream_size and n_samples < len(X):
            batch_end = min(n_samples + batch_size, len(X))
            X_batch = X.iloc[n_samples:batch_end].values
            y_batch = y.iloc[n_samples:batch_end].values
            
            # Convert to River-compatible format
            batch_data = [
                (dict(zip(feature_names, x)), y)
                for x, y in zip(X_batch, y_batch)
            ]
            
            # Train and evaluate
            y_preds = []
            y_probas = []
            
            for xi, yi in batch_data:
                y_pred = self.model.predict_one(xi)
                y_preds.append(y_pred)
                
                # Get probability if available
                if hasattr(self.model, 'predict_proba_one'):
                    y_proba = self.model.predict_proba_one(xi)
                    y_probas.append(y_proba.get(1, 0))  # Probability of positive class
                else:
                    y_probas.append(y_pred)
                
                self.model.learn_one(xi, yi)
                metric.update(yi, y_pred)
                
                if y_pred == yi:
                    correct += 1
            
            # Update peak memory
            current_mem = process.memory_info().rss / 1024 / 1024
            peak_mem = max(peak_mem, current_mem)
            
            # Calculate metrics for this batch
            y_batch = y_batch.astype(int)
            y_preds = np.array(y_preds).astype(int)
            y_probas = np.array(y_probas)
            
            batch_metrics = self._calculate_metrics(y_batch, y_preds, y_probas)
            
            # Update metrics history
            for metric_name, value in batch_metrics.items():
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append(value)
            
            n_samples += len(X_batch)
            
            # Adaptive learning
            batch_loss = 1 - (correct / n_samples)
            self.metrics_history['batch_loss'].append(batch_loss)
            
            if batch_loss < best_loss:
                best_loss = batch_loss
                last_improvement = n_samples
            elif n_samples - last_improvement > 500:
                self.logger.info("Reducing delta (split confidence) due to plateau...")
                self.model.delta *= 0.5
                last_improvement = n_samples
            
            # Log progress
            if n_samples % (batch_size * 10) == 0:
                self.logger.info(
                    f"Processed {n_samples}/{min(stream_size, len(X))} | "
                    f"Accuracy: {batch_metrics['accuracy']:.3f} | "
                    f"F1: {batch_metrics['f1']:.3f} | "
                    f"AUC: {batch_metrics.get('roc_auc', 0):.3f}"
                )
        
        # Final calculations
        end_time = time.time()
        training_time = end_time - start_time
        
        # Measure inference time
        inference_time = self._measure_inference_time(X, feature_names)
        
        # Update metrics with final values
        self.metrics_history.update({
            'training_time': training_time,
            'peak_memory_mb': peak_mem,
            'avg_inference_time': inference_time
        })
        
        # Final evaluation
        y_preds = []
        for xi in X.values:
            x_dict = dict(zip(feature_names, xi))
            y_preds.append(self.model.predict_one(x_dict))
        
        final_metrics = self._calculate_metrics(y.values, np.array(y_preds))
        
        self.logger.info("\nTraining Completed:")
        self.logger.info(f"Total samples processed: {n_samples}")
        self.logger.info(f"Time elapsed: {training_time:.2f} seconds")
        self.logger.info(f"Final Accuracy: {final_metrics['accuracy']:.3f}")
        self.logger.info(f"Final F1 Score: {final_metrics['f1']:.3f}")
        
        # Generate and save visualizations
        self._plot_metrics_history()
        self._analyze_regularization()
        
        # Save model
        model_path = self._save_model()
        return model_path
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Enhanced recommendation with cluster awareness and error handling"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.artist_id_map:
            self.logger.info(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self._get_popular_tracks(top_n)
        
        # Get feature names (ensure these match your actual features)
        feature_names = ['artist_id', 'track_id', 'danceability', 'energy', 'valence']
        
        # Get artist's tracks
        artist_tracks = list(self.artist_track_map.get(artist_id, []))
        
        # Get all candidate tracks (with cluster fallback)
        candidate_tracks = []
        
        try:
            # Try to use cluster if available
            if artist_tracks and 'cluster' in self.track_features.columns:
                sample_track = artist_tracks[0]
                cluster = self.track_features.loc[
                    self.track_features['track_id'] == sample_track, 'cluster'
                ].values[0]
                candidate_tracks = self.track_features[
                    self.track_features['cluster'] == cluster
                ]['track_id'].unique()
            else:
                # Fallback to all tracks if cluster not available
                candidate_tracks = self.track_features['track_id'].unique()
        except Exception as e:
            self.logger.warning(f"Cluster-based recommendation failed: {str(e)}")
            candidate_tracks = self.track_features['track_id'].unique()
        
        recommendations = []
        batch_size = 1000
        
        for i in range(0, len(candidate_tracks), batch_size):
            batch_tracks = candidate_tracks[i:i+batch_size]
            
            # Create feature matrix
            X_batch = []
            for track_id in batch_tracks:
                if track_id in self.track_id_map:
                    try:
                        track_data = self.track_features[self.track_features['track_id'] == track_id].iloc[0]
                        X_batch.append([
                            self.artist_id_map[artist_id],
                            self.track_id_map[track_id],
                            track_data['danceability'],
                            track_data['energy'],
                            track_data['valence']
                        ])
                    except Exception as e:
                        self.logger.warning(f"Skipping track {track_id}: {str(e)}")
                        continue
            
            if X_batch:
                # Convert to River-compatible format and predict
                scores = []
                for x in X_batch:
                    x_dict = dict(zip(feature_names, x))
                    try:
                        if hasattr(self.model, 'predict_proba_one'):
                            proba = self.model.predict_proba_one(x_dict)
                            scores.append(proba.get(1, 0))  # Probability of positive class
                        else:
                            pred = self.model.predict_one(x_dict)
                            scores.append(pred)
                    except Exception as e:
                        self.logger.warning(f"Prediction error: {str(e)}")
                        scores.append(0)
                
                scores = np.array(scores)
                
                for j, track_id in enumerate(batch_tracks):
                    if j < len(scores):  # Ensure we don't go out of bounds
                        try:
                            recommendations.append({
                                'track_id': track_id,
                                'score': scores[j],
                                'danceability': X_batch[j][2],
                                'energy': X_batch[j][3],
                                'valence': X_batch[j][4],
                                'cluster': cluster if 'cluster' in locals() else None
                            })
                        except Exception as e:
                            self.logger.warning(f"Error creating recommendation for track {track_id}: {str(e)}")
        
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
    
    def _measure_inference_time(self, X, feature_names, n_samples=100):
        """Measure inference time with proper data format conversion"""
        start = time.time()
        for _ in range(n_samples):
            xi = X.iloc[np.random.randint(0, len(X))].values
            x_dict = dict(zip(feature_names, xi))
            self.model.predict_one(x_dict)
        return (time.time() - start) / n_samples
    
    def _setup_logging(self, experiment_name):
        """Configure logging to file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{experiment_name}_log.txt"),
                logging.StreamHandler(),
                logging.FileHandler(f"{self.output_dir}/detailed_metrics.log")  # Separate metrics log
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        self.metrics_logger = logging.getLogger(f"{experiment_name}_metrics")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.propagate = False
        
        # Add handler for metrics only
        metrics_handler = logging.FileHandler(f"{self.output_dir}/metrics.csv")
        metrics_handler.setFormatter(logging.Formatter('%(message)s'))
        self.metrics_logger.addHandler(metrics_handler)
        
        # Write CSV header
        self.metrics_logger.info("batch,accuracy,precision,recall,f1,roc_auc,pr_auc,log_loss,cosine_sim")

if __name__ == "__main__":
    recommender = EnhancedMusicRecommender("music_rec_experiment1")
    
    try:
        # Step 1: Preprocess data with checkpointing for negative samples
        data = recommender.preprocess_data("../../../data/one_thousand.csv", 
                                         sample_size=1000,
                                         resume_negative_samples=True)
        
        # Step 2: Hyperparameter tuning with checkpointing
        recommender.logger.info("\nStarting hyperparameter tuning...")
        best_params = recommender.tune_hyperparameters(data, max_evals=30, resume=True)
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