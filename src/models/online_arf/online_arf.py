from river import ensemble, metrics, evaluate, preprocessing
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, 
                           confusion_matrix, classification_report)
from scipy.cluster.hierarchy import linkage, fcluster  # Dodaj te importy na początku pliku
import psutil
import time
import warnings
warnings.filterwarnings('ignore')

class ARFMusicRecommender:
    def __init__(self, experiment_name="music_rec_arf"):
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
    
    def preprocess_data(self, file_path, sample_size=10000):
        """Preprocess data with hierarchical clustering"""
        self.logger.info("Loading and preprocessing data with hierarchical clustering...")
        data = pd.read_csv(file_path)
        
        if 'interaction' not in data.columns:
            self.logger.info("'Interaction' column not found - creating positive interactions")
            data['interaction'] = 1
            self.logger.info("'Interaction' column created")

        # Create genre proxy from artist_id
        data['genre'] = data['artist_id'].apply(lambda x: hash(x) % 5)
        self.logger.info("'Genre' column created")
        
        # Feature selection
        feature_strategy = self._get_feature_strategy(strategy='content')
        self.logger.info("Feature strategy chosen as content")
        cols_to_keep = feature_strategy + ['artist_id', 'track_id', 'genre', 'interaction']
        self.logger.info("Columns to keep chosen")
        
        cols_to_keep = [col for col in cols_to_keep if col in data.columns]
        data = data[cols_to_keep]
        self.logger.info("Columns to keep filtered")

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
        
        # Generate negative samples
        negative_samples = []
        all_tracks = set(data['track_id'].unique())
        self.logger.info(f"Generating negative samples - setting track_id unique")

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
        
        # Visualize with Seaborn
        cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
        grid = sns.pairplot(cluster_data, hue='cluster', palette='viridis')
        fig = grid.figure
        self._save_visualization(fig, "hierarchical_clustering")
        
    def initialize_model(self, hyperparams=None):
        """Initialize ARF model with optional hyperparameters"""
        if hyperparams:
            self.model = ensemble.AdaptiveRandomForestClassifier(
                n_models=hyperparams['n_models'],
                max_features=hyperparams['max_features'],
                lambda_value=hyperparams['lambda_value'],
                drift_detector=hyperparams['drift_detector'],
                warning_detector=hyperparams['warning_detector'],
                metric=metrics.Accuracy()
            )
            self.best_hyperparams = hyperparams
        else:
            # Default parameters
            self.model = ensemble.AdaptiveRandomForestClassifier(
                n_models=10,
                max_features='sqrt',
                lambda_value=6,
                drift_detector=None,
                warning_detector=None,
                metric=metrics.Accuracy()
            )
    
    def tune_hyperparameters(self, data, max_evals=30):
        """Hyperparameter tuning for ARF with Hyperopt"""
        from river.drift import ADWIN  # Importujemy detektor dryfu
        
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        feature_names = X.columns.tolist()
        
        space = {
            'n_models': hp.quniform('n_models', 5, 30, 5),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'lambda_value': hp.uniform('lambda_value', 1, 10),
            'drift_detector': hp.choice('drift_detector', [None, ADWIN()]),  # Używamy ADWIN zamiast DriftDetector
            'warning_detector': hp.choice('warning_detector', [None, ADWIN()])
        }
        
        def objective(params):
            model = ensemble.AdaptiveRandomForestClassifier(
                n_models=int(params['n_models']),
                max_features=params['max_features'],
                lambda_value=params['lambda_value'],
                drift_detector=params['drift_detector'],
                warning_detector=params['warning_detector'],
                metric=metrics.Accuracy()
            )
            
            # K-Fold Cross Validation
            kf = KFold(n_splits=3, shuffle=True)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
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
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        
        # Initialize model with best params
        self.initialize_model({
            'n_models': int(best['n_models']),
            'max_features': best['max_features'],
            'lambda_value': best['lambda_value'],
            'drift_detector': best['drift_detector'],
            'warning_detector': best['warning_detector']
        })
        
        return best
    
    def train_with_cross_validation(self, data, n_splits=5):
        """Train ARF with KFold cross-validation"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        feature_names = X.columns.tolist()
        
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
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
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(y_trues, y_preds)
            for metric, value in fold_metrics.items():
                if metric in self.metrics_history:
                    self.metrics_history[metric].append(value)
            
            self.logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.3f}")
        
        return np.mean(fold_results), fold_results
    
    def train_on_stream(self, data, stream_size=2000, batch_size=100):
        """Streaming training for ARF with metrics tracking"""
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        feature_names = X.columns.tolist()
        
        self.logger.info(f"\nStarting ARF stream training with {stream_size} samples (batch size: {batch_size})")
        
        n_samples = 0
        correct = 0
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        peak_mem = start_mem
        
        metric = metrics.ClassificationReport()
        
        while n_samples < stream_size and n_samples < len(X):
            batch_end = min(n_samples + batch_size, len(X))
            X_batch = X.iloc[n_samples:batch_end].values
            y_batch = y.iloc[n_samples:batch_end].values
            
            batch_data = [
                (dict(zip(feature_names, x)), y)
                for x, y in zip(X_batch, y_batch)
            ]
            
            y_preds = []
            y_probas = []
            
            for xi, yi in batch_data:
                y_pred = self.model.predict_one(xi)
                y_preds.append(y_pred)
                
                if hasattr(self.model, 'predict_proba_one'):
                    y_proba = self.model.predict_proba_one(xi)
                    y_probas.append(y_proba.get(1, 0))
                else:
                    y_probas.append(y_pred)
                
                self.model.learn_one(xi, yi)
                metric.update(yi, y_pred)
                
                if y_pred == yi:
                    correct += 1
            
            # Update peak memory
            current_mem = process.memory_info().rss / 1024 / 1024
            peak_mem = max(peak_mem, current_mem)
            
            # Calculate metrics
            y_batch = y_batch.astype(int)
            y_preds = np.array(y_preds).astype(int)
            y_probas = np.array(y_probas)
            
            batch_metrics = self._calculate_metrics(y_batch, y_preds, y_probas)
            
            # Update metrics history
            for metric_name, value in batch_metrics.items():
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append(value)
            
            n_samples += len(X_batch)
            
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
        
        # Update metrics with final values
        self.metrics_history.update({
            'training_time': training_time,
            'peak_memory_mb': peak_mem
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
        
        # Save model
        model_path = self._save_model()
        return model_path
    
    def recommend_tracks(self, artist_id, top_n=5):
        """Enhanced recommendation with cluster awareness for ARF"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if artist_id not in self.artist_id_map:
            self.logger.info(f"Artist {artist_id} not seen during training. Returning popular tracks.")
            return self._get_popular_tracks(top_n)
        
        feature_names = ['artist_id', 'track_id', 'danceability', 'energy', 'valence']
        
        # Get artist's tracks and cluster
        artist_tracks = list(self.artist_track_map[artist_id])
        
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
                scores = []
                for x in X_batch:
                    x_dict = dict(zip(feature_names, x))
                    try:
                        if hasattr(self.model, 'predict_proba_one'):
                            proba = self.model.predict_proba_one(x_dict)
                            scores.append(proba.get(1, 0))
                        else:
                            pred = self.model.predict_one(x_dict)
                            scores.append(pred)
                    except Exception as e:
                        self.logger.warning(f"Prediction error: {str(e)}")
                        scores.append(0)
                
                scores = np.array(scores)
                
                for j, track_id in enumerate(batch_tracks):
                    if track_id in self.track_id_map:
                        recommendations.append({
                            'track_id': track_id,
                            'score': scores[j],
                            'danceability': X_batch[j][2],
                            'energy': X_batch[j][3],
                            'valence': X_batch[j][4],
                            'cluster': cluster
                        })
        
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
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{experiment_name}_log.txt"),
                logging.StreamHandler(),
                logging.FileHandler(f"{self.output_dir}/detailed_metrics.log")
            ]
        )
        self.logger = logging.getLogger(experiment_name)

# Example usage
if __name__ == "__main__":
    recommender = ARFMusicRecommender("music_rec_arf_experiment1")
    
    try:
        # Step 1: Preprocess data
        data = recommender.preprocess_data("../../../data/one_thousand.csv", sample_size=1000)
        
        # Step 2: Hyperparameter tuning
        recommender.logger.info("\nStarting ARF hyperparameter tuning...")
        best_params = recommender.tune_hyperparameters(data, max_evals=30)
        recommender.logger.info(f"\nBest hyperparameters: {best_params}")
        
        # Step 3: Train with cross-validation
        recommender.logger.info("\nTraining ARF with cross-validation...")
        avg_acc, fold_results = recommender.train_with_cross_validation(data)
        recommender.logger.info(f"\nAverage cross-validation accuracy: {avg_acc:.2f}")
        
        # Step 4: Stream training
        recommender.logger.info("\nTraining ARF on data stream...")
        model_path = recommender.train_on_stream(data, stream_size=2000, batch_size=100)
        
        # Step 5: Get recommendations
        example_artist = data['artist_id'].iloc[0]
        recommender.logger.info(f"\nARF Recommendations for artist {example_artist}:")
        recs = recommender.recommend_tracks(example_artist, top_n=5)
        
        recommender.logger.info("\nTop recommendations:")
        for i, rec in enumerate(recs, 1):
            recommender.logger.info(
                f"{i}. Track ID: {rec['track_id']} | Score: {rec['score']:.3f} | "
                f"Dance: {rec['danceability']:.2f} | Energy: {rec['energy']:.2f} | "
                f"Cluster: {rec['cluster']}"
            )
            
    except Exception as e:
        recommender.logger.error(f"ARF Experiment failed: {str(e)}", exc_info=True)