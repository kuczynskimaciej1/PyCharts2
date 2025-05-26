import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, 
                           confusion_matrix, classification_report)
import joblib
from datetime import datetime

class EnhancedMusicRecommender:
    def __init__(self, experiment_name="music_rec"):
        self._setup_logging(experiment_name)
        self._setup_directories(experiment_name)
        
        # Initialize other attributes
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
        
        # Save model
        model_path = self._save_model()
        return model_path

    # [Previous methods remain unchanged...]

# Example usage with enhanced metrics
if __name__ == "__main__":
    recommender = EnhancedMusicRecommender("music_rec_experiment1")
    
    try:
        # Load or preprocess data
        data = recommender.preprocess_data("../../data/full_training_data.csv", sample_size=5000)
        
        # Hyperparameter tuning
        recommender.logger.info("\nStarting hyperparameter tuning...")
        best_params = recommender.tune_hyperparameters(data, max_evals=30)
        
        # Train with cross-validation
        recommender.logger.info("\nTraining with cross-validation...")
        avg_acc, group_losses = recommender.train_with_cross_validation(data)
        
        # Stream training
        recommender.logger.info("\nTraining on data stream...")
        model_path = recommender.train_on_stream(data, stream_size=2000, batch_size=100)
        
        # Example recommendation
        example_artist = data['artist_id'].iloc[0]
        recommender.logger.info(f"\nGenerating recommendations for artist {example_artist}...")
        recs = recommender.recommend_tracks(example_artist, top_n=5)
        
        for i, rec in enumerate(recs, 1):
            recommender.logger.info(
                f"{i}. Track ID: {rec['track_id']} | "
                f"Score: {rec['score']:.3f} | "
                f"Dance: {rec['danceability']:.2f}"
            )
            
    except Exception as e:
        recommender.logger.error(f"Experiment failed: {str(e)}", exc_info=True)