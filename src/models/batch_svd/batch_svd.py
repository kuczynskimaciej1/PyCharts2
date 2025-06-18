from surprise import SVD, Dataset, Reader
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, 
                             classification_report, silhouette_score)
import shap
from sklearn.ensemble import RandomForestRegressor
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
import json
import random

warnings.filterwarnings('ignore')

class BatchSVDRecommender:

    NUMERIC_FEATURES = [
        'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
        'tempo', 'time_signature', 'valence'
    ]
    OTHER_FEATURES = ['explicit', 'artist_id', 'release_id']

    def __init__(self, experiment_name="batch_svd_rec"):
        self._setup_directories(experiment_name)
        self._setup_logging(experiment_name)
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
        self._init_metrics_history()
        self.feature_cols = None

    def _init_metrics_history(self):
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
        self.output_dir = f"outputs/{experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)

    def _save_visualization(self, fig, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.output_dir}/plots/{name}_{timestamp}.png"
        try:
            if hasattr(fig, 'savefig'):
                fig.savefig(path, bbox_inches='tight', dpi=300)
            else:
                plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            self.logger.info(f"Saved visualization: {path}")
        except Exception as e:
            self.logger.error(f"Error saving visualization {name}: {str(e)}")

    def _save_model(self):
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
        self.logger.info(f"Saved model: {model_path}")
        return model_path

    def _load_model(self, model_path):
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

    def plot_feature_distributions(self, data):
        feat_cols = self.feature_cols
        fig, axes = plt.subplots(1, min(3, len(feat_cols)), figsize=(15, 4))
        if len(feat_cols) == 1:
            axes = [axes]
        for ax, feat in zip(axes, feat_cols[:3]):
            if feat not in data.columns: continue
            sns.histplot(data[feat], ax=ax, kde=True)
            ax.set_title(f'Distribution of {feat}')
        plt.tight_layout()
        path = f"{self.output_dir}/plots/feature_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(path)
        plt.close(fig)
        self.logger.info(f"Saved feature distributions: {path}")

        # --- Zaawansowane metryki i raporty jak w DeepLearningRecommender ---

    def compute_diversity(self, recs):
        """Compute avg pairwise distance (diversity) in oryginalnym feature space."""
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

    def plot_elbow_and_silhouette(self, data, max_clusters=10):
        features = data[self.feature_cols].fillna(0).values
        inertias, silhouettes = [], []
        for k in range(2, max_clusters + 1):
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(features, labels))
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(range(2, max_clusters + 1), inertias, marker='o')
        ax[0].set_title('Elbow curve (inertia)')
        ax[1].plot(range(2, max_clusters + 1), silhouettes, marker='o', color='green')
        ax[1].set_title('Silhouette score')
        plt.tight_layout()
        path = f"{self.output_dir}/plots/elbow_silhouette_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(path)
        plt.close(fig)

    def _plot_training_history(self):
        hist = self.metrics_history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        bar_width = 0.35
        index = np.arange(1)  # tylko jeden punkt!
        # RMSE
        axes[0,0].bar(index, hist['train_rmse'], bar_width, label='Train RMSE')
        axes[0,0].bar(index+bar_width, hist['val_rmse'], bar_width, label='Val RMSE')
        axes[0,0].set_title('RMSE'); axes[0,0].legend()
        # MAE
        axes[0,1].bar(index, hist['train_mae'], bar_width, label='Train MAE')
        axes[0,1].bar(index+bar_width, hist['val_mae'], bar_width, label='Val MAE')
        axes[0,1].set_title('MAE'); axes[0,1].legend()
        # AUC
        axes[1,0].bar(index, hist['train_auc'], bar_width, label='Train AUC')
        axes[1,0].bar(index+bar_width, hist['val_auc'], bar_width, label='Val AUC')
        axes[1,0].set_title('AUC'); axes[1,0].legend()
        # Precision/Recall
        axes[1,1].bar(index, hist['train_precision'], bar_width, label='Train Prec')
        axes[1,1].bar(index+bar_width, hist['val_precision'], bar_width, label='Val Prec')
        axes[1,1].bar(index, hist['train_recall'], bar_width, label='Train Rec', alpha=0.5, bottom=hist['train_precision'])
        axes[1,1].bar(index+bar_width, hist['val_recall'], bar_width, label='Val Rec', alpha=0.5, bottom=hist['val_precision'])
        axes[1,1].set_title('Precision/Recall'); axes[1,1].legend()
        plt.tight_layout()
        self._save_visualization(fig, "training_history")

    def plot_shap(self, data):
        try:

            tracks_le = self.label_encoders['track_id']
            artists_le = self.label_encoders['artist_id']
            # Zbierz embeddingi userów/tracks jako features
            user_indices = artists_le.transform(data['artist_id'])
            item_indices = tracks_le.transform(data['track_id'])
            X = np.concatenate([
                self.model.pu[user_indices],      # user embeddings
                self.model.qi[item_indices],      # item embeddings
                data[self.feature_cols].values    # content features
            ], axis=1)
            y = data['interaction'].values
            model = RandomForestRegressor(n_estimators=40, random_state=1)
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X[:200])
            fig = shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.title("SHAP feature importances (users/items/content)")
            self._save_visualization(plt.gcf(), "shap_summary")
        except Exception as e:
            self.logger.warning(f"Could not plot SHAP: {e}")

    def _plot_feature_importance(self):
        # SVD nie ma feature importance, ale można wykreslić wariancję faktorów
        try:
            user_factors = self.model.pu
            item_factors = self.model.qi
            user_importance = np.var(user_factors, axis=0)
            item_importance = np.var(item_factors, axis=0)
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x=user_importance, y=[f"UF_{i}" for i in range(len(user_importance))], ax=ax1)
            ax1.set_title('User Latent Factors Importance')
            self._save_visualization(fig1, "user_factors_importance")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(x=item_importance, y=[f"IF_{i}" for i in range(len(item_importance))], ax=ax2)
            ax2.set_title('Item Latent Factors Importance')
            self._save_visualization(fig2, "item_factors_importance")
        except Exception as e:
            self.logger.warning(f"Could not plot feature importance: {str(e)}")

    def preprocess_data_balanced(
        self, 
        file_path, 
        sample_size=None, 
        n_clusters=6, 
        elbow_max_clusters=12,
        negatives_ratio=2
    ):
        self.logger.info("Loading CSV data...")
        data = pd.read_csv(file_path)
        if sample_size: data = data.sample(sample_size, random_state=42)
        if 'explicit' in data.columns: data['explicit'] = data['explicit'].astype(int)
        for col in ['artist_id', 'release_id']:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: eval(x)[0] if (isinstance(x, str) and x.startswith('[')) else x)
                le = LabelEncoder()
                data[col + '_le'] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        ALL_FEATURES = [f for f in self.NUMERIC_FEATURES + ['explicit', 'artist_id_le', 'release_id_le'] if f in data.columns]
        self.feature_cols = ALL_FEATURES

        self.track_features = data[['track_id'] + self.feature_cols].drop_duplicates('track_id')
        positive_pairs = data[['artist_id', 'track_id']].drop_duplicates().copy()
        positive_pairs['interaction'] = 1

        all_artists = data['artist_id'].unique()
        all_tracks = data['track_id'].unique()
        negative_samples = []
        artist_track_positive = set(zip(positive_pairs['artist_id'], positive_pairs['track_id']))
        np.random.seed(42)
        for artist in all_artists:
            pos_tracks = set(data[data['artist_id'] == artist]['track_id'])
            neg_tracks = list(set(all_tracks) - pos_tracks)
            # Agresywnie: więcej negatywnych niż pozytywnych, ale nie 100x!
            n_pos = len(pos_tracks)
            neg_samples_to_draw = min(len(neg_tracks), n_pos * negatives_ratio)
            if neg_samples_to_draw == 0: continue
            sampled_negatives = np.random.choice(neg_tracks, size=neg_samples_to_draw, replace=False)
            for track in sampled_negatives:
                negative_samples.append({'artist_id': artist, 'track_id': track, 'interaction': 0})

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        # Wywal rzadkich userów/itemów
        min_occurrences = 3
        value_counts_artist = all_pairs['artist_id'].value_counts()
        value_counts_track = all_pairs['track_id'].value_counts()
        to_keep_artist = value_counts_artist[value_counts_artist >= min_occurrences].index
        to_keep_track = value_counts_track[value_counts_track >= min_occurrences].index
        all_pairs = all_pairs[all_pairs['artist_id'].isin(to_keep_artist) & all_pairs['track_id'].isin(to_keep_track)]
        all_pairs = all_pairs.merge(self.track_features, how='left', on='track_id')

        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])

        scaler = MinMaxScaler()
        all_pairs[self.feature_cols] = scaler.fit_transform(all_pairs[self.feature_cols].fillna(0))
        self.scaler = scaler
        self.plot_elbow_and_silhouette(all_pairs, max_clusters=elbow_max_clusters)
        all_pairs = self._apply_hierarchical_clustering(all_pairs, n_clusters)
        self.logger.info(f"Applied hierarchical clustering with n_clusters={n_clusters}")
        self.plot_feature_distributions(all_pairs)
        self.track_features = all_pairs[['track_id'] + self.feature_cols + ['cluster']].drop_duplicates('track_id')
        return all_pairs

    def _apply_hierarchical_clustering(self, data, n_clusters = 6):
        cluster_features = data[self.feature_cols].fillna(0).values
        Z = linkage(cluster_features, method='ward')
        self.cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        data['cluster'] = self.cluster_labels
        # Visualization
        try:
            cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
            grid = sns.pairplot(
                cluster_data, 
                hue='cluster', 
                palette='viridis',
                plot_kws={'s':15, 'alpha':0.3},   # <-- tu!
                diag_kws={'alpha':0.3}
            )
            self._save_visualization(grid.fig, "hierarchical_clustering")
        except Exception as e:
            self.logger.error(f"Could not visualize clusters: {str(e)}")
        return data
    
    def gridsearch_model(self, data):
        reader = Reader(rating_scale=(0, 1))
        surprise_data = Dataset.load_from_df(data[['artist_id', 'track_id', 'interaction']], reader)
        param_grid = {
            'n_factors': [8, 12, 16, 24, 32],
            'n_epochs': [20, 30, 40],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.05, 0.08, 0.1, 0.2]   # WIĘKSZA regularizacja
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=4, n_jobs=-1)
        gs.fit(surprise_data)
        best_params = gs.best_params['rmse']
        self.logger.info(f"Best RMSE: {gs.best_score['rmse']:0.4f} Best params: {best_params}")
        self.initialize_model(
            n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'],
            lr_all=best_params['lr_all'], reg_all=best_params['reg_all']
        )
        return best_params
    
    def rebalance_global_positive_negative_pairs(self, all_pairs: pd.DataFrame, random_state=42):
        """
        Zapewnia balans liczby pozytywnych i negatywnych przykładów na poziomie całego zbioru.
        Wybiera losowy podzbiór negatywnych tak, by liczebność = liczba pozytywnych.
        """
        pos = all_pairs[all_pairs['interaction'] == 1]
        neg = all_pairs[all_pairs['interaction'] == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        if n_neg > n_pos:
            neg_balanced = neg.sample(n=n_pos, random_state=random_state)
        else:
            neg_balanced = neg
        all_pairs_balanced = pd.concat([pos, neg_balanced], ignore_index=True)
        all_pairs_balanced = all_pairs_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        self.logger.info(f"After global rebalancing: {all_pairs_balanced['interaction'].value_counts().to_dict()}")
        return all_pairs_balanced

    def initialize_model(self, n_factors=32, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.best_hyperparams = {
            'n_factors': n_factors, 'n_epochs': n_epochs,
            'lr_all': lr_all, 'reg_all': reg_all
        }
        self.logger.info(f"Initialized SVD model with: {self.best_hyperparams}")

    def _calculate_metrics(self, y_true, y_pred, y_proba=None, threshold=0.5):
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
        self.logger.info("\nClassification Report:\n" + classification_report(y_true_binary, y_pred_binary))
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        self._save_visualization(fig, "confusion_matrix")
        return metrics

    def train_model(self, data):
        self.logger.info("Starting SVD training...")
        start_time = time.time()
        reader = Reader(rating_scale=(0, 1))  # Bo 'interaction' 0 lub 1
        surprise_data = Dataset.load_from_df(data[['artist_id', 'track_id', 'interaction']], reader)
        # Podział train/val taki jak w deep learning
        all_rows = data.copy()
        X_train, X_val = train_test_split(all_rows, test_size=0.2, random_state=42, stratify=all_rows['interaction'])

        # Konwersja na Surprise set
        surprise_train = Dataset.load_from_df(X_train[['artist_id', 'track_id', 'interaction']], reader)
        surprise_val = Dataset.load_from_df(X_val[['artist_id', 'track_id', 'interaction']], reader)
        trainset = surprise_train.build_full_trainset()
        self.model.fit(trainset)

        # Predykcje train
        train_preds = []
        for _, row in X_train.iterrows():
            pred = self.model.predict(row['artist_id'], row['track_id']).est
            train_preds.append(pred)
        y_train_true = X_train['interaction'].values
        y_train_pred = np.array(train_preds)

        # Predykcje val
        val_preds = []
        for _, row in X_val.iterrows():
            pred = self.model.predict(row['artist_id'], row['track_id']).est
            val_preds.append(pred)
        y_val_true = X_val['interaction'].values
        y_val_pred = np.array(val_preds)

        # Ewaluacja
        tmetrics = self._calculate_metrics(y_train_true, y_train_pred, y_proba=y_train_pred, threshold=0.4)
        vmetrics = self._calculate_metrics(y_val_true, y_val_pred, y_proba=y_val_pred, threshold=0.4)
        # Zbierz do historii analogicznie jak w DL
        self.metrics_history['train_rmse'].append(tmetrics['rmse'])
        self.metrics_history['val_rmse'].append(vmetrics['rmse'])
        self.metrics_history['train_mae'].append(tmetrics['mae'])
        self.metrics_history['val_mae'].append(vmetrics['mae'])
        self.metrics_history['train_auc'].append(tmetrics.get('roc_auc', 0))
        self.metrics_history['val_auc'].append(vmetrics.get('roc_auc', 0))
        self.metrics_history['train_precision'].append(tmetrics['precision'])
        self.metrics_history['val_precision'].append(vmetrics['precision'])
        self.metrics_history['train_recall'].append(tmetrics['recall'])
        self.metrics_history['val_recall'].append(vmetrics['recall'])
        self.metrics_history['training_time'].append(time.time() - start_time)
        self._plot_training_history()
        self._plot_feature_importance()
        self.plot_shap(data)
        model_path = self._save_model()
        print("train_model SVD completed.")
        return model_path

    def recommend_tracks(self, artist_id, top_n=5):
        if artist_id not in self.label_encoders['artist_id'].classes_:
            return self._get_popular_tracks(top_n)
        # Zbierz tracki w tym samym klastrze
        artist_tracks = list(self.track_features[self.track_features['track_id'].isin(
            [t for t in self.artist_track_map.get(artist_id, [])])]['track_id'])
        sample_track = artist_tracks[0] if artist_tracks else None
        if sample_track is not None:
            cluster = self.track_features[self.track_features['track_id'] == sample_track]['cluster'].values[0]
            candidate_tracks = self.track_features[self.track_features['cluster'] == cluster]['track_id'].unique()
        else:
            candidate_tracks = self.track_features['track_id'].unique()
            cluster = None
        predictions = []
        for track_id in candidate_tracks:
            pred = self.model.predict(artist_id, track_id)
            track_data = self.track_features[self.track_features['track_id'] == track_id]
            if track_data.empty: continue
            row = track_data.iloc[0]
            predictions.append({
                'track_id': track_id,
                'score': pred.est,
                'danceability': row['danceability'],
                'energy': row['energy'],
                'valence': row['valence'],
                'cluster': row['cluster']
            })
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:top_n]

    def recommend_similar_tracks(self, base_track_id, top_n=5):
        # Uwaga! Oparto o podobieństwo cech latentnych itemów (qi), nie cech oryginalnych
        if base_track_id not in self.label_encoders['track_id'].classes_:
            return []
        idx = list(self.label_encoders['track_id'].classes_).index(base_track_id)
        track_emb = self.model.qi[idx]  # latent vector for track
        # For all tracks, get latent embeddings
        all_vecs = self.model.qi
        # exclude self
        track_idx_all = np.arange(len(all_vecs))
        base_idx = idx
        candidate_idx = track_idx_all[track_idx_all != base_idx]
        candidate_vecs = all_vecs[candidate_idx]
        scores = cosine_similarity([track_emb], candidate_vecs)[0]
        top_idx = np.argsort(scores)[::-1][:top_n]
        # map back to track ids
        all_tracks = np.array(self.label_encoders['track_id'].classes_)[candidate_idx]
        recs = []
        for i, idxi in enumerate(top_idx):
            t_id = all_tracks[idxi]
            row = self.track_features[self.track_features['track_id'] == t_id]
            recs.append({
                'track_id': t_id,
                'score': scores[idxi],
                'danceability': row['danceability'].values[0] if not row.empty else np.nan,
                'energy': row['energy'].values[0] if not row.empty else np.nan,
                'valence': row['valence'].values[0] if not row.empty else np.nan,
                'cluster': row['cluster'].values[0] if not row.empty else '-',
            })
        return recs

    def compute_similarity_metrics(self, base_track_id, recs, metric='cosine'):
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
        if metric == 'cosine':
            sims = cosine_similarity(base_vec, rec_vecs)[0]
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
        if self.track_features is None:
            return []
        popular_tracks = self.track_features.sort_values(
            by=['danceability', 'energy'], 
            ascending=False
        ).head(top_n)
        result = []
        for _, row in popular_tracks.iterrows():
            d = row.to_dict()
            d['score'] = None
            result.append(d)
        return result

    def _setup_logging(self, experiment_name):
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

    DATA_PATH = "../../../data/ten_thousand.csv"
    EXPERIMENT_NAME = "batch_svd_experiment1"
    N_TEST_ARTISTS = 10

    recommender = BatchSVDRecommender(EXPERIMENT_NAME)
    try:
        # 1. ETAP: Preprocessing i balanced negatywne próbki, featury, klastrowanie
        data = recommender.preprocess_data_balanced(DATA_PATH, sample_size=1000)
        data = recommender.rebalance_global_positive_negative_pairs(data)

        # 2. ETAP: Inicjalizacja/tuning modelu SVD
        recommender.initialize_model(
            n_factors=32,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02
        )
        # Alternatywnie: recommender.gridsearch_model(data)

        # 3. ETAP: Trening modelu + metryki + wykresy
        model_path = recommender.train_model(data)
        recommender.logger.info(f"\nModel trained and saved to: {model_path}")

        # 4. ETAP: Ocena rekomendacji na wielu artystach
        test_artists = data['artist_id'].drop_duplicates().sample(
            min(N_TEST_ARTISTS, data['artist_id'].nunique()), random_state=42
        ).tolist()
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

            # Różnorodność (diversity)
            diversity = recommender.compute_diversity(recs)
            diversity_scores.append(diversity)

            # Podobieństwo względem top1 utworu
            if recs:
                base_track = recs[0]['track_id']
                sim_metrics = recommender.compute_similarity_metrics(base_track, recs, metric='cosine')
                sim_metrics_scores.append(sim_metrics)
                recommender.logger.info(f"Artist {artist_id} - Similarity metrics for top recs: {sim_metrics}")

        # 5. ETAP: Coverage modelu
        coverage = recommender.compute_coverage(all_recs)
        recommender.logger.info(f"Model coverage: {coverage:.3%}")

        # 6. ETAP: Rysowanie boxplotów diversity i similarity
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

        # 7. ETAP: Pomiar średniego czasu rekomendacji jednego artysty
        mean_infer_time = recommender.measure_inference_time(test_artists[0], n_trials=10)

        # 8. ETAP: Raport końcowy - eksport do pliku JSON
        report = {
            "coverage": coverage,
            "diversity_median": float(np.median([d for d in diversity_scores if not np.isnan(d)])),
            "diversity_mean": float(np.mean([d for d in diversity_scores if not np.isnan(d)])),
            "mean_cosine_similarity_median": float(np.median([m for m in mean_cosine_list if not np.isnan(m)])),
            "mean_cosine_similarity_mean": float(np.mean([m for m in mean_cosine_list if not np.isnan(m)])),
            "mean_inference_time_s": mean_infer_time
        }
        report_path = f"{recommender.output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        recommender.logger.info(f"Summary report: {report}")

        print("FINISHED!")

    except Exception as e:
        recommender.logger.error(f"Experiment failed: {str(e)}", exc_info=True)