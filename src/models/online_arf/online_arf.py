from river import metrics as r_metrics
from river.forest import ARFClassifier
import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import logging
import warnings
import time

warnings.filterwarnings('ignore')


class OnlineARFRecommender:

    NUMERIC_FEATURES = [
        'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
        'tempo', 'time_signature', 'valence'
    ]
    OTHER_FEATURES = ['explicit', 'artist_id', 'release_id']

    def __init__(self, experiment_name="online_arf_rec"):
        self._setup_directories(experiment_name)
        self._setup_logging(experiment_name)
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.track_features = None
        self.artist_track_map = defaultdict(set)
        self.track_id_map = {}
        self.artist_id_map = {}
        self.cluster_labels = None
        self.feature_cols = None
        self._init_metrics_history()

    def _init_metrics_history(self):
        self.metrics_history = {
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
            'train_f1': [],
            'val_f1': [],
            'training_time': [],
        }

    def _setup_directories(self, experiment_name):
        self.output_dir = f"outputs/{experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)

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

    def _save_visualization(self, fig, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.output_dir}/plots/{name}_{timestamp}.png"
        if hasattr(fig, 'savefig'):
            fig.savefig(path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        self.logger.info(f"Saved visualization: {path}")

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
                'artist_id': self.artist_id_map
            },
            'cluster_labels': self.cluster_labels,
            'metrics': self.metrics_history
        }
        joblib.dump(save_data, model_path)
        self.logger.info(f"Saved model: {model_path}")
        return model_path

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
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        # RMSE/MAE
        axes[0, 0].plot(hist['train_rmse'], label='Train RMSE')
        axes[0, 0].plot(hist['val_rmse'], label='Val RMSE')
        axes[0, 0].set_title('RMSE'); axes[0, 0].legend()
        axes[0, 1].plot(hist['train_mae'], label='Train MAE')
        axes[0, 1].plot(hist['val_mae'], label='Val MAE')
        axes[0, 1].set_title('MAE'); axes[0, 1].legend()
        # AUC
        axes[1, 0].plot(hist['train_auc'], label='Train AUC')
        axes[1, 0].plot(hist['val_auc'], label='Val AUC')
        axes[1, 0].set_title('AUC'); axes[1, 0].legend()
        # Precision, Recall
        axes[1, 1].plot(hist['train_precision'], label='Train Prec')
        axes[1, 1].plot(hist['val_precision'], label='Val Prec')
        axes[1, 1].plot(hist['train_recall'], label='Train Rec')
        axes[1, 1].plot(hist['val_recall'], label='Val Rec')
        axes[1, 1].set_title('Precision/Recall'); axes[1, 1].legend()
        axes[2, 0].plot(hist['train_f1'], label='Train F1')
        axes[2, 0].plot(hist['val_f1'], label='Val F1')
        axes[2, 0].set_title('F1 Score'); axes[2, 0].legend()
        plt.tight_layout()
        self._save_visualization(fig, "training_history")

    def _calculate_metrics(self, y_true, y_pred, y_proba=None, threshold=0.2):
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        y_true_binary = (np.array(y_true) >= threshold).astype(int)
        metrics_out = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_squared_error(y_true, y_pred),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }
        if y_proba is not None:
            metrics_out.update({
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
        return metrics_out
    
    def compute_diversity(self, recs):
        """Compute avg pairwise distance (diversity) in feature space among recommended tracks."""
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
        unique_tracks = set()
        for recs in all_recommendations:
            for rec in recs:
                unique_tracks.add(rec['track_id'])
        coverage = len(unique_tracks) / len(self.track_features)
        self.logger.info(f"Coverage: {coverage:.3%}")
        return coverage

    def measure_inference_time(self, track_id, n_trials=20):
        import time
        times = []
        for _ in range(n_trials):
            t0 = time.time()
            try:
                _ = self.recommend_similar_tracks(track_id)
            except Exception:
                continue
            t1 = time.time()
            times.append(t1 - t0)
        mean_time = np.mean(times)
        self.logger.info(f"Average inference time ({n_trials} trials): {mean_time:.4f} s")
        return mean_time

    def preprocess_data_balanced(
        self, 
        file_path, 
        sample_size=None, 
        n_clusters=6, 
        elbow_max_clusters=12,
        negatives_ratio=2,
        hard_negatives=False
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
        # ---- ADDED KLASTER HERE! -----
        self.track_features = self._apply_hierarchical_clustering(self.track_features, n_clusters)
        self.logger.info(f"Applied hierarchical clustering for track features with n_clusters={n_clusters}")

        # Pozytywy
        positive_pairs = data[['artist_id', 'track_id']].drop_duplicates().copy()
        positive_pairs['interaction'] = 1

        all_artists = data['artist_id'].unique()
        all_tracks = set(self.track_features['track_id'].unique())
        negative_samples = []
        np.random.seed(42)

        if not hard_negatives:
            # Standard negatives (losowe)
            for artist in all_artists:
                pos_tracks = set(data[data['artist_id'] == artist]['track_id'])
                neg_tracks = list(all_tracks - pos_tracks)
                n_pos = len(pos_tracks)
                neg_samples_to_draw = min(len(neg_tracks), n_pos * negatives_ratio)
                if neg_samples_to_draw == 0: continue
                sampled_negatives = np.random.choice(list(neg_tracks), size=neg_samples_to_draw, replace=False)
                for track in sampled_negatives:
                    negative_samples.append({'artist_id': artist, 'track_id': track, 'interaction': 0})
        else:
            # HARD negatives: tylko z tego samego klastru co najczęściej używany przez artystę
            cluster_lookup = dict(self.track_features[['track_id', 'cluster']].values)
            for artist in all_artists:
                pos_tracks = set(data[data['artist_id'] == artist]['track_id'])
                # Ustalamy, w których klastrach artysta ma utwory
                clusters_counts = pd.Series([cluster_lookup[t] for t in pos_tracks if t in cluster_lookup]).value_counts()
                if clusters_counts.empty: continue
                main_cluster = clusters_counts.idxmax()
                # id tracków w tym klastrze, których artysta jeszcze nie ma
                possible_neg = set(self.track_features[self.track_features['cluster'] == main_cluster]['track_id']) - pos_tracks
                n_pos = len(pos_tracks)
                neg_samples_to_draw = min(len(possible_neg), n_pos * negatives_ratio)
                if neg_samples_to_draw == 0: continue
                picked = np.random.choice(list(possible_neg), size=neg_samples_to_draw, replace=False)
                for track in picked:
                    negative_samples.append({'artist_id': artist, 'track_id': track, 'interaction': 0})

        negative_pairs = pd.DataFrame(negative_samples)
        all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        # --- Dalej cechy, mapping, scaling i cluster na all_pairs ---
        min_occurrences = 3
        value_counts_artist = all_pairs['artist_id'].value_counts()
        value_counts_track = all_pairs['track_id'].value_counts()
        to_keep_artist = value_counts_artist[value_counts_artist >= min_occurrences].index
        to_keep_track = value_counts_track[value_counts_track >= min_occurrences].index
        all_pairs = all_pairs[all_pairs['artist_id'].isin(to_keep_artist) & all_pairs['track_id'].isin(to_keep_track)]
        all_pairs = all_pairs.merge(self.track_features, how='left', on='track_id')
        self.artist_track_map = defaultdict(set)
        for _, row in positive_pairs.iterrows():
            self.artist_track_map[row['artist_id']].add(row['track_id'])

        self.label_encoders['artist_id'] = LabelEncoder().fit(all_pairs['artist_id'])
        self.label_encoders['track_id'] = LabelEncoder().fit(all_pairs['track_id'])
        scaler = MinMaxScaler()
        all_pairs[self.feature_cols] = scaler.fit_transform(all_pairs[self.feature_cols].fillna(0))
        self.scaler = scaler
        # self.plot_elbow_and_silhouette(all_pairs, max_clusters=elbow_max_clusters)
        all_pairs = self._apply_hierarchical_clustering(all_pairs, n_clusters)
        self.logger.info(f"Applied hierarchical clustering with n_clusters={n_clusters}")
        self.plot_feature_distributions(all_pairs)
        self.track_features = all_pairs[['track_id'] + self.feature_cols + ['cluster']].drop_duplicates('track_id')
        return all_pairs

    def _apply_hierarchical_clustering(self, data, n_clusters=6):
        cluster_features = data[self.feature_cols].fillna(0).values
        Z = linkage(cluster_features, method='ward')
        self.cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        data['cluster'] = self.cluster_labels
        try:
            cluster_data = data[['danceability', 'energy', 'valence', 'cluster']]
            grid = sns.pairplot(cluster_data, hue='cluster', palette='viridis')
            self._save_visualization(grid.fig, "hierarchical_clustering")
        except Exception as e:
            self.logger.error(f"Could not visualize clusters: {str(e)}")
        return data
    
    def rebalance_global_positive_negative_pairs(self, all_pairs: pd.DataFrame, random_state=42):
        """
        Zapewnia balans liczby pozytywnych i negatywnych przykładów na poziomie całego zbioru.
        """
        pos = all_pairs[all_pairs['interaction'] == 1]
        neg = all_pairs[all_pairs['interaction'] == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        if n_neg > n_pos:
            neg_balanced = neg.sample(n=n_pos, random_state=random_state)
        else:
            neg_balanced = neg

        if n_pos < n_neg:
            pos = pd.concat([pos] * int(np.ceil(n_neg / n_pos)), ignore_index=True)[:n_neg]
        all_pairs_balanced = pd.concat([pos, neg_balanced], ignore_index=True)
        all_pairs_balanced = all_pairs_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        self.logger.info(f"After rebalancing: {all_pairs_balanced['interaction'].value_counts().to_dict()}")
        return all_pairs_balanced

    def initialize_model(self, n_models=10, max_features='sqrt', lambda_value=6):
        self.model = ARFClassifier(n_models=n_models,
                                  max_features=max_features,
                                  lambda_value=lambda_value,
                                  metric=r_metrics.Accuracy())
        self.logger.info(f"Initialized ARF model: n_models={n_models}, max_features={max_features}")

    def train_model(self, data, batch_size=100, threshold=0.2):
        """
        Trening modelu ARF na batchach z pełnym monitoringiem metryk.
        Każdy batch to równa/maksymalnie równa liczba klasy 0 oraz 1 (w batchu!), co chroni przed biasem.
        Metryki liczone na bieżąco na zbiorze walidacyjnym.
        :param data: Dane wejściowe (pandas DataFrame).
        :param batch_size: Wielkość pojedynczego batcha uczenia (domyślnie 100).
        :param threshold: Próg decyzyjny dla klasyfikacji (domyślnie 0.2).
        """

        self.logger.info("Starting ARF training (balanced batches)...")
        start_time = time.time()
        # Podział na train/val z mieszaniem i zachowaniem proporcji klasy
        X = data.drop(columns=['interaction'])
        y = data['interaction']
        feature_names = X.columns.tolist()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Konwersja do numpy, żeby wygodnie dobierać batchy wg klasy
        Xy_train = X_train.copy()
        Xy_train['interaction'] = y_train.values

        # Pętle batchowe
        n_train = len(X_train)
        n_epochs = 1  # Możesz zwiększyć do kilku, jeżeli masz mały batch_size
        all_train_preds, all_train_trues = [], []

        for epoch in range(n_epochs):
            # Shuffle
            Xy_train = Xy_train.sample(frac=1.0, random_state=epoch).reset_index(drop=True)
            idx_1 = Xy_train[Xy_train['interaction'] == 1].index.tolist()
            idx_0 = Xy_train[Xy_train['interaction'] == 0].index.tolist()
            n_batches = max(len(idx_0), len(idx_1)) // (batch_size // 2) + 1

            for b in range(n_batches):
                # Wyciągnij half batchy z każdej klasy (możliwie równe)
                b_start0 = b * (batch_size // 2)
                b_end0   = b_start0 + (batch_size // 2)
                b_start1 = b * (batch_size // 2)
                b_end1   = b_start1 + (batch_size // 2)

                batch_idx_0 = idx_0[b_start0:b_end0]
                batch_idx_1 = idx_1[b_start1:b_end1]
                batch_idx = batch_idx_0 + batch_idx_1

                batch = Xy_train.iloc[batch_idx]
                batch_X = batch.drop(columns=['interaction'])
                batch_y = batch['interaction']

                batch_dicts = [
                    dict(zip(feature_names, row))
                    for row in batch_X.values
                ]

                for xi, yi in zip(batch_dicts, batch_y):
                    pred = self.model.predict_one(xi)
                    all_train_preds.append(pred if pred is not None else 0)
                    all_train_trues.append(yi)
                    self.model.learn_one(xi, yi)

                # --- Ewaluacja na walidacji po każdym batchu ---
                X_val_dicts = [
                    dict(zip(feature_names, row))
                    for row in X_val.values
                ]
                val_preds = []
                val_probas = []
                for xv in X_val_dicts:
                    pred = self.model.predict_one(xv)
                    val_preds.append(pred if pred is not None else 0)
                    proba_dict = self.model.predict_proba_one(xv)
                    val_probas.append(proba_dict.get(1, 0) if proba_dict else 0)

                # Analogicznie batch train predictions
                train_preds = np.array(all_train_preds)
                train_trues = np.array(all_train_trues)
                train_probas = train_preds.astype(float)

                # Liczenie metryk
                tmetrics = self._calculate_metrics(train_trues, train_preds, train_probas, threshold=threshold)
                vmetrics = self._calculate_metrics(y_val.values, val_preds, val_probas, threshold=threshold)
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
                self.metrics_history['train_f1'].append(tmetrics['f1'])
                self.metrics_history['val_f1'].append(vmetrics['f1'])
        self._plot_training_history()
        model_path = self._save_model()
        print("train_model ARF completed.")
        return model_path

    def recommend_similar_tracks(self, base_track_id, top_n=5):
        if base_track_id not in self.label_encoders['track_id'].classes_:
            return self._get_popular_tracks(top_n)
        idx = list(self.label_encoders['track_id'].classes_).index(base_track_id)
        row_b = self.track_features[self.track_features['track_id'] == base_track_id]
        if len(row_b)==0: return []
        cluster = row_b['cluster'].values[0]
        candidate_tracks = self.track_features[self.track_features['cluster'] == cluster]['track_id'].unique()
        recs = []
        for track_id in candidate_tracks:
            row = self.track_features[self.track_features['track_id'] == track_id]
            if row.empty: continue
            features = {}
            for f in self.feature_cols:
                features[f] = row.iloc[0][f]
            # Wersja hybrid: dodaj feature artist_id_le (dowolny lub najczęstszy z treningu)
            artist_le = 0
            if 'artist_id_le' in features:
                artist_le = features['artist_id_le']
            features['artist_id_le'] = artist_le
            score = self.model.predict_proba_one(features)
            scr = score.get(1, 0) if score else 0
            recs.append({
                'track_id': track_id,
                'score': scr,
                'danceability': row['danceability'].values[0] if not row.empty else np.nan,
                'energy': row['energy'].values[0] if not row.empty else np.nan,
                'valence': row['valence'].values[0] if not row.empty else np.nan,
                'cluster': row['cluster'].values[0] if not row.empty else '-'
            })
        recs = [x for x in recs if x['track_id']!=base_track_id]  # nie polecaj samego siebie
        recs.sort(key=lambda x: x['score'], reverse=True)
        return recs[:top_n]

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

# --- Przykładowe użycie (pipeline bliźniaczy do SVD!) ---
if __name__ == "__main__":
    DATA_PATH = "../../../data/ten_thousand.csv"
    EXPERIMENT_NAME = "online_arf_experiment1"
    N_TEST_TRACKS = 10

    recommender = OnlineARFRecommender(EXPERIMENT_NAME)

    try:
        # 1. Przetwarzanie i featury (tak samo jak w SVD)
        data = recommender.preprocess_data_balanced(DATA_PATH, sample_size=1000)
        data = recommender.rebalance_global_positive_negative_pairs(data)

        # 2. Inicjalizacja i ewentualny tuning
        recommender.initialize_model(
            n_models=12,
            max_features='sqrt',
            lambda_value=6
        )

        # 3. Trening i zapisywanie postępu metryk/krzywych uczenia
        model_path = recommender.train_model(data)
        recommender.logger.info(f"\nModel trained and saved to: {model_path}")

        # 4. Rekomendacje podobnych utworów dla N tracków
        test_tracks = data['track_id'].drop_duplicates().sample(
            min(N_TEST_TRACKS, data['track_id'].nunique()), random_state=42
        ).tolist()
        all_recs = []
        diversity_scores = []
        sim_metrics_scores = []

        for track_id in test_tracks:
            recommender.logger.info(f"\nSimilar tracks for base track_id {track_id}:")
            recs = recommender.recommend_similar_tracks(base_track_id=track_id, top_n=5)
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
                recommender.logger.info(f"Base track {track_id} - Similarity metrics for top recs: {sim_metrics}")

        # 5. Coverage
        coverage = recommender.compute_coverage(all_recs)
        recommender.logger.info(f"Model coverage: {coverage:.3%}")

        # 6. Wykresy diversity/similarity
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

        # 7. Czas inference
        mean_infer_time = recommender.measure_inference_time(test_tracks[0], n_trials=10)

        # 8. Raport
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
        recommender.logger.error(f"ARF Experiment failed: {str(e)}", exc_info=True)
           