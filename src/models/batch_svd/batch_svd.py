import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import json
from datetime import datetime
from functools import partial
import warnings

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pingouin as pg

# Konfiguracja
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set_palette('husl')

# 1. Przygotowanie danych
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    
    # Symulacja danych użytkowników
    np.random.seed(42)
    data['user_id'] = np.random.randint(0, 1000, size=len(data))
    
    # Dodanie kolumny genre na podstawie artist_name (uproszczone)
    genre_map = {
        'Pedro Rivera': 'Regional Mexican',
        'The Killers': 'Rock',
        'Fila Fresh Crew': 'Hip-Hop',
        'Larry Pierce': 'Electronic',
        'Joe Nichols': 'Country',
        'Parangolé': 'Latin',
        'Hot Tuna': 'Blues'
    }
    data['genre'] = data['artist_name'].apply(lambda x: genre_map.get(x.strip("[]'").split(',')[0].strip("'"), 'Other'))
    
    return data

# 2. Optymalizacja hiperparametrów
def optimize_hyperparameters(data):
    reader = Reader(rating_scale=(0, 100))
    surprise_data = Dataset.load_from_df(data[['user_id', 'track_id', 'popularity']], reader)
    
    space = {
        'n_factors': hp.quniform('n_factors', 50, 200, 1),
        'n_epochs': hp.quniform('n_epochs', 10, 30, 1),
        'lr_all': hp.loguniform('lr_all', np.log(0.001), np.log(0.1)),
        'reg_all': hp.loguniform('reg_all', np.log(0.001), np.log(0.1)),
        'reg_bu': hp.loguniform('reg_bu', np.log(0.001), np.log(0.1)),
        'reg_bi': hp.loguniform('reg_bi', np.log(0.001), np.log(0.1)),
        'reg_pu': hp.loguniform('reg_pu', np.log(0.001), np.log(0.1)),
        'reg_qi': hp.loguniform('reg_qi', np.log(0.001), np.log(0.1))
    }
    
    def objective(params, data):
        params['n_factors'] = int(params['n_factors'])
        params['n_epochs'] = int(params['n_epochs'])
        
        algo = SVD(
            n_factors=params['n_factors'],
            n_epochs=params['n_epochs'],
            lr_all=params['lr_all'],
            reg_all=params['reg_all'],
            reg_bu=params['reg_bu'],
            reg_bi=params['reg_bi'],
            reg_pu=params['reg_pu'],
            reg_qi=params['reg_qi']
        )
        
        kf = KFold(n_splits=5)
        results = cross_validate(algo, data, measures=['RMSE'], cv=kf, verbose=False)
        
        return {'loss': np.mean(results['test_rmse']), 'status': STATUS_OK}
    
    trials = Trials()
    best_params = fmin(
        fn=partial(objective, data=surprise_data),
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    
    # Konwersja typów dla parametrów
    best_params['n_factors'] = int(best_params['n_factors'])
    best_params['n_epochs'] = int(best_params['n_epochs'])
    
    return best_params, surprise_data

# 3. Klasteryzacja hierarchiczna
def perform_clustering(data):
    audio_features = data[['acousticness', 'danceability', 'energy', 'instrumentalness', 
                          'liveness', 'speechiness', 'valence']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(audio_features)
    
    # Wybór optymalnej liczby klastrów
    best_n_clusters = 0
    best_score = -1
    silhouette_scores = []
    
    for n_clusters in range(2, 6):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(scaled_features)
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
    
    # Finalna klasteryzacja
    final_clusterer = AgglomerativeClustering(n_clusters=best_n_clusters)
    data['audio_cluster'] = final_clusterer.fit_predict(scaled_features)
    
    # Wizualizacja wyników klasteryzacji
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(2, 6), y=silhouette_scores, marker='o')
    plt.title('Wartości współczynnika silhouette dla różnych liczb klastrów')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Średni współczynnik silhouette')
    
    return data, best_n_clusters

# 4. Trenowanie modelu z callbackami
def train_model(params, data):
    algo = SVD(
        n_factors=params['n_factors'],
        n_epochs=params['n_epochs'],
        lr_all=params['lr_all'],
        reg_all=params['reg_all'],
        reg_bu=params['reg_bu'],
        reg_bi=params['reg_bi'],
        reg_pu=params['reg_pu'],
        reg_qi=params['reg_qi']
    )
    
    # Implementacja ReduceLROnPlateau
    class Callback:
        def __init__(self):
            self.losses = []
            self.lr_history = []
        
        def __call__(self, epoch, algo):
            # Symulacja obliczania straty - w rzeczywistości potrzebny by był zestaw walidacyjny
            current_loss = np.random.random()  # W prawdziwym kodzie należy obliczyć rzeczywistą stratę
            self.losses.append(current_loss)
            self.lr_history.append(algo.lr_all)
            
            if len(self.losses) > 5:
                if min(self.losses[-5:]) > min(self.losses[:-5]):
                    algo.lr_all *= 0.5  # Redukcja LR
    
    callback = Callback()
    trainset = data.build_full_trainset()
    algo.fit(trainset, callbacks=[callback])
    
    return algo, callback

# 5. Ocena modelu i analiza strat
def evaluate_model(algo, data, original_data):
    # Walidacja krzyżowa
    kf = KFold(n_splits=5)
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=kf, verbose=False)
    
    # Przewidywania
    testset = data.build_full_trainset().build_testset()
    predictions = algo.test(testset)
    
    # Przygotowanie danych do analizy
    pred_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    pred_df = pred_df.merge(original_data, left_on='iid', right_on='track_id')
    pred_df['error'] = abs(pred_df['r_ui'] - pred_df['est'])
    
    # Analiza strat w grupach
    def analyze_group_losses(df, group_col):
        group_loss = df.groupby(group_col)['error'].agg(['mean', 'std', 'count'])
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=group_col, y='error', data=df)
        plt.title(f'Rozkład błędów wg {group_col}')
        return group_loss
    
    # Analiza dla wykonawców i gatunków
    artist_loss = analyze_group_losses(pred_df, 'artist_name')
    genre_loss = analyze_group_losses(pred_df, 'genre')
    cluster_loss = analyze_group_losses(pred_df, 'audio_cluster')
    
    # Testy statystyczne
    anova_results = pg.anova(data=pred_df, dv='error', between=['genre', 'audio_cluster'])
    
    return {
        'cv_results': cv_results,
        'artist_loss': artist_loss,
        'genre_loss': genre_loss,
        'cluster_loss': cluster_loss,
        'anova': anova_results,
        'predictions': pred_df
    }

# 6. Funkcyjna strategia rekomendacji
def hybrid_recommendation(user_id, algo, data, n_recommendations=5):
    # Rekomendacje z modelu
    user_tracks = data[data['user_id'] == user_id]['track_id'].unique()
    all_tracks = data['track_id'].unique()
    unseen_tracks = [t for t in all_tracks if t not in user_tracks]
    
    predictions = [algo.predict(user_id, track_id) for track_id in unseen_tracks]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations*3]
    
    # Filtracja przez klasteryzację i gatunki
    recommended_tracks = [pred.iid for pred in top_predictions]
    recommended_data = data[data['track_id'].isin(recommended_tracks)]
    
    # Grupowanie i wybór najlepszych z każdego klastra i gatunku
    final_recommendations = recommended_data.groupby(['audio_cluster', 'genre']).apply(
        lambda x: x.nlargest(1, 'popularity')).reset_index(drop=True)
    
    return final_recommendations.head(n_recommendations)

# 7. Zapis wyników
def save_results(model, params, evaluation, data, cluster_n):
    # Utwórz katalog na wyniki
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Zapisz model
    joblib.dump(model, os.path.join(results_dir, 'model.joblib'))
    
    # Zapisz parametry
    with open(os.path.join(results_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Zapisz metryki
    metrics = {
        'RMSE': np.mean(evaluation['cv_results']['test_rmse']),
        'MAE': np.mean(evaluation['cv_results']['test_mae']),
        'FCP': np.mean(evaluation['cv_results']['test_fcp']),
        'fit_time': np.mean(evaluation['cv_results']['fit_time']),
        'test_time': np.mean(evaluation['cv_results']['test_time'])
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Zapisz dane z predykcjami
    evaluation['predictions'].to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    
    # Zapisz wykresy
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(os.path.join(results_dir, f'plot_{i}.png'))
        plt.close()
    
    # Raport
    report = f"""
    RAPORT MODELU REKOMENDACYJNEGO
    ------------------------------
    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PARAMETRY MODELU:
    {json.dumps(params, indent=4)}
    
    METRYKI:
    - RMSE: {metrics['RMSE']:.4f}
    - MAE: {metrics['MAE']:.4f}
    - FCP: {metrics['FCP']:.4f}
    - Czas trenowania: {metrics['fit_time']:.2f}s
    - Czas testowania: {metrics['test_time']:.2f}s
    
    KLUSTERYZACJA:
    - Liczba klastrów: {cluster_n}
    - Rozkład klastrów:
    {data['audio_cluster'].value_counts().to_string()}
    
    ANALIZA STRAT:
    - Straty wg wykonawcy (top 5):
    {evaluation['artist_loss'].head(5).to_string()}
    
    - Straty wg gatunku:
    {evaluation['genre_loss'].to_string()}
    
    - Wyniki ANOVA:
    {evaluation['anova'].to_string()}
    """
    
    with open(os.path.join(results_dir, 'report.txt'), 'w') as f:
        f.write(report)
    
    return results_dir

# 8. Główna funkcja wykonawcza
def main():
    # Wczytanie i przygotowanie danych
    data = load_and_prepare_data('../../data/million_data.csv.csv')
    
    # Optymalizacja hiperparametrów
    print("Rozpoczęcie optymalizacji hiperparametrów...")
    best_params, surprise_data = optimize_hyperparameters(data)
    print("Zakończono optymalizację. Najlepsze parametry:", best_params)
    
    # Klasteryzacja hierarchiczna
    print("\nPrzeprowadzanie klasteryzacji...")
    data, best_n_clusters = perform_clustering(data)
    print(f"Znaleziono optymalną liczbę klastrów: {best_n_clusters}")
    
    # Trenowanie modelu
    print("\nTrenowanie modelu...")
    model, callback = train_model(best_params, surprise_data)
    
    # Ocena modelu
    print("\nOcena modelu...")
    evaluation = evaluate_model(model, surprise_data, data)
    
    # Przykładowe rekomendacje
    print("\nGenerowanie przykładowych rekomendacji...")
    sample_user = data['user_id'].sample(1).iloc[0]
    recommendations = hybrid_recommendation(sample_user, model, data)
    print("\nPrzykładowe rekomendacje dla użytkownika", sample_user)
    print(recommendations[['track_name', 'artist_name', 'genre', 'audio_cluster', 'popularity']])
    
    # Zapis wyników
    print("\nZapisywanie wyników...")
    results_dir = save_results(model, best_params, evaluation, data, best_n_clusters)
    print(f"\nWyniki zapisane w katalogu: {results_dir}")

if __name__ == "__main__":
    main()