from models import batch_ncf, batch_svd, online_arf, online_hoeffding_tree, online_sgd
import pandas as pd

def recommend_svd() -> None:
    # Przewidywanie popularności dla wszystkich utworów
    y_pred_svd = batch_svd.model_svd.predict(batch_svd.X_test)

    # Tworzenie rankingu
    recommendations_svd = pd.DataFrame({'track_id': batch_svd.data.iloc[batch_svd.X_test.index]['track_id'], 'predicted_popularity': y_pred_svd.flatten()})
    recommendations_svd = recommendations_svd.sort_values(by='predicted_popularity', ascending=False)

    # Top-N rekomendacji
    top_n = 10
    print("Top-N rekomendacje (SVD):")
    print(recommendations_svd.head(top_n))



def recommend_ncf() -> None:
    # Przewidywanie na zbiorze testowym
    y_pred_ncf = batch_ncf.model_ncf.predict([batch_ncf.track_ids_test, batch_ncf.artist_ids_test, batch_ncf.release_ids_test, batch_ncf.numeric_features_test])

    # Tworzenie rankingu
    recommendations_ncf = pd.DataFrame({'track_id': batch_ncf.X_test['track_id'], 'predicted_popularity': y_pred_ncf.flatten()})
    recommendations_ncf = recommendations_ncf.sort_values(by='predicted_popularity', ascending=False)

    # Top-10 rekomendacji
    print("Top-10 rekomendacji (NCF):")
    print(recommendations_ncf.head(10))



def recommend_arf() -> None:
    pass



def recommend_hd() -> None:
    pass



def recommend_sgd() -> None:
    pass