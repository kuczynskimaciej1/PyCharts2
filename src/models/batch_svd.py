from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
data = pd.read_csv("../../data/full_training_data.csv")
data = data.drop(columns=['track_name', 'artist_name', 'release_name'])
data['explicit'] = data['explicit'].astype(int)

# Przygotowanie danych
artist_ids = data['artist_id'].astype('category').cat.codes.values  # Symulacja użytkowników (artystów)
track_ids = data['track_id'].astype('category').cat.codes.values
release_ids = data['release_id'].astype('category').cat.codes.values

# Podział na cechy (X) i target (y)
X = data.drop(columns=['popularity'])
y = data['popularity'] / 100.0  # Normalizacja popularności do zakresu [0, 1]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja cech liczbowych (bez kolumn tekstowych)
numeric_features = X_train.drop(columns=['track_id', 'artist_id', 'release_id'])
scaler = StandardScaler()
numeric_features_train = scaler.fit_transform(numeric_features)
numeric_features_test = scaler.transform(X_test.drop(columns=['track_id', 'artist_id', 'release_id']))

# Przygotowanie danych dla embeddingów
track_ids_train = X_train['track_id'].astype('category').cat.codes.values
artist_ids_train = X_train['artist_id'].astype('category').cat.codes.values
release_ids_train = X_train['release_id'].astype('category').cat.codes.values

track_ids_test = X_test['track_id'].astype('category').cat.codes.values
artist_ids_test = X_test['artist_id'].astype('category').cat.codes.values
release_ids_test = X_test['release_id'].astype('category').cat.codes.values

# Liczba unikalnych wartości dla każdej kolumny kategorycznej
n_tracks = len(data['track_id'].unique())
n_artists = len(data['artist_id'].unique())
n_releases = len(data['release_id'].unique())

# Wymiar embeddingów
embedding_dim = 21

# Warstwy wejściowe
track_input = Input(shape=(1,), name='track_input')
artist_input = Input(shape=(1,), name='artist_input')
release_input = Input(shape=(1,), name='release_input')
numeric_input = Input(shape=(numeric_features_train.shape[1],), name='numeric_input')

# Warstwy embeddingowe
track_embedding = Embedding(n_tracks, embedding_dim, name='track_embedding')(track_input)
artist_embedding = Embedding(n_artists, embedding_dim, name='artist_embedding')(artist_input)
release_embedding = Embedding(n_releases, embedding_dim, name='release_embedding')(release_input)

# Spłaszczenie embeddingów
track_vec = Flatten()(track_embedding)
artist_vec = Flatten()(artist_embedding)
release_vec = Flatten()(release_embedding)

# Połączenie wszystkich cech
concat = Concatenate()([track_vec, artist_vec, release_vec, numeric_input])

# Warstwy gęste
dense_1 = Dense(64, activation='relu')(concat)
dense_2 = Dense(32, activation='relu')(dense_1)
output_layer = Dense(1, activation='sigmoid')(dense_2)

# Budowa modelu
model_svd = Model(inputs=[track_input, artist_input, release_input, numeric_input], outputs=output_layer)
model_svd.compile(optimizer='adam', loss='mse')

# Callback do zapisywania modelu
checkpoint = ModelCheckpoint('model_svd_best.h5', monitor='val_loss', save_best_only=True, mode='min')

# Trening modelu
history_svd = model_svd.fit(
    [track_ids_train, artist_ids_train, release_ids_train, numeric_features_train],
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint],
    verbose=1
)

# Wykres dokładności
plt.plot(history_svd.history['loss'], label='Train Loss')
plt.plot(history_svd.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SVD Model Accuracy')
plt.legend()
plt.show()