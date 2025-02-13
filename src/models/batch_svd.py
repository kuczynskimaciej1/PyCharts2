from tensorflow.python.keras.layers import Input, Dense
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

# Normalizacja cech liczbowych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parametry modelu
n_features = X_train.shape[1]  # Liczba cech
latent_dim = 21  # Wymiar ukrytych cech (SVD)

# Budowa modelu SVD
input_layer = Input(shape=(n_features,), name='input_layer')
svd_layer = Dense(latent_dim, activation='linear', name='svd_layer')(input_layer)  # Redukcja wymiarowości
output_layer = Dense(1, activation='sigmoid', name='output_layer')(svd_layer)  # Przewidywanie popularności

model_svd = Model(inputs=input_layer, outputs=output_layer)
model_svd.compile(optimizer='adam', loss='mse')

# Callback do zapisywania modelu
checkpoint = ModelCheckpoint('model_svd_best.h5', monitor='val_loss', save_best_only=True, mode='min')

# Trening modelu
history_svd = model_svd.fit(
    X_train,
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