import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Wczytanie danych
data = pd.read_csv("../../data/full_training_data.csv")
data = data.drop(columns=['Unnamed: 0'])
data['explicit'] = data['explicit'].astype(int)

# Kodowanie kategorii
label_encoders = {}
categorical_columns = ['track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Podział na cechy i target
X = data.drop(columns=['popularity'])
y = data['popularity'] / 100.0  # Normalizacja

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja cech liczbowych
numeric_features = X_train.drop(columns=categorical_columns)
scaler = StandardScaler()
numeric_features_train = scaler.fit_transform(numeric_features)
numeric_features_test = scaler.transform(X_test.drop(columns=categorical_columns))

# Przygotowanie embeddingów
embeddings = {}
for col in categorical_columns:
    embeddings[col] = Embedding(input_dim=len(data[col].unique()), output_dim=10, name=f'{col}_embedding')

# Wejścia do modelu
inputs = {col: Input(shape=(1,), name=col) for col in categorical_columns}
numeric_input = Input(shape=(numeric_features_train.shape[1],), name='numeric_input')

# Przetwarzanie embeddingów
embedded = [Flatten()(embeddings[col](inputs[col])) for col in categorical_columns]
concat = Concatenate()([*embedded, numeric_input])

# Warstwy gęste
dense_1 = Dense(64, activation='relu')(concat)
dense_2 = Dense(32, activation='relu')(dense_1)
output_layer = Dense(1, activation='sigmoid')(dense_2)

# Model NCF
model_ncf = Model(inputs=[*inputs.values(), numeric_input], outputs=output_layer)
model_ncf.compile(optimizer='adam', loss='mse')
checkpoint = ModelCheckpoint('model_ncf_best.h5', monitor='val_loss', save_best_only=True, mode='min')

# Trenowanie modelu
history = model_ncf.fit(
    [X_train[col] for col in categorical_columns] + [numeric_features_train],
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint],
    verbose=1
)

# Wykres dokładności
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NCF Model Accuracy')
plt.legend()
plt.show()

# Funkcja rekomendacji
def generate_playlist(input_tracks, model, data, label_encoders):
    track_ids = [label_encoders['track_id'].transform([track])[0] for track in input_tracks]
    pred_scores = model.predict([np.array(track_ids)])
    sorted_indices = np.argsort(pred_scores.flatten())[::-1]
    recommended_tracks = data.iloc[sorted_indices[:10]]['track_name'].values
    return recommended_tracks

# Przykładowa playlista
example_tracks = ['0009Q7nGlWjFzSjQIo9PmK', '000EFWe0HYAaXzwGbEU3rG']
recommended_playlist = generate_playlist(example_tracks, model_ncf, data, label_encoders)
print("Recommended Playlist:", recommended_playlist)
