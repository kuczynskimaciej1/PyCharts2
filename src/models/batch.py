import tensorflow as tf
from tensorflow.python.keras import layers, Model

def create_audio_text_model(num_audio_features, text_embedding_dim, output_dim):
    # Audio Features Branch
    audio_input = layers.Input(shape=(num_audio_features,), name="Audio_Input")
    x_audio = layers.Dense(128, activation="relu")(audio_input)
    x_audio = layers.Dense(64, activation="relu")(x_audio)

    # Album Embeddings Branch
    album_input = layers.Input(shape=(text_embedding_dim,), name="Album_Input")
    x_album = layers.Dense(128, activation="relu")(album_input)
    x_album = layers.Dense(64, activation="relu")(x_album)

    # Artist Embeddings Branch
    artist_input = layers.Input(shape=(text_embedding_dim,), name="Artist_Input")
    x_artist = layers.Dense(128, activation="relu")(artist_input)
    x_artist = layers.Dense(64, activation="relu")(x_artist)

    # Concatenate all branches
    concatenated = layers.Concatenate(name="Concatenate")([x_audio, x_album, x_artist])

    # Fully connected layers after concatenation
    x = layers.Dense(128, activation="relu")(concatenated)
    x = layers.Dropout(0.3)(x)  # Dropout for regularization
    x = layers.Dense(64, activation="relu")(x)

    # Output Layer
    if output_dim == 1:
        # Regression Task (e.g., popularity prediction)
        output = layers.Dense(output_dim, activation="linear", name="Output")(x)
    else:
        # Classification Task (e.g., genre prediction)
        output = layers.Dense(output_dim, activation="softmax", name="Output")(x)

    # Create the model
    model = Model(inputs=[audio_input, album_input, artist_input], outputs=output)

    return model

# Example Usage
if __name__ == "__main__":
    # Model hyperparameters
    num_audio_features = 13  # Example: danceability, energy, loudness, etc.
    text_embedding_dim = 768  # Example: Text embeddings from Sentence Transformers
    output_dim = 10  # Example: 10 genres for classification

    # Create the model
    model = create_audio_text_model(num_audio_features, text_embedding_dim, output_dim)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy" if output_dim > 1 else "mse",  # Use MSE for regression tasks
        metrics=["accuracy"] if output_dim > 1 else ["mae"]
    )

    # Summary of the model
    model.summary()
