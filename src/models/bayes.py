from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from globals import np

class BayesianRecommendationSystem:
    def __init__(self):
        # Initialize the vectorizer for text features (e.g., genres, albums, artists)
        self.vectorizer = CountVectorizer()

        # Initialize label encoder for categorical labels (e.g., user preferences)
        self.label_encoder = LabelEncoder()

        # Initialize the Multinomial Naive Bayes classifier
        self.model = MultinomialNB()

        # Keep track of whether the model is trained
        self.is_trained = False

    def train(self, X_text, y):
        """
        Train the Bayesian classifier with initial data.

        Args:
            X_text (list of str): Text features for training (e.g., genres, albums, artists).
            y (list or array): Labels for training (e.g., liked/disliked, or user ratings).
        """
        # Convert text to numerical features
        X = self.vectorizer.fit_transform(X_text)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Fit the model for the first time
        self.model.fit(X, y_encoded)
        self.is_trained = True
        print("Initial training complete.")

    def update_model(self, X_text, y):
        """
        Update the model with new incoming data (online learning).

        Args:
            X_text (list of str): Text features for new data.
            y (list or array): Labels for new data.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained initially using `train()` before updating.")

        # Convert text to numerical features using existing vocabulary
        X = self.vectorizer.transform(X_text)

        # Encode labels using existing label encoder
        y_encoded = self.label_encoder.transform(y)

        # Update the model incrementally
        self.model.partial_fit(X, y_encoded)
        print("Model updated with new data.")

    def predict(self, X_text):
        """
        Predict labels for new data.

        Args:
            X_text (list of str): Text features for prediction.

        Returns:
            list: Predicted labels.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        # Convert text to numerical features using existing vocabulary
        X = self.vectorizer.transform(X_text)

        # Predict labels
        predictions = self.model.predict(X)

        # Decode labels back to original form
        return self.label_encoder.inverse_transform(predictions)

# Example Usage
if __name__ == "__main__":
    # Example data: Song genres and user preferences
    genres = ["pop", "rock", "jazz", "pop rock", "classical", "hip hop"]
    preferences = ["like", "dislike", "like", "like", "dislike", "like"]

    # Initialize the Bayesian recommendation system
    recommender = BayesianRecommendationSystem()

    # Train the model with initial data
    recommender.train(genres, preferences)

    # Simulate new data arriving dynamically
    new_genres = ["blues", "pop", "electronic"]
    new_preferences = ["like", "like", "dislike"]

    # Update the model with new data
    recommender.update_model(new_genres, new_preferences)

    # Make predictions for new songs
    test_genres = ["classical", "hip hop", "blues", "pop"]
    predictions = recommender.predict(test_genres)

    print("Predictions:", predictions)
