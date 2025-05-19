import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score

class PlayerStateClassifier:
    def __init__(self, model_type="random_forest"):
        """
        Initializes the classifier with the chosen model.
        Supported models: "random_forest" (default) or "knn".
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_mapping = {"beginner": 0, "intermediate": 1, "advanced": 2}
        self.feature_weights = None  # Feature weights will be automatically determined later

        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=3)
        else:
            raise ValueError("Unsupported model type. Choose 'random_forest' or 'knn'.")

    def load_data(self, directory="ML"):
        """
        Loads training data from CSVs in the specified directory.
        Dynamically detects multiple CSVs for the same label (if any).
        """
        self.data = pd.DataFrame()  # Initialize an empty DataFrame
        dataframes = []  # List to store DataFrames for each skill level

        # Dynamically map filenames to labels based on filename prefixes.
        for label, label_id in self.label_mapping.items():
            label_files = [f for f in os.listdir(directory) if f.startswith(label) and f.endswith(".csv")]

            if not label_files:
                print(f"No CSV files found for label '{label}' in directory '{directory}'.")
                continue

            label_dataframes = []  # Store DataFrames specific to this label
            for file in label_files:
                path = os.path.join(directory, file)
                try:
                    df = pd.read_csv(path)
                    df["Label"] = label_id  # Assign corresponding label ID
                    label_dataframes.append(df)
                    print(f"Loaded {len(df)} rows from {file} for label '{label}'.")
                except Exception as e:
                    print(f"Error loading file {file}: {e}")

            if label_dataframes:
                combined_label_data = pd.concat(label_dataframes, ignore_index=True)
                dataframes.append(combined_label_data)

        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            print(f"Data successfully loaded. Total rows: {len(self.data)}.")
        else:
            raise FileNotFoundError("No CSV files found for any labels in the specified directory.")

    def preprocess_data(self):
        """
        Prepares the dataset for training by extracting features and normalizing.
        """
        feature_columns = self.data.columns[1:-1]  # Exclude 'Frame' and 'Label'
        X = self.data[feature_columns]
        y = self.data["Label"]

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        print("Data Preprocessing Completed.")

    def train_model(self):
        """
        Trains the chosen ML model and calculates feature weights automatically.
        """
        self.model.fit(self.X_train, self.y_train)
        print("Model Trained Successfully.")

        # Assign feature importance values as feature weights (only for models that support it)
        if hasattr(self.model, "feature_importances_"):
            feature_columns = self.data.columns[1:-1]  # Feature column names
            self.feature_weights = dict(zip(feature_columns, self.model.feature_importances_))
            print("Feature weights (based on model importance):", self.feature_weights)
        else:
            print("Feature importance is not available for this model.")

        # Evaluate performance
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

    def predict(self, player_csv):
        """
        Predicts the player's state from a new CSV file.
        """
        player_data = pd.read_csv(player_csv)
        feature_columns = self.data.columns[1:-1]  # Match training features
        X_new = player_data[feature_columns]  # Extract feature columns for the new data

        # Apply weights if available
        if self.feature_weights:
            X_new = X_new.copy()  # Avoid modifying a view
            for column, weight in self.feature_weights.items():
                if column in X_new.columns:
                    # Multiply feature weights and ensure dtype compatibility
                    X_new.loc[:, column] = (X_new[column] * weight).astype(float)

        # Normalize the new player's data using the previously fitted scaler
        X_new_scaled = self.scaler.transform(X_new)

        # Predict the class
        prediction = self.model.predict(X_new_scaled)

        # Map prediction back to the respective state (e.g., 'beginner', 'intermediate', 'advanced')
        state_mapping = {v: k for k, v in self.label_mapping.items()}
        return state_mapping[prediction[0]]

    def plot_feature_importances(self):
        """
        Plots the feature importances automatically extracted after training the model.
        """
        if not self.feature_weights:
            print("No feature importance data available. Train the model first.")
            return

        # Sort features by importance
        feature_names = list(self.feature_weights.keys())
        importances = list(self.feature_weights.values())

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importances (Learned from Model)')
        plt.show()


# Usage example
classifier = PlayerStateClassifier(model_type="random_forest")
classifier.load_data("ML")
classifier.preprocess_data()
classifier.train_model()
classifier.plot_feature_importances()

# Predict a new player's state
player_state = classifier.predict("output/biomechanical_data_Inter.csv")
print(f"Predicted Player State: {player_state}")