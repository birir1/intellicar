import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

MODEL_PATH = "models/saved/rf_model.pkl"

def train_predictive_model(can_data: pd.DataFrame, nlp_features: list):
    # Simple logic: merge NLP threat keywords (as dummy features)
    can_data['nlp_threat'] = [1 if i < len(nlp_features) else 0 for i in range(len(can_data))]

    # Add a dummy 'threat' label for now (manually annotate later)
    if 'threat' not in can_data.columns:
        can_data['threat'] = np.random.choice([0, 1], size=len(can_data))  # Randomly assign threat labels

    features = can_data.drop(columns=['threat'])
    labels = can_data['threat']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[✓] Model trained with accuracy: {acc:.2f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # Predict on full data
    can_data['predicted_threat'] = clf.predict(features)

    print("[✓] Model predictions:")
    print(can_data[['speed', 'rpm', 'temp', 'brake', 'predicted_threat']])

    return can_data, clf, y_test, y_pred  # Return necessary data for visualization
