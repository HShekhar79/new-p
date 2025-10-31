import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data(path):
    print(f"Loading cleaned data from {path} ...")
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def prepare_data(df):
    # Separate features and target
    X = df.drop(columns=['Label']) if 'Label' in df.columns else df
    y = df['Label'] if 'Label' in df.columns else None

    print("Splitting data into train and test sets (75% train, 25% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.25, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, path="models/rf_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    data_path = 'data/processed/cic_ids_cleaned.csv'
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()
