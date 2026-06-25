# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Model Training
def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    # Train a classifier on the given data
    # Parameters : X_train , y_train
    # Model_type = str - 'Logistic', 'random_forest', or 'xgboost'
    # Kwargs = additional parameters to pass to the model constructor

    # Returns : model = trained sklearn-compatible model

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == 'xgboost':
        model = XGBClassifier(random_state = 42, use_label_encoder=False, eval_metric= 'logloss', **kwargs)
    else:
        raise ValueError("model_type must be 'logistic', 'random_forest', or 'xgboost' ")
    
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_val, y_val):
    # Print: Accuracy, Classification report, and confusion matrix for validation data.

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
    return y_pred, acc

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train, model_type='random_forest', param_grid=None, cv=5):
    # Perform GridSearchCV for hyperparameter tuning.
    # Returns: best_model = best estimator from GridSearchCV
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("model_type must be 'logistic', 'random_forest', or 'xgboost'")
    
    if param_grid is None:
        # Default grids
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:  # logistic
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            }
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross_validation accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_

def save_model(model, filepath='../models/best_model.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='../models/best_model.pkl'):
    return joblib.load(filepath)