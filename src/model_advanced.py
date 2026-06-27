# src/model_advanced.py

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os


def train_decision_tree(X_train, y_train, **kwargs):
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_mlp_sklearn(X_train, y_train, **kwargs):
    model = MLPClassifier(
        random_state=42,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


# ---------- PyTorch Model ----------
class TitanicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1)) 
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(X_train, y_train, X_val, y_val, input_dim, 
                        epochs=100, batch_size=32, lr=0.001, hidden_dims=[64, 32]):


    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
    

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    model = TitanicMLP(input_dim, hidden_dims=hidden_dims)
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            history['val_loss'].append(val_loss)
            
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
            history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return model, history


def save_advanced_model(model, filepath='../models/advanced_model.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def save_pytorch_model(model, filepath='../models/pytorch_model.pth'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"PyTorch model saved to {filepath}")


def load_pytorch_model(input_dim, filepath='../models/pytorch_model.pth', hidden_dims=[64, 32]):
    model = TitanicMLP(input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


def evaluate_advanced_model(model, X_val, y_val, model_type='sklearn'):
    if model_type == 'sklearn':
        y_pred = model.predict(X_val)
    elif model_type == 'pytorch':
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_val_t)
            y_pred = (outputs >= 0.5).float().numpy().flatten()
    else:
        raise ValueError("model_type must be 'sklearn' or 'pytorch'")
    
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
    return y_pred, acc