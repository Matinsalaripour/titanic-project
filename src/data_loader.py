# src/data_loader.py

import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def load_train_data():
    file_path = os.path.join(RAW_DATA_DIR, "train.csv")
    return pd.read_csv(file_path)

def load_test_data():
    file_path = os.path.join(RAW_DATA_DIR, "test.csv")
    return pd.read_csv(file_path)

def load_processed_data():
    file_path = os.path.join(BASE_DIR, "data", "processed", "train_clean.csv")
    return pd.read_csv(file_path)