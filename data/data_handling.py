import numpy as np
import logging
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def load_data(dataset_name):
    try:
        if dataset_name == "diabetes":
            diabetes = load_diabetes()
            X, y = diabetes.data, diabetes.target
            # Normalize the features
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Diabetes dataset loaded and split successfully.")
            return (X_train, y_train), (X_test, y_test)
        else:
            raise ValueError("Unsupported dataset")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_data(X, y, num_splits):
    try:
        data_splits = []
        split_size = len(X) // num_splits
        for i in range(num_splits):
            start = i * split_size
            end = (i + 1) * split_size if i != num_splits - 1 else len(X)
            data_splits.append((X[start:end], y[start:end]))
        logging.info("Data successfully split into %d parts.", num_splits)
        return data_splits
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise