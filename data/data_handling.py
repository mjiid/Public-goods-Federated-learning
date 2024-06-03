import numpy as np
import logging
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

def load_data(dataset_name):
    try:
        if dataset_name == "diabetes":
            data = load_diabetes()
            X = data.data
            y = data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logging.info("Diabetes dataset loaded and standardized successfully.")
            return (X_train, y_train), (X_test, y_test)
        else:
            raise ValueError("Unsupported dataset")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_data(X, y, num_splits):
    try:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        split_size = len(X) // num_splits
        data_splits = []
        for i in range(num_splits):
            start = i * split_size
            end = (i + 1) * split_size if i != num_splits - 1 else len(X)
            data_splits.append((X[start:end], y[start:end]))

        logging.info("Data successfully split into %d parts.", num_splits)
        return data_splits
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise
