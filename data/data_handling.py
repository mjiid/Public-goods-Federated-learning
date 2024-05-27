import numpy as np
import logging
from tensorflow import keras

logging.basicConfig(level=logging.INFO)

def load_data(dataset_name):
    try:
        if dataset_name == "mnist":
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            logging.info("MNIST dataset loaded successfully.")
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
            end = (i + 1) * split_size
            data_splits.append((X[start:end], y[start:end]))
        logging.info("Data successfully split into %d parts.", num_splits)
        return data_splits
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise
