import tensorflow as tf
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO)

def initialize_global_model():
    try:
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("Global model initialized successfully.")
        return model
    except Exception as e:
        logging.error(f"Error initializing global model: {e}")
        raise

def local_training(model, dataset, epochs=1):
    try:
        X, y = dataset
        model.fit(X, y, epochs=epochs, verbose=0)
        logging.info("Local training completed.")
        return model.get_weights()
    except Exception as e:
        logging.error(f"Error in local training: {e}")
        raise
