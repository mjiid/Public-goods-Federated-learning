import tensorflow as tf
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO)

def initialize_global_model():
    try:
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])
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
