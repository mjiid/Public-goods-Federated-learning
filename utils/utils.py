import os
import logging
from tensorflow import keras

logging.basicConfig(level=logging.INFO)

def save_model(model, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keras.models.save_model(model, path)  # Save the model using Keras save_model function
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file does not exist at: {path}")
        
        model = keras.models.load_model(path)  # Load the model using Keras load_model function
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
