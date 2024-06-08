import os
import logging
from tensorflow import keras

logging.basicConfig(level=logging.INFO)

def save_model(model, path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Use the recommended Keras format (.keras)
        if not path.endswith('.keras'):
            path += '.keras'
            
        model.save(path)  # Save the model using the new Keras format
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(path):
    try:
        # Ensure the path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file does not exist at: {path}")
        
        model = keras.models.load_model(path)  # Load the model using Keras load_model function
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
