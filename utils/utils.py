import os
import logging

logging.basicConfig(level=logging.INFO)

def save_model(model, path):
    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        model.save(path, save_format='tf')
        logging.info("Model saved successfully at %s", path)
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise
