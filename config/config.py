import os
import numpy as np

DATASET_NAME = "diabetes"
NUM_ORGANIZATIONS = 10 
TOTAL_TRAINING_TIME = 2000 
LOCAL_EPOCHS = 20 
PROCESSING_CAPACITIES = [np.random.uniform(0.8, 1.2) for _ in range(NUM_ORGANIZATIONS)] 
COST_PER_UNIT = 0.05
FAIRNESS_EPSILON = 0.1
MODEL_SAVE_PATH = os.path.join("saved_models", "global_model")
LOGGING_LEVEL = "INFO"
