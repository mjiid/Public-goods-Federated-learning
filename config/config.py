import os
import numpy as np

DATASET_NAME = "diabetes"
NUM_ORGANIZATIONS = 2
TOTAL_TRAINING_TIME = 80
LOCAL_EPOCHS = 10
BATCH_SIZE = 32
PROCESSING_CAPACITIES = [np.random.uniform(0.5, 1.5) for _ in range(NUM_ORGANIZATIONS)]
COST_PER_UNIT = 0.1
FAIRNESS_EPSILON = 0.5
MODEL_SAVE_PATH = os.path.join("saved_models", "global_model.h5")
LOGGING_LEVEL = "INFO"
