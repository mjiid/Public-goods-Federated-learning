import asyncio
from config.config import DATASET_NAME, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS, PROCESSING_CAPACITIES, COST_PER_UNIT, MODEL_SAVE_PATH, FAIRNESS_EPSILON
from data.data_handling import load_data, split_data
from training.training import federated_learning
from utils.utils import save_model
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    try:
        (X_train, y_train), (X_test, y_test) = load_data(DATASET_NAME)
        dataset_splits = split_data(X_train, y_train, NUM_ORGANIZATIONS)
        
        final_model, compensations = await federated_learning(dataset_splits, TOTAL_TRAINING_TIME, LOCAL_EPOCHS, PROCESSING_CAPACITIES, COST_PER_UNIT, FAIRNESS_EPSILON)
        
        save_model(final_model, MODEL_SAVE_PATH)
        logging.info("Final compensations: %s", compensations)
        final_model.evaluate(X_test, y_test)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
