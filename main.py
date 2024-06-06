import asyncio
from config.config import DATASET_NAME, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS, PROCESSING_CAPACITIES, COST_PER_UNIT, MODEL_SAVE_PATH, FAIRNESS_EPSILON, BATCH_SIZE
from data.data_handling import load_data
from training.training import federated_learning
from sklearn.metrics import r2_score
from utils.utils import save_model
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    try:
        (X_train, y_train), (X_test, y_test) = load_data(DATASET_NAME)
        logging.info("Data loaded successfully.")
                        
        final_model, compensations = await federated_learning(
            X_train, y_train, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS,
            PROCESSING_CAPACITIES, COST_PER_UNIT, BATCH_SIZE, FAIRNESS_EPSILON
        )
        
        save_model(final_model, MODEL_SAVE_PATH)
        logging.info("Final compensations: %s", compensations)

        evaluation = final_model.evaluate(X_test, y_test)
        logging.info(f"Final model evaluation: {evaluation}")

        y_pred = final_model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        logging.info(f"R-squared: {r_squared:.4f}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
