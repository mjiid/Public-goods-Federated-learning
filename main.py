import asyncio
import logging
import os
from config.config import DATASET_NAME, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS, PROCESSING_CAPACITIES, COST_PER_UNIT, MODEL_SAVE_PATH, FAIRNESS_EPSILON, BATCH_SIZE
from data.data_handling import load_data
from training.training import federated_learning, incentivized_federated_learning
from sklearn.metrics import r2_score
from utils.utils import save_model

logging.basicConfig(level=logging.INFO)

async def main():
    try:
        (X_train, y_train), (X_test, y_test) = load_data(DATASET_NAME)
        logging.info("Data loaded successfully.")
                        
        # Normal Federated Learning
        final_model_normal = await federated_learning(
            X_train, y_train, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS, BATCH_SIZE
        )
        
        # Evaluate Normal Federated Learning
        y_pred_normal = final_model_normal.predict(X_test).flatten()
        r2_normal = r2_score(y_test, y_pred_normal)
        logging.info(f"Normal Federated Learning R^2: {r2_normal}")

        # Save Normal Federated Learning model
        normal_model_path = os.path.join(MODEL_SAVE_PATH, 'normal_model')
        save_model(final_model_normal, normal_model_path)
        logging.info(f"Normal Federated Learning model saved at {normal_model_path}")
        
        # Incentivized Federated Learning
        final_model_incentivized, compensations = await incentivized_federated_learning(
            X_train, y_train, NUM_ORGANIZATIONS, TOTAL_TRAINING_TIME, LOCAL_EPOCHS,
            PROCESSING_CAPACITIES, COST_PER_UNIT, BATCH_SIZE, epsilon=FAIRNESS_EPSILON
        )
        
        # Evaluate Incentivized Federated Learning
        y_pred_incentivized = final_model_incentivized.predict(X_test).flatten()
        r2_incentivized = r2_score(y_test, y_pred_incentivized)
        logging.info(f"Incentivized Federated Learning R^2: {r2_incentivized}")
        
        # Save Incentivized Federated Learning model
        incentivized_model_path = os.path.join(MODEL_SAVE_PATH, 'incentivized_model')
        save_model(final_model_incentivized, incentivized_model_path)
        logging.info(f"Incentivized Federated Learning model saved at {incentivized_model_path}")
        
        # Print final compensations
        logging.info(f"Final compensations: {compensations}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
