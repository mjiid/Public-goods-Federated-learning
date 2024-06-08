import numpy as np
import logging
import asyncio
import nest_asyncio
from tensorflow import keras
from model.model import initialize_global_model, local_training
from mechanism.incentive_mechanism import should_participate, redistribute_compensation, calculate_utility_and_cost, calculate_compensation
from data.data_handling import split_data

logging.basicConfig(level=logging.INFO)

nest_asyncio.apply()

def aggregate(client_updates):
    try:
        aggregated_update = [np.zeros_like(update) for update in client_updates[0]]
        for update in client_updates:
            for i in range(len(aggregated_update)):
                aggregated_update[i] += update[i] / len(client_updates)
        return aggregated_update
    except (ValueError, TypeError, Exception) as e:
        logging.error(f"Error in aggregation: {e}")
        raise

async def async_local_training(global_weights, data, local_epochs, processing_capacity, batch_size=32):
    try:
        local_model = initialize_global_model()
        local_model.set_weights(global_weights)
        
        # Adjust local epochs or batch size based on processing capacity
        adjusted_epochs = int(local_epochs * processing_capacity)
        adjusted_batch_size = int(batch_size * processing_capacity)
        
        new_weights = await asyncio.to_thread(local_training, local_model, data, adjusted_epochs, adjusted_batch_size)
        return [(new_w - global_w) for new_w, global_w in zip(new_weights, global_weights)]
    except (ValueError, TypeError, Exception) as e:
        logging.error(f"Error in async local training: {e}")
        raise

async def incentivized_federated_learning(X_train, y_train, num_splits, total_time, local_epochs, processing_capacities, cost_per_unit, batch_size=32, epsilon=0.1, compensation_cap=1.0):
    try:
        global_model = initialize_global_model()
        global_weights = global_model.get_weights()
        rounds = total_time // local_epochs

        utilities = [0] * num_splits
        costs = [0] * num_splits
        total_compensations = [0] * num_splits
        
        for r in range(rounds):
            logging.info(f"Round {r + 1}/{rounds}")

            # Shuffle and split data for this round
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            dataset_splits = split_data(X_train_shuffled, y_train_shuffled, num_splits)
             
            # Pre-Participation Check with updated compensations
            participants = [
                i for i in range(num_splits)
                if should_participate(processing_capacities[i], len(dataset_splits[i][0]), cost_per_unit, total_compensations[i])
            ]
            
            tasks = [
                async_local_training(global_weights, dataset_splits[i], local_epochs, processing_capacities[i], batch_size) 
                for i in participants
            ]
            client_updates = await asyncio.gather(*tasks)
            
            global_update = aggregate(client_updates)
            global_weights = [global_w + update for global_w, update in zip(global_weights, global_update)]
            global_model.set_weights(global_weights)
            
            for i in participants:
                utility, cost = calculate_utility_and_cost(processing_capacities[i], len(dataset_splits[i][0]), cost_per_unit)
                utilities[i] += utility
                costs[i] += cost

            # Calculate and update compensations
            compensations = calculate_compensation(utilities, costs)
                        
            total_compensations = [total_compensations[i] + compensations[i] for i in range(num_splits)]
            total_compensations = redistribute_compensation(utilities, total_compensations, epsilon)

            # Adjust processing capacities based on compensations
            for i in range(num_splits):
                if i in participants:
                    processing_capacities[i] = max(processing_capacities[i] + total_compensations[i] * 0.01, 0.5)  # ensure processing capacity does not go below 0.5
                else:
                    processing_capacities[i] = max(processing_capacities[i] - 0.01, 0.5)  # decrease processing capacity for non-participants

        logging.info("Incentivized Federated learning completed successfully.")
        return global_model, total_compensations
    except (ValueError, TypeError, Exception) as e:
        logging.error(f"Error during incentivized federated learning: {e}")
        raise


async def federated_learning(X_train, y_train, num_splits, total_time, local_epochs, batch_size=32):
    try:
        global_model = initialize_global_model()
        global_weights = global_model.get_weights()
        rounds = total_time // local_epochs

        for r in range(rounds):
            logging.info(f"Round {r + 1}/{rounds}")

            # Shuffle and split data for this round
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            dataset_splits = split_data(X_train_shuffled, y_train_shuffled, num_splits)
             
            tasks = [
                async_local_training(global_weights, dataset_splits[i], local_epochs, 1, batch_size) 
                for i in range(num_splits)
            ]
            client_updates = await asyncio.gather(*tasks)
            
            global_update = aggregate(client_updates)
            global_weights = [global_w + update for global_w, update in zip(global_weights, global_update)]
            global_model.set_weights(global_weights)

        logging.info("Federated learning completed successfully.")
        return global_model
    except (ValueError, TypeError, Exception) as e:
        logging.error(f"Error during federated learning: {e}")
        raise
