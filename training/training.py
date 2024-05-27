import numpy as np
import logging
import asyncio
import nest_asyncio
from model.model import initialize_global_model, local_training
from mechanism.incentive_mechanism import calculate_utility_and_cost, calculate_compensation

logging.basicConfig(level=logging.INFO)

nest_asyncio.apply()

def aggregate(client_updates):
    try:
        aggregated_update = [np.zeros_like(update) for update in client_updates[0]]
        for update in client_updates:
            for i in range(len(aggregated_update)):
                aggregated_update[i] += update[i] / len(client_updates)
        return aggregated_update
    except Exception as e:
        logging.error(f"Error in aggregation: {e}")
        raise

async def async_local_training(global_weights, data, local_epochs):
    local_model = initialize_global_model()
    local_model.set_weights(global_weights)
    new_weights = await asyncio.to_thread(local_training, local_model, data, local_epochs)
    client_update = [new_w - global_w for new_w, global_w in zip(new_weights, global_weights)]
    return client_update

async def federated_learning(dataset_splits, total_time, local_epochs, processing_capacities, cost_per_unit):
    try:
        global_model = initialize_global_model()
        global_weights = global_model.get_weights()
        num_organizations = len(dataset_splits)
        rounds = int(total_time / local_epochs)
        
        utilities = [0] * num_organizations
        costs = [0] * num_organizations
        
        for r in range(rounds):
            logging.info(f"Round {r + 1}/{rounds}")
            
            tasks = [async_local_training(global_weights, data, local_epochs) for data in dataset_splits]
            client_updates = await asyncio.gather(*tasks)
            
            global_update = aggregate(client_updates)
            global_weights = [global_w + update for global_w, update in zip(global_weights, global_update)]
            global_model.set_weights(global_weights)
            
            for i, data in enumerate(dataset_splits):
                utility, cost = calculate_utility_and_cost(processing_capacities[i], len(data[0]), cost_per_unit)
                utilities[i] += utility
                costs[i] += cost

        compensations = calculate_compensation(utilities, costs)
        
        logging.info("Federated learning completed successfully.")
        return global_model, compensations
    except Exception as e:
        logging.error(f"Error during federated learning: {e}")
        raise
