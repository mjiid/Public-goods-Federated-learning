import numpy as np
import logging
import asyncio
import nest_asyncio
from model.model import initialize_global_model, local_training
from mechanism.incentive_mechanism import calculate_utility_and_cost, calculate_compensation, redistribute_compensation, should_participate

logging.basicConfig(level=logging.INFO)

nest_asyncio.apply()

def aggregate(client_updates):
    """
    Aggregate the client updates received from multiple organizations.

    Args:
    - client_updates (list): List of client updates (gradients).

    Returns:
    - aggregated_update (list): Aggregated update (gradient).
    """
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
    """
    Perform local training asynchronously for an organization.

    Args:
    - global_weights (list): Global model weights.
    - data (tuple): Training data for the organization.
    - local_epochs (int): Number of local training epochs.

    Returns:
    - client_update (list): Update (gradient) computed by the organization.
    """
    local_model = initialize_global_model()
    local_model.set_weights(global_weights)
    new_weights = await asyncio.to_thread(local_training, local_model, data, local_epochs)
    client_update = [new_w - global_w for new_w, global_w in zip(new_weights, global_weights)]
    return client_update

async def federated_learning(dataset_splits, total_time, local_epochs, processing_capacities, cost_per_unit):
    """
    Perform federated learning asynchronously.

    Args:
    - dataset_splits (list): List of data splits for each organization.
    - total_time (int): Total training time in epochs.
    - local_epochs (int): Number of local training epochs per round.
    - processing_capacities (list): Processing capacities of organizations.
    - cost_per_unit (float): Cost per unit of processing.

    Returns:
    - global_model: Global model trained with federated learning.
    - compensations (list): Compensations for each organization.
    """
    try:
        global_model = initialize_global_model()
        global_weights = global_model.get_weights()
        num_organizations = len(dataset_splits)
        rounds = int(total_time / local_epochs)
        
        utilities = [0] * num_organizations
        costs = [0] * num_organizations
        
        for r in range(rounds):
            logging.info(f"Round {r + 1}/{rounds}")

            # Pre-Participation Check
            participants = [
                i for i in range(num_organizations)
                if should_participate(processing_capacities[i], len(dataset_splits[i][0]), cost_per_unit, 1)
            ]
            
            tasks = [
                async_local_training(global_weights, dataset_splits[i], local_epochs) 
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

        compensations = calculate_compensation(utilities, costs)
        compensations = redistribute_compensation(compensations)
        
        logging.info("Federated learning completed successfully.")
        return global_model, compensations
    except Exception as e:
        logging.error(f"Error during federated learning: {e}")
        raise
