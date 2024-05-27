import logging

logging.basicConfig(level=logging.INFO)

def should_participate(processing_capacity, data_size, cost_per_unit, alpha):
    """
    Determine whether an organization should participate in a federated learning round based on cost-benefit analysis.

    Args:
    - processing_capacity (int): Processing capacity of the organization.
    - data_size (int): Size of the data processed by the organization.
    - cost_per_unit (float): Cost per unit of processing.
    - alpha (float): Utility scaling factor.

    Returns:
    - bool: True if the organization should participate, False otherwise.
    """
    utility = processing_capacity * data_size * alpha
    cost = processing_capacity * data_size * cost_per_unit
    return utility > cost


def redistribute_compensation(compensations):
    """
    
    """
    total_compensation = sum(compensations)
    avg_compensation = total_compensation / len(compensations)
    
    for i in range(len(compensations)):
        if compensations[i] < avg_compensation:
            compensations[i] = avg_compensation

    return compensations

def calculate_utility_and_cost(processing_capacity, data_size, cost_per_unit):
    try:
        utility = processing_capacity * data_size
        cost = cost_per_unit * processing_capacity
        return utility, cost
    except Exception as e:
        logging.error(f"Error calculating utility and cost: {e}")
        raise

def calculate_compensation(utilities, costs, minimum_compensation=0):
    try:
        compensations = [max(u - c, minimum_compensation) for u, c in zip(utilities, costs)]
        logging.info("Compensation calculated successfully.")
        return compensations
    except Exception as e:
        logging.error(f"Error calculating compensation: {e}")
        raise
