import logging

logging.basicConfig(level=logging.INFO)


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
