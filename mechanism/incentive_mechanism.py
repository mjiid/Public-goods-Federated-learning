import logging

logging.basicConfig(level=logging.INFO)

def should_participate(processing_capacity, data_size, cost_per_unit, alpha, beta):
    utility = processing_capacity * data_size * alpha
    cost = processing_capacity * data_size * cost_per_unit * beta
    return utility > cost

def redistribute_compensation(utilities, compensations, epsilon):
    try:
        total_compensation = sum(compensations)
        avg_compensation = total_compensation / len(compensations)
        
        fairness_metric = max(utilities) - min(utilities)
        
        if fairness_metric > epsilon:
            for i in range(len(compensations)):
                if compensations[i] < avg_compensation:
                    compensations[i] = avg_compensation
        
        logging.info("Compensations redistributed successfully.")
        return compensations
    except Exception as e:
        logging.error(f"Error redistributing compensations: {e}")
        raise

def calculate_utility_and_cost(processing_capacity, data_size, cost_per_unit):
    try:
        utility = processing_capacity * data_size
        cost = cost_per_unit * processing_capacity * data_size
        return utility, cost
    except Exception as e:
        logging.error(f"Error calculating utility and cost: {e}")
        raise

def calculate_compensation(utilities, costs, minimum_compensation=0):
    try:
        compensations = [max(c - u, minimum_compensation) for u, c in zip(utilities, costs)]
        logging.info("Compensation calculated successfully.")
        return compensations
    except Exception as e:
        logging.error(f"Error calculating compensation: {e}")
        raise
 