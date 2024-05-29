# Federated Learning Implementation
### Overview
This project provides an implementation of **Federated Learning**, a machine learning approach that enables training models across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This implementation focuses on asynchronous federated learning to improve performance, leveraging asynchronous operations correctly, and optimizing them for parallel execution. Additionally, it incorporates an **incentive mechanism** inspired by the **public goods game** to incentivize participation and ensure fairness among the participating organizations.

### Features
- Asynchronous tasks for parallel local training.
- Efficient utilization of asynchronous operations to minimize waiting times.
- Proper aggregation of local updates for global model update.
- Dynamic incentive mechanism based on the public goods game for compensating participating organizations.
- Pre-participation check to prevent zero or negative compensation scenarios.

### Requirements
- Python 3.7 or later
- Required Python packages: numpy, asyncio, nest_asyncio

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mjiid/Public-goods-Federated-learning
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. Prepare your dataset.

2. Configure the parameters in `config/config.py` according to your dataset and system setup.

3. Run the main script:
    ```bash
    python main.py
    ```

### Incentive Mechanism: Public Goods Game
The incentive mechanism implemented in this project is inspired by the concept of the public goods game. In Federated Learning, each participating organization acts as a player in this game, contributing resources (computation power, data) to train a global model while receiving compensation based on their contribution and the costs incurred. The goal is to incentivize cooperation and ensure fairness among the players.

#### Key Components:
- **Utility Calculation**: Each organization calculates its utility based on its contribution to the global model training process.
- **Cost Calculation**: The cost for each organization represents the resources expended during local training.
- **Compensation Calculation**: Compensation is computed as the difference between utility and cost, ensuring fair rewards for contributions.
- **Dynamic Adjustment**: The system dynamically adjusts incentives to prevent scenarios where organizations receive zero or negative compensation.

### Detailed Explanation of the Incentive Mechanism

#### Utility Calculation
Utility represents the benefit an organization provides to the global model. It is calculated as:
\[ \text{Utility} = \text{processing\_capacity} \times \text{data\_size} \times \alpha \]

#### Cost Calculation
Cost represents the resources expended by an organization during local training. It is calculated as:
\[ \text{Cost} = \text{processing\_capacity} \times \text{data\_size} \times \text{cost\_per\_unit} \]

#### Compensation Calculation
Compensation is the amount rewarded to an organization after subtracting its cost from its utility. This ensures that organizations are fairly compensated for their contributions:
\[ \text{Compensation} = \max(\text{Utility} - \text{Cost}, \text{minimum\_compensation}) \]

#### Fairness Mechanism
To ensure fairness among participating organizations, a fairness metric is used. The fairness mechanism ensures that the disparity in utilities among organizations does not exceed a specified threshold (`epsilon`). If the fairness metric exceeds this threshold, compensations are adjusted accordingly.

- **Fairness Metric**: 
  \[ \text{Fairness\_metric} = \max(\text{utilities}) - \min(\text{utilities}) \]

- **Redistribute Compensations**: If the fairness metric exceeds the specified threshold (`epsilon`), compensations are adjusted to maintain balance:
  ```python
  def redistribute_compensation(utilities, costs, compensations, epsilon):
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

### Configuration:
- **DATASET_NAME**: Name of the dataset used for training.
- **NUM_ORGANIZATIONS**: Number of participating organizations.
- **TOTAL_TRAINING_TIME**: Total training time in epochs.
- **LOCAL_EPOCHS**: Number of local training epochs per round.
- **PROCESSING_CAPACITIES**: Processing capacities of organizations.
- **COST_PER_UNIT**: Cost per unit of processing.
- **MODEL_SAVE_PATH**: Path to save the trained global model.
- **FAIRNESS_EPSILON**: Fairness threshold for redistribution of compensations.