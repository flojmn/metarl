import gym
from envs.meta_lander import MetaLander
from train import SAC
import numpy as np


def main():

    # Define Hyperparameters
    params = {
        "max_epochs": 100000,  # Maximum Training Epochs
        "n_warmup": 1000,  # Number of Warmup Steps with random policy
        "lr_actor": 3e-4,  # Actor Learning Rate
        "lr_critic": 1e-3,  # Critc Learning Rate
        "gamma": 0.99,  # Discount Factor
        "tau": 0.02,  # Target Network Update Factor
        "alpha": 0.005,  # Entropy Parameter
        "buffer_size": 10000000,  # Total Replay Buffer Size
        "batch_size": 64,  # Batch Size
        "hidden_size": 128,  # Hidden Dim of NN
        "n_rollouts": 1,  # Number ob rollout epochs before training
        "n_updates": 5,  # Number of Training Updates
        "n_task_train": 1000,  # Number of Training Tasks
        "n_task_test": 100,  # Number of Test Task
        "n_kappa": 2,  # Number of Reward Function Parameter
    }

    # Environment
    dummy_env = MetaLander(continuous=True, enable_wind=False)

    # Initialize Agent
    agent = SAC(dummy_env, params)

    # Training Warmup
    print(f'Warmp started. Number of total warmup steps: {params["n_warmup"]}')
    agent.warmup()
    print(f"Warmp finished")

    # Start Training
    print(f'Traing started. Number of training episodes: {params["max_epochs"]}')
    agent.train()
    print(f"Training finished")

    # Test Loop


if __name__ == "__main__":
    main()
