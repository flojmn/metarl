import os
import gym
from envs.meta_lander import MetaLander
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import copy
from datetime import datetime

# Actor Net
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)
        self.linear3_ = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.linear3(x)

        log_std = self.linear3_(x)
        log_std = torch.clamp(log_std, min=-2, max=2)
        std = log_std.exp()

        return mean, std


# Critic Net
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        x = F.relu(self.linear1(torch.cat((s, a), dim=1)))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()


# Task
class Task(object):
    def __init__(self, num_kappa, num_train, num_test):
        self.min = 0
        self.max = 1
        self.num_kappa = num_kappa
        self.num_train = num_train
        self.num_test = num_test
        self.kappa_train, self.kappa_test = self.generate()

    def generate(self):
        kappa_train = np.random.uniform(
            self.min, self.max, (self.num_train, self.num_kappa)
        )
        kappa_test = np.random.uniform(
            self.min, self.max, (self.num_test, self.num_kappa)
        )
        return list(kappa_train), list(kappa_test)

    def sample_train(self):
        return list(self.kappa_train[np.random.choice(self.num_train)])

    def sample_test(self):
        return list(self.kappa_test[np.random.choice(self.num_test)])


# Replay Buffer
class ReplayBuffer(object):
    def __init__(self, buffersize, state_dim, action_dim):
        self.buffersize = buffersize
        self.count = 0
        self.size = 0
        self.state_buffer = np.zeros((self.buffersize, state_dim))
        self.action_buffer = np.zeros((self.buffersize, action_dim))
        self.reward_buffer = np.zeros((self.buffersize, 1))
        self.nextstate_buffer = np.zeros((self.buffersize, state_dim))
        self.done_buffer = np.zeros((self.buffersize, 1))

    def add(self, state, action, reward, next_state, done):

        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.nextstate_buffer[self.count] = next_state
        self.done_buffer[self.count] = done
        self.count = (
            self.count + 1
        ) % self.buffersize  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(
            self.size + 1, self.buffersize
        )  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling

        state_batch = self.state_buffer[index]
        action_batch = self.action_buffer[index]
        reward_batch = self.reward_buffer[index]
        next_state_batch = self.nextstate_buffer[index]
        done_batch = self.done_buffer[index]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class SAC(object):
    def __init__(self, env, params):

        # Create Task
        self.task = Task(
            params["n_kappa"], params["n_task_train"], params["n_task_test"]
        )

        # np.random.seed(0)

        # Hyperparameters
        self.max_epochs = params["max_epochs"]
        self.n_warmup = params["n_warmup"]
        self.lr_actor = params["lr_actor"]
        self.lr_critic = params["lr_critic"]
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.alpha = params["alpha"]
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch_size"]
        self.hidden_size = params["hidden_size"]
        self.n_rollouts = params["n_rollouts"]
        self.n_updates = params["n_updates"]
        self.n_kappa = params["n_kappa"]

        # State and Action Dimension
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Replay Buffer
        self.memory = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_1 = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_2 = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=self.lr_critic
        )

        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=self.lr_critic
        )

        # Small Epsilon
        self.eps = np.finfo(np.float32).eps.item()

        # Path where models are saved
        self.models_path = os.path.join(os.path.dirname(__file__), "models")

        # Path for logging files
        now = datetime.now()
        self.log_path = os.path.join(os.path.dirname(__file__), "logs", str(now))

        # Initialize Summarywriter
        self.writer = SummaryWriter(log_dir=self.log_path)

    def get_action(self, state):

        # Check if state is tensor, else transform it to a Tensor
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)

        # Forward Pass in Actor Network
        mean, std = self.actor.forward(state)

        dist = Normal(mean, std)
        action_raw = dist.rsample()
        action = F.softsign(action_raw)
        log_prob = dist.log_prob(action_raw) - torch.log(1 - action.pow(2) + self.eps)

        return action, log_prob

    def warmup(self):
        # Warmp to improve exploration at the start of training
        for i in range(self.n_warmup):

            print(f" Warmup episode: {i+1}/{self.n_warmup}")

            if i % 5 == 0:
                kappa = self.task.sample_train()
                # env = MetaLander(continuous=True, enable_wind=False)
                env = MetaLander(continuous=True, enable_wind=False, kappa=kappa)

            state, _ = env.reset()

            done = False

            for _ in range(1000):
                action = env.action_space.sample()
                observation, reward, trunc, term, _ = env.step(action)
                done = trunc or term
                self.memory.add(state, action, reward, observation, done)
                state = observation
                if done:
                    break

            env.close()

    def rollout(self, env):
        # Rollout
        state, _ = env.reset()
        done = False
        sum_reward = 0
        for _ in range(1000):
            action, _ = self.get_action(state)
            action = action.detach().numpy()
            observation, reward, trunc, term, _ = env.step(action)
            done = trunc or term
            self.memory.add(state, action, reward, observation, done)
            sum_reward += reward
            state = observation
            if done:
                break

        env.close()

        return sum_reward

    def train(self):
        # Perform Training

        countr_trainloops = 0
        countr_rolloutloops = 0

        reward_history_rollout = deque([], maxlen=50)

        for ep in range(self.max_epochs):

            if ep % 5 == 0:
                kappa = self.task.sample_train()
                env = MetaLander(continuous=True, enable_wind=False, kappa=kappa)

            print(f"ep: {ep}, kappa: {kappa}")

            # Collect Data with current policy
            sum_reward_rollout = 0
            for _ in range(self.n_rollouts):
                reward = self.rollout(env)
                sum_reward_rollout += reward
                reward_history_rollout.append(reward)

                self.writer.add_scalar(
                    "Rollout/Reward",
                    reward,
                    countr_rolloutloops,
                )

                countr_rolloutloops += 1

            # Average Reward over the last n_rollouts Rollouts
            self.writer.add_scalar(
                "Rollout/Reward Last Rollouts",
                sum_reward_rollout / self.n_rollouts,
                ep,
            )

            # Average Reward over last ... Training Rollouts
            self.writer.add_scalar(
                "Training/Avg. Reward Rollouts",
                sum(reward_history_rollout) / len(reward_history_rollout),
                ep,
            )

            print(
                f"Episode: {ep+1}, Reward last Rollouts: {sum_reward_rollout / self.n_rollouts}"
            )
            print(
                f"Episode: {ep+1}, Average Reward Rollouts: {sum(reward_history_rollout) / len(reward_history_rollout)}"
            )

            for _ in range(self.n_updates):

                # Sample a random batch
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = self.memory.sample(self.batch_size)

                # Convert batch from numpy arrays to tensors
                state_batch = torch.FloatTensor(state_batch)
                action_batch = torch.FloatTensor(action_batch)
                reward_batch = torch.FloatTensor(reward_batch).squeeze()
                next_state_batch = torch.FloatTensor(next_state_batch)
                done_batch = torch.FloatTensor(done_batch).squeeze()

                # Line 12 (https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode)
                with torch.no_grad():
                    # Action based on current policy calculated on next state
                    action_tilde_dash, log_prob_tilde_dash = self.get_action(
                        next_state_batch
                    )

                    log_prob_tilde_dash = torch.sum(log_prob_tilde_dash, 1)

                    # Q values given next state and action based on current policy
                    Q_tar_1 = self.critic_target_1(next_state_batch, action_tilde_dash)
                    Q_tar_2 = self.critic_target_2(next_state_batch, action_tilde_dash)

                    # Target Q
                    y = reward_batch + self.gamma * (1 - done_batch) * (
                        torch.minimum(Q_tar_1, Q_tar_2)
                        - self.alpha * log_prob_tilde_dash
                    )

                # Line 13 (https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode)
                batch_Q_1 = self.critic_1(state_batch, action_batch)
                batch_Q_2 = self.critic_2(state_batch, action_batch)

                critic_loss_1 = F.mse_loss(y, batch_Q_1)
                critic_loss_2 = F.mse_loss(y, batch_Q_2)
                self.critic_1_optimizer.zero_grad()
                self.critic_2_optimizer.zero_grad()
                critic_loss_1.backward()
                critic_loss_2.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.step()

                # Line 14 (https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode)
                a_tilde_rep, log_prob_tilde_rep = self.get_action(state_batch)

                log_prob_tilde_rep = torch.sum(log_prob_tilde_rep, 1)

                current_Q_1 = self.critic_1(state_batch, a_tilde_rep)
                current_Q_2 = self.critic_2(state_batch, a_tilde_rep)

                actor_loss = -torch.mean(
                    torch.minimum(current_Q_1, current_Q_2)
                    - self.alpha * log_prob_tilde_rep
                )

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Line 15 (https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode)
                # Softly update the target networks
                for param, target_param in zip(
                    self.critic_1.parameters(), self.critic_target_1.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.critic_2.parameters(), self.critic_target_2.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                self.writer.add_scalar(
                    "Train/Actor Loss", actor_loss, countr_trainloops
                )
                self.writer.add_scalar(
                    "Train/Critic 1 Loss", critic_loss_1, countr_trainloops
                )
                self.writer.add_scalar(
                    "Train/Critic 2 Loss", critic_loss_2, countr_trainloops
                )

                countr_trainloops += 1

            if ep % 100 == 0:
                self.save()

    # def simulate(self):
    #     # Simulation
    #     state, _ = self.env.reset()
    #     sum_rewards = 0
    #     done = False
    #     while not done:
    #         action = self.get_action(state)
    #         observation, reward, trunc, term, _ = self.env.step(action)
    #         sum_rewards += reward
    #         done = trunc or term
    #         state = observation

    #     return sum_rewards

    def save(self):
        # Create Models Folder if it doesnt exist
        os.makedirs(self.models_path, exist_ok=True)

        now = datetime.now()

        # Save Nets
        torch.save(self.actor, os.path.join(self.models_path, "actor" + "_" + str(now)))
        # torch.save(
        #     self.critic_1, os.path.join(self.models_path, "critic_1" + "_" + str(now))
        # )
        # torch.save(
        #     self.critic_2, os.path.join(self.models_path, "critic_2" + "_" + str(now))
        # )

    def load(self):
        # Load Saved Models
        self.actor = torch.load(os.path.join(self.models_path, "actor"))
        self.critic = torch.load(os.path.join(self.models_path, "critic"))
        self.critic_target_1 = self.critic
        self.critic_target_2 = self.critic


def main():

    # Define Hyperparameters
    params = {
        "max_epochs": 100000,  # Maximum Training Epochs
        "n_warmup": 10000,  # Number of Warmup Steps with random policy
        "lr_actor": 3e-4,  # Actor Learning Rate
        "lr_critic": 1e-3,  # Critc Learning Rate
        "gamma": 0.99,  # Discount Factor
        "tau": 0.02,  # Target Network Update Factor
        "alpha": 0.005,  # Entropy Parameter
        "buffer_size": 10000000,  # Total Replay Buffer Size
        "batch_size": 64,  # Batch Size
        "hidden_size": 32,  # Hidden Dim of NN
        "n_rollouts": 1,  # Number ob rollout epochs before training
        "n_updates": 5,  # Number of Training Updates
    }

    # Create Environment
    # env = gym.make("Pendulum-v1", g=9.81)
    env = gym.make("LunarLander-v2", continuous=True)

    # Initialize Agent
    agent = SAC(env, params)

    # Training Warmup
    print(f'Warmp started. Number of total warmup steps: {params["n_warmup"]}')
    agent.warmup()
    print(f"Warmp finished")
    # Start Training
    print(f'Traing started. Number of training episodes: {params["max_epochs"]}')
    agent.train()


if __name__ == "__main__":
    main()
