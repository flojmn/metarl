import os
import gym
from envs.meta_lander import MetaLander
from train import Actor
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt


def main():
    agent_file = "actor_2022-10-23 13:49:48.797131"
    kappa = [0.9, 0.4]
    agent = torch.load(os.path.join("/home/florian/metarl/src/models", agent_file))

    # Environment
    env = MetaLander(continuous=True, enable_wind=False, kappa=kappa)

    reward_lst = []
    action_lst = []

    for ep in range(10):
        state, _ = env.reset()
        done = False
        sum_reward = 0
        act_traj = []

        for _ in range(1000):
            # env.render()
            state = torch.FloatTensor(state)
            action_raw = agent.forward(state)[0].detach().numpy()
            action = np.clip(action_raw, -1.0, 1.0)
            act_traj.append(action)
            observation, reward, trunc, term, _ = env.step(action)
            sum_reward += reward
            done = trunc or term
            state = observation
            if done:
                reward_lst.append(sum_reward)
                action_lst.append(act_traj)
                print(f"Episode: {ep}, Reward: {sum_reward}")
                break

    env.close()

    # Create Plots
    plt.figure()
    for act in action_lst:
        plt.plot(np.arange(0, len(act), 1), list(list(zip(*act))[0]))
    plt.title("Actions")
    plt.grid(True, which="major")
    plt.ylim([-1.5, 1.5])
    plt.show()
    plt.savefig(
        os.path.join("/home/florian/metarl/plots", "Actions.png"),
        dpi=300,
    )


if __name__ == "__main__":
    main()
