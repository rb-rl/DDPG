# ######################################################################################################################
# A reinforcement learning agent.
#
# Code adaption of [1].
#
# [1] https://github.com/rb-rl/DDQN/network.py.
#
# ######################################################################################################################

import network
import memory

if network.FAST_DEVELOPMENT_MODE:
    import importlib

    importlib.reload(network)
    importlib.reload(memory)
    print("Fast development reload: network")
    print("Fast development reload: memory")

from network import Actor, Critic, NeuralNetwork
from memory import ReplayMemory

import torch

import numpy as np
import torch.nn.functional as F

from numpy import ndarray
from pathlib import Path
from random import randint, random
from torch import Tensor
from torch.optim import Adam
from typing import Tuple

# GENERAL --------------------------------------------------------------------------------------------------------------

# The used device in {"cuda", "cpu"}.
#
# Note that if you have a GPU which requires at least CUDA 9.0, the usage of the CPU is recommended, because otherwise
# the execution might be unexpectedly slow.
DEVICE = "cpu"

# LEARNING -------------------------------------------------------------------------------------------------------------

# The learning rate.
LEARNING_RATE = 0.001

# The interval the epsilon value of epsilon greediness may be in.
EPSILON_INTERVAL = [0.2, 0.2]

# The amount of epsilon decay.
EPSILON_DECAY = 1

# Decay rate of the Ornstein-Uhlenbeck noise in [0,1], see [1], where 0 full noise conservation and 1 maximum noise
# decay.
#
# [1] en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
THETA = 0.15

# The discount factor.
GAMMA = 0.99

# The batch size.
BATCH_SIZE = 64

# The loss function.
LOSS = "mse_loss"

# The soft update rate of target deep Q-network.
TAU = 0.001

# The number of frames per update of the target deep Q-network.
FRAMES_PER_UPDATE = 4


class Agent:
    """
    An agent based on deep deterministic gradient descent.
    """

    def __init__(self, number_sensors: int, number_motors: int):
        """
        Initialize the agent.

        Args:
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        self.__number_motors = number_motors

        device_name = DEVICE if torch.cuda.is_available() else "cpu"
        print("Used device:", device_name)

        print()

        self.__device = torch.device(device_name)

        self.__policy_network = Actor(number_sensors, number_motors).to(self.__device)
        self.__policy_network_target = Actor(number_sensors, number_motors).to(self.__device)

        self.__q_network = Critic(number_sensors, number_motors).to(self.__device)
        self.__q_network_target = Critic(number_sensors, number_motors).to(self.__device)

        print("Policy Network -", self.__policy_network)
        print()
        print("Target Policy Network - Target", self.__policy_network_target)
        print()

        print("Q Network -", self.__q_network)
        print()
        print("Target Q Network - Target", self.__q_network_target)
        print()

        self.__policy_network_optimizer = Adam(self.__policy_network.parameters(), lr=LEARNING_RATE)
        self.__q_network_optimizer = Adam(self.__q_network.parameters(), lr=LEARNING_RATE)

        self.__replay_memory = ReplayMemory(self.__device)

        self.__epsilon = EPSILON_INTERVAL[1]
        self.__noise = self.__epsilon * np.random.randn(number_motors)

        self.__step = 0

    def __call__(self, state: ndarray) -> ndarray:
        """
        Let the agent act on the given state based on the policy network and an Ornstein-Uhlenbeck noise.

        See equation (7) on page 4 and Algorithm 1 on page 5 of [1].

        [1] Continuous control with deep reinforcement learning, 2015, arxiv.org/pdf/1509.02971.pdf

        Args:
            state: The current state of the agent.

        Returns:
            The selected action in [-1,1].
        """
        input = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)

        self.__policy_network.eval()

        with torch.no_grad():
            output = self.__policy_network(input)

        action = np.clip(output[0].cpu().data.numpy() + self.__noise, -1, 1)

        return action

    def learn(self, state: ndarray, action: ndarray, reward: float, next_state: ndarray, done: bool) -> \
            Tuple[float, Tuple[float, float]]:
        """
        Perform a learning step.

        Args:
            state: The current state.
            action: The action taken in the current state, where every component is in [-1, 1].
            reward: The reward obtained by going from the current to the next state.
            next_state: The next state.
            done: Is the episode done?

        Returns:
            The current epsilon value and the losses of the actor and critic.
        """
        self.__replay_memory.add(state, action, reward, next_state, done)

        batch_size = min(BATCH_SIZE, len(self.__replay_memory))
        experiences = self.__replay_memory.extract_random_experiences(batch_size)

        loss_policy_network = self.__update_policy_network(experiences[0])
        loss_q_network = self.__update_q_network(experiences)

        losses = (loss_policy_network, loss_q_network)

        self.__step += 1
        if self.__step % FRAMES_PER_UPDATE == 0:
            self.__soft_update_all()

        self.__epsilon_decay()
        self.__update_noise()

        return self.__epsilon, losses

    def save(self, path: str):
        """
        Save the neural networks of the agent.

        Args:
            path: The path to the files where the neural networks should be stored, excluded the file suffix and ending.
        """
        actor_path = Path(path + "_actor").with_suffix(".pt")
        critic_path = Path(path + "_critic").with_suffix(".pt")

        torch.save(self.__policy_network.state_dict(), actor_path)
        torch.save(self.__q_network.state_dict(), critic_path)

        print(f"Agent saved in ({actor_path}, {critic_path})")

    def load(self, path: str):
        """
        Load the neural networks of the agent.

        Note that the loading is asymmetric to the saving, for simplicity, because we do not save the epsilon value.
        Hence, it is not possible to continue training from a loaded model.

        Args:
            path: The path to the files where the neural networks should be loaded from, excluded the file suffix and
                  ending.
        """
        critic_path = Path(path + "_critic").with_suffix(".pt")
        actor_path = Path(path + "_actor").with_suffix(".pt")

        self.__q_network.load_state_dict(torch.load(critic_path))
        self.__policy_network.load_state_dict(torch.load(actor_path))

        self.__epsilon = 0

        print(f"Agent loaded from ({actor_path}, {critic_path})")

    def __epsilon_decay(self):
        """
        Perform epsilon decay.
        """
        self.__epsilon = max(EPSILON_INTERVAL[0], EPSILON_DECAY * self.__epsilon)

    def __update_noise(self):
        """
        Update the noise according to the Ornstein-Uhlenbeck process, see [1]:

            noise <- (1 - theta) * noise + epsilon * normal_distribution

        where we use the current epsilon as the standard deviation in this process.

        [1] en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
        """
        self.__noise = (1 - THETA) * self.__noise + self.__epsilon * np.random.randn(self.__number_motors)

    def __update_policy_network(self, states: Tensor) -> float:
        """
        Update the policy network.

        See Algorithm 1 on page 5 of [1]: L = -Q(s,pi(s))/N

        Note that a minus occurs above, because a loss is minimized and Q-values are maximized. N is the batch size.

        [1] Continuous control with deep reinforcement learning, 2015, arxiv.org/pdf/1509.02971.pdf

        Args:
            states: The states used for the update.

        Returns:
            The loss.
        """
        # pi(s)
        self.__policy_network.train()
        actions = self.__policy_network(states)

        # Q(s,pi(s))
        self.__q_network.eval()
        q_values = self.__q_network(states, actions)

        loss = -q_values.mean()

        self.__policy_network_optimizer.zero_grad()
        loss.backward()

        self.__policy_network_optimizer.step()

        return float(loss.cpu().data.numpy())

    def __update_q_network(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> float:
        """
        Update the deep Q-network.

        See Algorithm 1 on page 5 of [1]: L = LOSS(r' + gamma * Q'(s',pi(s')) - Q(s,a))

        [1] Continuous control with deep reinforcement learning, 2015, arxiv.org/pdf/1509.02971.pdf

        Args:
            experiences: The experiences used for the update.

        Returns:
            The loss.
        """
        states, actions, rewards, next_states, dones = experiences

        adjusted_rewards = rewards.unsqueeze(1)
        adjusted_dones = dones.unsqueeze(1).float()

        # pi'(s')
        self.__policy_network_target.eval()
        next_actions = self.__policy_network_target(next_states).detach()

        # Q'(s',pi'(s'))
        self.__q_network_target.eval()
        target_q_values = self.__q_network_target(next_states, next_actions).detach()

        # r'+gamma*Q'(s',pi'(s'))
        targets = (adjusted_rewards + (GAMMA * target_q_values * (1 - adjusted_dones))).detach()

        # Q(s,a)
        self.__q_network.train()
        q_values = self.__q_network(states, actions)

        loss = eval("F." + LOSS)(q_values, targets)

        self.__q_network_optimizer.zero_grad()
        loss.backward()

        self.__q_network_optimizer.step()

        return float(loss.cpu().data.numpy())

    @staticmethod
    def __soft_update(neural_network: NeuralNetwork, neural_network_target: NeuralNetwork):
        """
        Perform a soft update of a target neural network.

        Args:
            neural_network: The neural network used for soft-updating.
            neural_network_target: The target neural network to be soft-updated.
        """
        for parameters, parameters_target in zip(neural_network.parameters(), neural_network_target.parameters()):
            parameters_target.data.copy_((1 - TAU) * parameters_target.data + TAU * parameters.data)

    def __soft_update_all(self):
        """
        Perform a soft update of the target policy networks and deep Q-networks.
        """
        Agent.__soft_update(self.__policy_network, self.__policy_network_target)
        Agent.__soft_update(self.__q_network, self.__q_network_target)
