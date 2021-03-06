########################################################################################################################
# A neural network.
#
# Code adaption of [1].
#
# [1] https://github.com/rb-rl/DDQN/network.py.
#
########################################################################################################################

import torch

from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import relu, tanh
from typing import Tuple

# TOPOLOGY -------------------------------------------------------------------------------------------------------------

# The activation function used in all but the last layer (for both actor and critic).
ACTIVATION_FUNCTION = "relu"

# The number of hidden layers of the actor network.
ACTOR_NUMBER_HIDDEN_LAYERS = 2

# The number of hidden neurons per layer of the actor network.
ACTOR_NUMBER_HIDDEN_NEURONS_PER_LAYER = 64

# The number of hidden layers of the critic network.
CRITIC_NUMBER_HIDDEN_LAYERS = 2

# The number of hidden neurons per layer of the critic network.
CRITIC_NUMBER_HIDDEN_NEURONS_PER_LAYER = 64

# DEVELOPMENT ----------------------------------------------------------------------------------------------------------

# Is the fast development mode activated which reloads imports?
#
# The advantage of the fast development mode is that one does not have to restart Python from scratch fear each
# development increment which makes the development faster.
FAST_DEVELOPMENT_MODE = True


class NeuralNetwork(Module):
    """
    A neural network based on fully connected layers.

    Note that the inputs are concatenated when fed into the neural network.
    """

    def __init__(self, numbers_inputs: Tuple[int, ...], number_outputs: int,
                 number_hidden_layers: int, number_hidden_neurons_per_layer: int, normalized_output: bool):
        """
        Initialize the fully connected layers.

        Args:
            numbers_inputs: The numbers of input neurons.
            number_outputs: The number of output neurons.
            number_hidden_layers: The number of hidden layers.
            number_hidden_neurons_per_layer: The number of hidden neurons per layer.
            normalized_output: Is the output normalized to [-1, 1]?
        """
        super(NeuralNetwork, self).__init__()

        NUMBER_OUTPUT_LAYERS = 1

        self.__number_layers = NUMBER_OUTPUT_LAYERS + number_hidden_layers
        self.__normalized_output = normalized_output

        number_before = sum(numbers_inputs)

        for index in range(self.__number_layers):
            if index < self.__number_layers - 1:
                number_after = number_hidden_neurons_per_layer
            else:
                number_after = number_outputs

            exec("self.__linear_" + str(index) + " = Linear(number_before, number_after)")

            number_before = number_after

    def __call__(self, *inputs: Tensor) -> Tensor:
        """
        Perform a forward propagation on the given inputs.

        Args:
            inputs: The inputs to be forward propagated after concatenation.

        Returns:
            The output resulting from the forward propagation.
        """

        activations = torch.cat(inputs, dim=1)

        for index in range(self.__number_layers):
            activations = eval("self.__linear_" + str(index) + "(activations)")
            if index < self.__number_layers - 1:
                activations = eval(ACTIVATION_FUNCTION + "(activations)")
            elif self.__normalized_output:
                activations = tanh(activations)

        return activations


class Actor(NeuralNetwork):
    """
    An actor network of the form a = pi(s), with the vector state s and vector action a.
    """

    def __init__(self, number_sensors: int, number_motors: int):
        """
        Initialize the actor network.

        Args:
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        super(Actor, self).__init__((number_sensors,), number_motors,
                                    ACTOR_NUMBER_HIDDEN_LAYERS, ACTOR_NUMBER_HIDDEN_NEURONS_PER_LAYER, True)


class Critic(NeuralNetwork):
    """
    A critic network of the form Q(s, a), with the vector state s, vector action a and scalar action-value Q.
    """

    def __init__(self, number_sensors: int, number_motors: int):
        """
        Initialize the critic network.

        Args:
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        NUMBER_OUTPUTS = 1

        super(Critic, self).__init__((number_sensors, number_motors), NUMBER_OUTPUTS,
                                     CRITIC_NUMBER_HIDDEN_LAYERS, CRITIC_NUMBER_HIDDEN_NEURONS_PER_LAYER, False)
