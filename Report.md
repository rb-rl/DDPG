# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning agent used in this project is based on a deep deterministic policy gradient. 

### Actor Updates

In this approach, two policies `$\pi$(s)` and `$\pi$'(s)` as well as two action-value functions `Q(s,a)` and `Q'(s,a)` are used, where `s` is the state and `a` the action. Each of these four functions is approximated by a neural network as described further below. The first of the two policies is updated by backpropagation based on the loss

`L\_actor = - Q(s,$\pi$(s))/N

where `N` is short hand for the `batch size`.

The second policy is updated via a soft update according to

`$\pi$'(s) <- (1 - tau) * $\pi$'(s) + tau * $\pi$(s)`

with the soft update rate `tau`. Note that this update is not performed every frame but only every `frames per update` frame.

### Critic Updates

The first of the two action-value functions is updated according to the rule

`Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (r + gamma * max_a'Q'(s', $\pi$'(s')))` (1)

where `alpha` is the learning rate, `r` the reward when going from state `s` to `s'` and `gamma` is the discount factor. More on the details of this update will be shown further below.

The second action-value function is updated via a soft update according to

`Q'(s,a) <- (1 - tau) * Q'(s,a) + tau * Q(s,a)`

### Network topology

Each of the two policies `$\pi$(s)` and `$\pi$'(s)` is represented by a neural network consisting of 3 hidden layers with 64 neurons per layer, yielding the network architecture

`33 -> 64 -> 64 -> 64 -> 4`

The numbers 33 and 4 come from the sizes of the state and action spaces, because a policy network takes a state as an input and outputs an action.

For the action-value functions `Q(s,a)` and `Q'(s,a)`, the inputs are states and actions while the outputs are the scalar Q-values. As the states and actions are concatenated when entering the corresponding neural network, this yields the in- and output sizes 37 and 1. The network architectures chosen for the deep Q-networks are therefore similar to the one shown above, but now of the form

`37 -> 64 -> 64 -> 64 -> 1`

The hidden layers of both architectures have the `rectified linear unit` (=relu) as the activation function. The output layer of the policy networks has the `tanh` as the activation whereas the output layers of the deep Q-networks are purely linear.

### Backpropagation

Backpropagation of all four neural networks is done with mini-batch gradient descent based on the `batch size=64`. The loss function is the mean square error loss and the optimizer the an Adam optimizer.
