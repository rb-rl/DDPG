# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning agent used in this project is based on a deep deterministic policy gradient. 

### Actor Updates

In this approach, two policies `π(s)` and `π'(s)` as well as two action-value functions `Q(s,a)` and `Q'(s,a)` are used, where `s` is the state and `a` the action. Each of these four functions is approximated by its own neural network as described further below. The first of the two policies is updated by backpropagation based on the loss

`L_actor = - Q(s,π(s))`

averaged over a mini-batch. The second policy is updated via a soft update according to

`π'(s) <- (1 - τ) * π'(s) + τ * π(s)` (1)

with the soft update rate `τ`. Note that this update is not performed every frame but only every `frames per update` frame.

### Critic Updates

The first of the two action-value functions `Q(s,a)` and `Q'(s,a)` is updated by backpropagation with the loss

`L_critic = (r + γ * max_a'Q'(s',a') - Q(s,a))^2`

averaged over also a mini-batch, where `r` is the reward when going from the current state `s` to the next state `s'` and `γ` is the discount factor.

The second action-value function is updated via a soft update according to

`Q'(s,a) <- (1 - τ) * Q'(s,a) + τ * Q(s,a)`

similar to the replacement rule (1) and with the same frequency.

### Network topology

Each of the two policies `π(s)` and `π'(s)` is represented by a fully connected neural network consisting of 2 hidden layers with 64 neurons per layer, yielding the network architecture

`33 -> 64 -> 64 -> 4`

The numbers 33 and 4 come from the sizes of the state and action spaces, because a policy network takes a state as an input and outputs an action.

For the action-value functions `Q(s,a)` and `Q'(s,a)`, the inputs are states and actions while the outputs are the scalar Q-values. As the states and actions are concatenated when entering the corresponding neural network, this yields the in- and output sizes 37 and 1. The fully connected network architectures chosen for the deep Q-networks are therefore similar to the one shown above, but now of the form

`37 -> 64 -> 64 -> 1`

The hidden layers of all architectures have the `rectified linear unit` (=relu) as the activation function. The output layers of the policy networks have a `tanh` activation function whereas the output layers of the deep Q-networks are purely linear.

### Backpropagation

Backpropagation of the neural networks behind the policy `π(s)` and action-value function `Q(s,a)` is done with mini-batch gradient descent based on the the learning rate `α=0.001` and `batch size=64`. The optimizer used for that purpose is an Adam optimizer.
