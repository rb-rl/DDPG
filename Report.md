# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning agent used in this project is based on a deep deterministic policy gradient [1]. 

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

### Policy

The policy is based on using the policy network `π(s)` in combination with a noise `N`:

`a = π(s) + N`

The noise is given by an Ornstein-Uhlenbeck process [2], i.e. it is updated according to

`N <- (1 - θ) * N + ε * W`

where `W` stands for the Wiener process and is a 4-vector for each update step distributed according to a multivariate Gaussian. The constant `θ` is the decay rate of the noise and `ε` is the standard deviation of the newly added noise. In the code, the value of `ε` could basically be decayed as in epsilon decay of [DDQN](https://github.com/rb-rl/DDQN/blob/main/Report.md). However, hyperparameter optimization led to a fixation to a constant value.

### Replay memory

Also, a replay memory is used, which can store 10000 elements, where the oldest elements are discared if the limit of the memory is reached.

## Ideas for improvements

Although the environment has been solved by the present approach, there are several possible ways to make improvements. Such improvements will impact in how many episodes the average score of +30 mentioned above is reached. And they will also affect the maximum average score reachable if the training would continue indefinitely.

The suggested improvements are the following ones:
- Continued manual adjustment of the hyperparameters: A certain amount of manual hyperparameter tuning (including network topology) was invested in this project. However, the upper limit has not yet been reached here. Unfortunetly, the tweaking of the hyperparameters becomes the more time intensive, the more fine-tuned they are.
- Auto machine learning: The hyperparameters can also be tuned automatically by performing a grid search or even better a random search.
- Extension of state space by past: By using time delay neural networks or recurrent layers, the state space could be extended by the past states.
- Prioritized replay memory: The replay memory used in this project is not prioritized such that there is an improvement option.
- Distributed Distributional DDPG (=D4PG) [3]: In DDPG, every state, action pair (s,a) has only a single scalar value Q. Distributional approaches extend this by providing a distribution over multiple Q-values.
- Twin Delayed Deep Deterministic (=TD3) [3]: DDPG can be extended by using the two action-value functions in a different way, having the policy network being soft-updated at a lower rate than the deep Q-network and by introducing an extra noise in the loss of the critic.
- Attention: Primarily used in natural language processing, attention layers could also be explored in this context of this project.

### References

[1] Continuous control with deep reinforcement learning, 2015, [arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)  
[2] [en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process)  
[3] [lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
