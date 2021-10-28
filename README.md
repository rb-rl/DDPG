# DDPG
This repository contains a deep reinforcement learning agent based on a deep deterministic policy gradient (=DDPG) used for controlling a robotic arm in a 3D Unity environment.

The DDPG is an actor-critic based on a continuous action space and thus extends the critic-only [DDQN](https://github.com/rb-rl/DDQN) based on discrete actions.

## Environment

The environment is a floor in 3D-space with a robotic arm consisting of two joints placed on it. It is based on the Unity engine and is provided by Udacity. The continuous states, continuous actions and the rewards are given as follows:

**State**

- 33 floating point values = position, rotation, velocity and angular velocity of robotic arm

**Action**

- 4 floating point values in \[-1,1\] = torque applied to joints of robotic arm

**Reward**

- +0.1 = robotic arm's hand is in goal location

The environment is episodic. The return per episode, which is the non-discounted cumulative reward, is referred to as a score. The environment is considered as solved if the score averaged over the 100 most recent episodes reaches +30.

## Demo

The repository adresses both training and inference of the agent. The training process can be observed in a Unity window, as shown in the following video.

https://user-images.githubusercontent.com/92691697/138750479-e34f19d7-2697-4c2d-bd03-494e252565bc.mp4

When the training is stopped, the actor and critic neural networks of the agent are stored in the files called agent_actor.pt and agent_critic.pt.

The files [agent_actor.pt](agent_actor.pt) and [agent_critic.pt](agent_critic.pt) provided in this repository are the neural networks of a successfully trained agent.

The application of the agent on the environment, i.e. the inference process, can also be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/138762044-c914e2e5-8d15-4026-b3b0-0813b35f2b4e.mp4

## Installation

In order to install the project provided in this repository on Windows 10, follow these steps:

- For Windows users: If you do not know whether you have a 64-bit operating system, you can use this [help](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
- Install [Anaconda](https://anaconda.cloud/installers)
- Open the Anaconda prompt and execute the following commands:
```
conda create --name drlnd python=3.6
activate drlnd

git clone https://github.com/udacity/deep-reinforcement-learning.git

cd deep-reinforcement-learning/python
```
- Remove `torch==0.4.0` in the file `requirements.txt` located in the current folder `.../python`
- Continue with the following commands:
```
pip install .
pip install keyboard
conda install pytorch=0.4.0 -c pytorch

python -m ipykernel install --user --name drlnd --display-name "drlnd"

cd ..
cd ..

git clone git@github.com:rb-rl/DDPG.git
cd DDPG
```
- Download the Udacity Unity Reacher environment matching your environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
  - [Amazon Web Services](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
- Unzip the zip file into the folder `DDPG` (for Windows (64-bit), the `Reacher.exe` in the zip file should have the relative path `DDPG\Reacher_Windows_x86_64\Reacher.exe`, and for the other environments this path should be similar, but can also be adapted as shown further below)
- For Amazon Web Services users: You have to deactivate the [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) and perform the training in headless mode. For inference, you have to activate the virtual screen and use the Linux Version above
- Start a jupyter notebook with the following command:
```
jupyter notebook
```
- Open `Main.ipynb`
- In the Jupyter notebook, select `Kernel -> Change Kernel -> drlnd`
- If you are not using Windows (64-bit), search for `UnityEnvironment("Reacher_Windows_x86_64\Reacher.exe")` in the notebook and update the path to the corresponding file of the environment you downloaded above

## Usage

In order to do the training and inference by yourself, simply open [Main.ipynb](Main.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
