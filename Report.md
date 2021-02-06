# Udacity Deep Reinforcement Learning Nanodegree - Project 2: Continuous Control

## Description of the implementation

### Method

I followed the steps suggested in the **Continuous Control** project:

1. First I studied the [Deep Deterministic Policy Gradients (DDPG) paper](https://arxiv.org/pdf/1509.02971.pdf) and the [ddpg-pendulum example](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as well as the [DeepRL-Framework](https://github.com/ShangtongZhang/DeepRL) by [Shangtong Zhang](https://github.com/ShangtongZhang) from the "Actor-Critic Methods" lesson.


2. Next I adapted the code from the [ddpg-pendulum example](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to the [Reacher-Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).
   This uses an [DDPG-Agent](https://arxiv.org/pdf/1509.02971.pdf) with [Experience Replay](https://paperswithcode.com/method/experience-replay) and [Ornstein-Uhlenbeck noise](https://arxiv.org/pdf/1702.00032.pdf).
   I decided on option 2 (**Multi-Agent Environment**), for this adjustments to the **DDPG-Function** in the [Continuous_Control.ipynb](Continuous_Control.ipynb) notebook were necessary.
   After that I was able to train the DDPG agent, but it only achieved a score less than 1.0, which is well below the requirement of +30.


3. In the next step I made small adjustments to the network architecture.
   Here the size of the hidden layer was reduced to 200 nodes and a [Batch Normlization](https://arxiv.org/pdf/1502.03167.pdf) layer was added.
   In addition, the **ReLU** activation functions have been replaced by [Leaky ReLU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7) activation functions.

   
4. Last I adjusted the hyperparameters. Here only the BATCH_SIZE was increased from 128 to 256. 


After these modifications, the DDPG agent easily achieved the required target score of **+30** over **100** consecutive epsiods. 


##### Actor Network Architecture

- Input layer with 33 nodes for the states (position, rotation, velocity, and angular velocities of the arm)
- Batch Normalization layer 
- First Fully-Connected layer with 200 nodes
- Second Fully-Connected layer with 200 nodes
- Output layer with 4 nodes for the possible actions (rotation angles of the two joints)

![](./resources/DDPG-Agent-Actor.png) 

##### Critic Network Architecture

- Input layer with 33 nodes for the states (position, rotation, velocity, and angular velocities of the arm)
- Batch Normalization layer 
- First Fully-Connected layer with 200 nodes
- Second Fully-Connected layer with 200 nodes
- Output layer with 1 node for the Q-value

![](resources/DDPG-Agent-Critic.png)  

## Training & Evaluation

I trained the DDPG aent until they got an average score of **+13** in the last **100** consecutive episodes. 
I have also adjusted the hyperparameters to get the best training results.

### Hyperparameters
The learning process can be influenced by changing the following parameters:  

Parameters for controlling the training length in the [Continuous_Control.ipynb](Continuous_Control.ipynb) file:  

|Parameter         |Value |Description|
|:-----------------|-----:|:----------|
|n_episodes        | 10000|Maximum number of training episodes|
|max_t             |  1000|Maximum number of timesteps per episode|
|print_every       |   100|Number of episodes for calculating the average score value (sliding window)|

DDPG-Agent parameters in the [ddpg_agent.py](ddpg_agent.py) file:

|Parameter                 |Value     |Description|
|:-------------------------|---------:|:----------|
|BUFFER_SIZE               |       1e5|Replay buffer size|
|BATCH_SIZE                |       256|Batch size|
|GAMMA                     |      0.99|Discount factor for expected rewards|
|TAU                       |      1e-3|Multiplicative factor for updating the target network weights|
|LR_ACTOR                  |      1e-4|Learning rate of the actor network|
|LR_CRITIC                 |      1e-3|Learning rate of the critic network|
|WEIGHT_DECAY              |         0|L2 weight decay|

### Plot of Rewards
This graph shows the rewards per episode within the training phase of the DDPG agent, as well as the moving average score.  
It illustrates that the agent was trained until an average score of at least **+30** over **100** episodes was reached.   

In this case, the Agent solved the environment after **122 episodes**.

![](./resources/Training-Result.png)


### Evaluation result 
This graph shows the rewards per episode within the evaluation of the DDPG agent over 100 episodes and the average score.
It illustrates that the agent is able to achieve an average score of about **+36.98** over **100** episodes.

![](./resources/Evaluation-Result.png)

## Ideas for Future Work

Of course there are numerous ways to improve the algorithm.  
One possibility would be to use [Proximal Policy Optimization (PPO)](https://openai.com/blog/openai-baselines-ppo/) or [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/pdf?id=SyZipzbCb), as suggested in the course.   
Another possibility would be to use [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf). 
