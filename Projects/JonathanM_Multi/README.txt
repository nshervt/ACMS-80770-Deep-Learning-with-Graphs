Multi-Agent Reinforcement Learning Using Graphs
======================================

In reinforcement learning, an agent seeks to maximize the expected return of a 
reward function with respect to a specific environment or situation. 
Cooperative multi-agent reinforcement learning involves multiple agents that must 
cooperate in an environment in order to maximize their individual returns. The 
focus of this project is to explore multi-agent cooperation in an environment 
where communication between agents is essential to maximizing the teams total 
reward. The implementation involves using the TD3 algorithm combined with a graph 
neural network for each, the critic and actor networks.  While many graph 
reinforcement learning algorithms have already been proposed in order to solve 
cooperative multi-agent environments, the purpose of this project is not to 
develop a new algorithm, but to instead explore the learned patterns of 
communication and cooperation in a trained graph reinforcement learning agent. 
The results indicate that, while the agent shows evidence of learning, successful 
communication is difficult to accomplish. However, even though the communication 
does not lead to optimal decision making, the results show evidence of multi-agent 
coordination. This evidence supports the idea that successful multi-agent 
communication is possible with the general algorithm outline presented here, and 
that the shortcomings of the model might be remedied with further exploration of 
hyperparameters, state representations, etc.

## Getting Started

### Dependencies

This implementation requires:

* Python (>= 3.5)
* PyTorch (>= 1.5.0)
* NumPy (>= 1.18.1)
* Matplotlib (>= 3.1.1)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

### Data

Data samples are generated concurrently with training. 


## Run

### Training

The model is trained using `main.py`. This code does not take any arguments.
Arguments and some hyperparameters can be chosen in the main.py file itself. 

```
python3 main.py
```

to train the base model.

### Testing The Model

Testing the model can be performed by removing the randomness in action
selection within the agent class. In addition, the call to agent.learn
and agent.store can be commented out.

## Current State of the File

Currently, the learning and storing calls are commented out in main.py. 
Before uncommenting these sections, specify a directory in actor.py and 
critic.py for model saving. 



