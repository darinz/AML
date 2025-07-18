# Reinforcement Learning

## Overview

Reinforcement Learning (RL) is a framework for sequential decision-making, where agents learn to maximize cumulative reward through interaction with an environment.

## 1. RL Fundamentals

- **Agent, Environment, Reward, Policy, Value Function**
- **Markov Decision Process (MDP)**: $(S, A, P, R, \gamma)$

### Bellman Equation
```math
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s \right]
```

## 2. Q-Learning

- **Q-Function**: $Q(s, a)$ estimates expected return for action $a$ in state $s$
- **Update Rule**:
```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
```

#### Example: Tabular Q-Learning
```python
import numpy as np
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for ep in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])
            s_, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_]) - Q[s, a])
            s = s_
    return Q
```

## 3. Policy Gradient Methods

- **REINFORCE**: Update policy parameters in direction of expected reward gradient
- **Actor-Critic**: Combine value and policy learning

## 4. Deep RL
- **DQN**: Deep Q-Networks
- **DDPG, PPO**: Policy gradient methods for continuous action spaces

## 5. Multi-Agent Systems
- Multiple agents interact, cooperate, or compete

## Applications
- Game AI (Atari, Go, Chess)
- Robotics
- Recommendation systems
- Financial trading

## Summary
- RL optimizes sequential decision-making
- Q-learning and policy gradients are core algorithms
- Deep RL scales to complex environments
- Multi-agent RL enables cooperation and competition 