"""
AI Agents and Reinforcement Learning Tutorial
This file demonstrates the implementation of AI agents using reinforcement learning.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import gym

class DQNAgent:
    """
    Deep Q-Network (DQN) agent implementation.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        
        Parameters:
        state_size: Size of the state space
        action_size: Size of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = []
        self.memory_size = 2000
        self.batch_size = 32
        
        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build the neural network model"""
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target network with weights from main network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        minibatch = np.random.choice(self.memory, self.batch_size, replace=False)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(env, agent, episodes):
    """Train the agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        score = 0
        
        for time in range(500):  # max steps per episode
            # Choose and perform action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            score += reward
            
            if done:
                # Update target network
                agent.update_target_model()
                break
        
        scores.append(score)
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
    
    return scores

def plot_training_results(scores):
    """Plot training results"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create and train agent
    agent = DQNAgent(state_size, action_size)
    scores = train_agent(env, agent, episodes=100)
    
    # Plot results
    plot_training_results(scores)
    
    # Test the trained agent
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for _ in range(1000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"\nTest Score: {total_reward}")
    env.close()

"""
Key Concepts Explained:

1. Reinforcement Learning:
   - Learning through interaction with environment
   - Agent takes actions to maximize rewards
   - Learning from experience

2. Deep Q-Learning:
   - Combines Q-learning with deep neural networks
   - Uses experience replay for stability
   - Implements target network for stable learning

3. Components:
   - State: Current situation
   - Action: Possible moves
   - Reward: Feedback from environment
   - Policy: Strategy for choosing actions
   - Value Function: Expected future rewards

4. Best Practices:
   - Experience replay
   - Target network
   - Epsilon-greedy exploration
   - Reward shaping
   - Proper hyperparameter tuning

Next Steps:
- Implementing more advanced RL algorithms
- Multi-agent systems
- Hierarchical reinforcement learning
- Inverse reinforcement learning
""" 