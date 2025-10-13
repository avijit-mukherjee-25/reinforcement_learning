import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import gymnasium as gym

class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        # State value
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        shared_out = self.shared(state)
        # Action probabilities
        # self.actor(shared_out)
        # Takes the shared hidden representation and passes it through the actor head (a linear layer)
        # Outputs raw logits (unnormalized scores) for each action
        # Shape: [batch_size, action_dim]
        # F.softmax(..., dim=-1) Converts raw logits into probabilities using the softmax function
        # dim=-1 means apply softmax along the last dimension (across actions)
        action_probs = F.softmax(self.actor(shared_out), dim=-1)
        state_value = self.critic(shared_out)
        return action_probs, state_value
    
    def get_action(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """Compute discounted returns and GAE advantages"""
        # Adds the value of the next state to handle episode boundaries
        # If episode ended, last_value = 0; otherwise it's V(s_next)
        values = self.values + [last_value]
        advantages = []
        gae = 0
        
        # Compute GAE advantages
        for t in reversed(range(len(self.rewards))):
            # Temporal Difference Error: How wrong was our value estimate?
            # TD error: r_t + γV(s_{t+1}) - V(s_t)
            # (1 - self.dones[t]) zeros out future value if episode ended
            # values[t+1] exists for all t, including the last timestep due to appending last_value to values
            # last_value is obtained in the PPOAgent.update function
            # The done flag at timestep t indicates whether the episode ended after taking action at timestep t.
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            # prepend gae
            advantages.insert(0, gae)
            # alternative approaches:
            #   1. append + reverse
            #   2. pre-allocate
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return returns, advantages

    # this function is somewhat misleading
    # it is returning all the data
    # real batching happens at _update_epoch function in PPOAgent
    
    def get_batch(self):
        return {
            'states': torch.FloatTensor(self.states),
            'actions': torch.LongTensor(self.actions),
            'log_probs': torch.FloatTensor(self.log_probs),
            'values': torch.FloatTensor(self.values),
            'rewards': torch.FloatTensor(self.rewards),
            'dones': torch.BoolTensor(self.dones)
        }
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

class PPOAgent:
    """PPO Agent"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, 
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize network and optimizer
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = PPOBuffer()
        
    def get_action(self, state):
        """Get action from current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store experience in buffer"""
        self.buffer.store(state, action, reward, log_prob, value, done)
    
    def update(self, next_state, epochs=10, batch_size=64):
        """Update policy using PPO"""
        # Get last value for bootstrap
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            _, last_value = self.policy(next_state)
            last_value = last_value.item()
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.lam)
        
        # Get batch data
        batch = self.buffer.get_batch()
        batch['returns'] = torch.FloatTensor(returns)
        batch['advantages'] = torch.FloatTensor(advantages)
        
        # Normalize advantages
        # Normalizing advantages is a critical stability technique in PPO.
        #   1. Policy gradients are extremely sensitive to the scale of advantages.
        #       Without normalization: The same relative performance differences can 
        #       cause vastly different gradient magnitudes.
        #   2. Different environments have different reward scales.
        #       Without normalization: The same learning rate would work very differently 
        #       across environments.
        batch['advantages'] = (batch['advantages'] - batch['advantages'].mean()) / \
                             (batch['advantages'].std() + 1e-8)
        
        # Update for multiple epochs
        for _ in range(epochs):
            self._update_epoch(batch, batch_size)
        
        # Clear buffer
        self.buffer.clear()
    
    def _update_epoch(self, batch, batch_size):
        """Single epoch update"""
        # This line creates a random permutation (shuffling) of all the indices in the batch. 
        indices = torch.randperm(len(batch['states']))
        
        for start_idx in range(0, len(batch['states']), batch_size):
            end_idx = min(start_idx + batch_size, len(batch['states']))
            batch_indices = indices[start_idx:end_idx]
            
            # Get mini-batch
            states = batch['states'][batch_indices]
            actions = batch['actions'][batch_indices]
            old_log_probs = batch['log_probs'][batch_indices]
            returns = batch['returns'][batch_indices]
            advantages = batch['advantages'][batch_indices]
            
            # Forward pass with CURRENT policy
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            
            # Compute policy loss
            new_log_probs = dist.log_prob(actions)
            # THIS IS THE POLICY RATIO!
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Compute entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

def train_ppo(env_name='CartPole-v1', total_timesteps=100000, update_freq=2048):
    """Train PPO agent on environment"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    state, _ = env.reset()
    episode_rewards = deque(maxlen=100)
    episode_reward = 0
    timestep = 0
    
    print(f"Training PPO on {env_name}")
    print("=" * 50)
    
    while timestep < total_timesteps:
        # Collect experience
        for _ in range(update_freq):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Handle both termination conditions
            # Store in buffer
            # log_prob is the old policy log_prob
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            
            if done:
                episode_rewards.append(episode_reward)
                state, _ = env.reset()  # Reset environment, get new episode
                episode_reward = 0
                # NO BREAK - Continue collecting!
                
                if len(episode_rewards) % 10 == 0:
                    avg_reward = np.mean(episode_rewards)
                    print(f"Timestep: {timestep:6d} | Avg Reward: {avg_reward:.2f}")
            
            if timestep >= total_timesteps:
                break   # Only break for this condition
        
        # Update policy using ALL collected experience
        if len(agent.buffer.states) > 0:
            agent.update(state) # buffer is cleared at the end of this function call
    
    env.close()
    return agent

# Example usage
if __name__ == "__main__":
    # Train the agent
    trained_agent = train_ppo('CartPole-v1', total_timesteps=50000)
    
    # Test the trained agent
    env = gym.make('CartPole-v1')
    state, _ = env.reset()
    total_reward = 0
    
    print("\nTesting trained agent...")
    for _ in range(1000):
        action, _, _ = trained_agent.get_action(state)
        state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        total_reward += reward
        
        if done:
            break
    
    print(f"Test episode reward: {total_reward}")
    env.close()