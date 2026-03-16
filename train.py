import numpy as np
import torch
from env import DroneDeliveryEnv
from agent import DQNAgent

def train():
    env = DroneDeliveryEnv()
    state_dim = 4
    action_dim = 5
    agent = DQNAgent(state_dim, action_dim)
    
    episodes = 500
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.learn()
            
            obs = next_obs
            score += reward
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if ep % 10 == 0:
            agent.update_target_network()
            print(f"Episode: {ep} | Score: {score:.2f} | Epsilon: {epsilon:.2f}")

    torch.save(agent.policy_net.state_dict(), "drone_model.pth")
    print("Model saved to drone_model.pth")

if __name__ == "__main__":
    train()
