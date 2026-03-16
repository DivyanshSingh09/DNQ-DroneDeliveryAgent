import numpy as np
from env import DroneDeliveryEnv

def run_baseline():
    env = DroneDeliveryEnv()
    episodes = 50
    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            agent_pos = obs['agent']
            packages = obs['packages']
            
            target = env._depot_location
            if np.any(packages == 1):
                min_dist = np.inf
                for i in range(len(env._package_locations)):
                    if packages[i] == 1:
                        dist = np.linalg.norm(agent_pos - env._package_locations[i])
                        if dist < min_dist:
                            min_dist = dist
                            target = env._package_locations[i]

            diff = target - agent_pos
            if abs(diff[0]) > abs(diff[1]):
                action = 0 if diff[0] > 0 else 1
            elif abs(diff[1]) > 0:
                action = 2 if diff[1] > 0 else 3
            else:
                action = 4

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        
        total_rewards.append(score)
    
    print(f"Baseline Average Reward: {np.mean(total_rewards)}")

if __name__ == "__main__":
    run_baseline()
