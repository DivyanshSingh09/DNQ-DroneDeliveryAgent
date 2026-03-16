import pygame
import sys
from env import DroneDeliveryEnv

def run_human_demo():
    env = DroneDeliveryEnv(render_mode="human")
    obs, _ = env.reset()
    done = False

    while not done:
        action = 4 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT: action = 0
                elif event.key == pygame.K_LEFT: action = 1
                elif event.key == pygame.K_DOWN: action = 2
                elif event.key == pygame.KEY_UP: action = 3
                elif event.key == pygame.K_SPACE: action = 4
                
                obs, reward, terminated, truncated, _ = env.step(action)
                print(f"Action: {action} | Reward: {reward}")
                
                if terminated or truncated:
                    print("Mission Over")
                    done = True

    env.close()

if __name__ == "__main__":
    run_human_demo()
