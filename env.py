import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os

class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=20, render_mode="rgb_array"):
        super(DroneDeliveryEnv, self).__init__()
        self.grid_size = grid_size
        self.window_size = 512
        self.num_packages = 2
        self.render_mode = render_mode
        
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
            "packages": spaces.MultiBinary(self.num_packages),
        })
        
        self.action_space = spaces.Discrete(5) 
        self._action_to_direction = {
            0: np.array([1, 0]), 1: np.array([-1, 0]), 
            2: np.array([0, 1]), 3: np.array([0, -1]), 4: np.array([0, 0]), 
        }

        self._package_locations = [np.array([3, 3]), np.array([15, 15])]
        self._depot_location = np.array([0, 0])
        self._obstacle_location = np.array([10, 10])

        self.window = None
        self.clock = None

        if self.render_mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

    def _get_obs(self):
        return {"agent": self._agent_location.copy(), "packages": self._packages_status.copy()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self._depot_location.copy()
        self._packages_status = np.ones(self.num_packages, dtype=int)
        
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}

    def step(self, action):
        if np.all(self._packages_status == 0):
            target = self._depot_location
        else:
            target = self._package_locations[0] if self._packages_status[0] == 1 else self._package_locations[1]
        
        dist_before = np.linalg.norm(self._agent_location - target)
        self._agent_location = np.clip(self._agent_location + self._action_to_direction[action], 0, self.grid_size - 1)
        dist_after = np.linalg.norm(self._agent_location - target)

        terminated = False
        reward = -1 
        
        if dist_after < dist_before: reward += 0.5
        elif dist_after > dist_before: reward -= 0.1
        
        if np.array_equal(self._agent_location, self._obstacle_location):
            reward = -500 
            terminated = True
        else:
            for i in range(self.num_packages):
                if self._packages_status[i] == 1 and np.array_equal(self._agent_location, self._package_locations[i]):
                    self._packages_status[i] = 0
                    reward = 200 
        
        if np.all(self._packages_status == 0) and np.array_equal(self._agent_location, self._depot_location):
            reward = 1000 
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_size = self.window_size / self.grid_size

        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pix_size * self._obstacle_location, (pix_size, pix_size)))
        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(pix_size * self._depot_location, (pix_size, pix_size)))

        for i in range(self.num_packages):
            if self._packages_status[i] == 1:
                pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(pix_size * self._package_locations[i], (pix_size, pix_size)))

        pygame.draw.circle(canvas, (0, 0, 0), (self._agent_location + 0.5) * pix_size, pix_size / 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
