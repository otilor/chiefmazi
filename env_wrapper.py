import gymnasium as gym
import numpy as np


class LabyrinthWrapper(gym.Wrapper):
    """
    Labyrinth task from A3C paper (Section 5.4):
    - Apples: +1 reward
    - Portal: +10 reward, respawn agent, regenerate apples
    - Episode ends after 60 seconds
    """
    
    APPLE_REWARD = 1.0
    PORTAL_REWARD = 10.0
    NUM_APPLES = 10
    PICKUP_RADIUS = 1.0
    EPISODE_TIME_SECS = 60
    
    def __init__(self, env):
        super().__init__(env)
        self.env_core = self.env.unwrapped
        self.max_steps = self.EPISODE_TIME_SECS * 30
        self.step_count = 0
        self.apple_positions = []
        self.portal_position = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self._spawn_objects()
        return obs, info
    
    def _get_valid_spawn_pos(self):
        try:
            if hasattr(self.env_core, 'room'):
                room = self.env_core.room
                min_x, max_x = room.min_x + 1, room.max_x - 1
                min_z, max_z = room.min_z + 1, room.max_z - 1
            else:
                min_x, max_x = 1, 9
                min_z, max_z = 1, 9
        except Exception:
            min_x, max_x = 1, 9
            min_z, max_z = 1, 9
        
        return (np.random.uniform(min_x, max_x), np.random.uniform(min_z, max_z))
    
    def _spawn_objects(self):
        self.apple_positions = [self._get_valid_spawn_pos() for _ in range(self.NUM_APPLES)]
        self.portal_position = self._get_valid_spawn_pos()
    
    def _check_apple_pickup(self, agent_pos):
        reward = 0.0
        ax, az = agent_pos[0], agent_pos[2]
        
        for i, pos in enumerate(self.apple_positions):
            if pos is None:
                continue
            dist = np.sqrt((ax - pos[0])**2 + (az - pos[1])**2)
            if dist < self.PICKUP_RADIUS:
                reward += self.APPLE_REWARD
                self.apple_positions[i] = None
        return reward
    
    def _check_portal_entry(self, agent_pos):
        if self.portal_position is None:
            return False
        ax, az = agent_pos[0], agent_pos[2]
        px, pz = self.portal_position
        return np.sqrt((ax - px)**2 + (az - pz)**2) < self.PICKUP_RADIUS

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        try:
            agent_pos = self.env_core.agent.pos
        except Exception:
            if self.step_count >= self.max_steps:
                truncated = True
            return obs, reward, terminated, truncated, info
        
        reward += self._check_apple_pickup(agent_pos)
        
        if self._check_portal_entry(agent_pos):
            reward += self.PORTAL_REWARD
            terminated = False
            try:
                self.env_core.place_agent()
                obs = self.env_core.render()
            except Exception:
                pass
            self._spawn_objects()

        if self.step_count >= self.max_steps:
            truncated = True
        
        info['apples_remaining'] = sum(1 for a in self.apple_positions if a is not None)
        return obs, reward, terminated, truncated, info
