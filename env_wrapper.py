import gymnasium as gym
import numpy as np
import os
import shutil

# Copy custom meshes to miniworld's mesh directory
def _setup_custom_meshes():
    try:
        import miniworld
        miniworld_dir = os.path.dirname(miniworld.__file__)
        mesh_dir = os.path.join(miniworld_dir, 'meshes')
        local_mesh_dir = os.path.join(os.path.dirname(__file__), 'meshes')
        
        if os.path.exists(local_mesh_dir):
            for mesh_file in ['apple.obj', 'portal.obj']:
                src = os.path.join(local_mesh_dir, mesh_file)
                dst = os.path.join(mesh_dir, mesh_file)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
        return True
    except Exception:
        return False

try:
    from miniworld.entity import MeshEnt, Ball
    _setup_custom_meshes()
    ENTITIES_AVAILABLE = True
except ImportError:
    ENTITIES_AVAILABLE = False


class Apple(MeshEnt if ENTITIES_AVAILABLE else object):
    """Red apple collectible."""
    def __init__(self):
        if ENTITIES_AVAILABLE:
            try:
                super().__init__(mesh_name="apple", height=0.4, static=False)
            except Exception:
                # Fallback if mesh not found
                self.__class__ = Ball
                Ball.__init__(self, color="red", size=0.4)


class Portal(MeshEnt if ENTITIES_AVAILABLE else object):
    """Purple portal ring."""
    def __init__(self):
        if ENTITIES_AVAILABLE:
            try:
                super().__init__(mesh_name="portal", height=0.6, static=False)
            except Exception:
                # Fallback if mesh not found
                self.__class__ = Ball
                Ball.__init__(self, color="purple", size=0.7)


class LabyrinthWrapper(gym.Wrapper):
    """
    Labyrinth task from A3C paper (Section 5.4):
    - Apples (red): +1 reward
    - Portal (purple ring): +10 reward, respawn agent, regenerate apples
    - Episode ends after 60 seconds
    """
    
    APPLE_REWARD = 1.0
    PORTAL_REWARD = 10.0
    NUM_APPLES = 10
    PICKUP_RADIUS = 0.8
    EPISODE_TIME_SECS = 60
    
    def __init__(self, env):
        super().__init__(env)
        self.env_core = self.env.unwrapped
        self.max_steps = self.EPISODE_TIME_SECS * 30
        self.step_count = 0
        self.apple_entities = []
        self.portal_entity = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self._spawn_objects()
        obs = self._render_with_objects()
        return obs, info
    
    def _get_spawn_bounds(self):
        try:
            if hasattr(self.env_core, 'room'):
                room = self.env_core.room
                return room.min_x + 1, room.max_x - 1, room.min_z + 1, room.max_z - 1
        except Exception:
            pass
        return 1, 9, 1, 9
    
    def _spawn_objects(self):
        self._clear_objects()
        min_x, max_x, min_z, max_z = self._get_spawn_bounds()
        
        if ENTITIES_AVAILABLE:
            for _ in range(self.NUM_APPLES):
                apple = Apple()
                apple.pos = np.array([
                    np.random.uniform(min_x, max_x),
                    0,
                    np.random.uniform(min_z, max_z)
                ])
                apple.dir = 0
                if hasattr(self.env_core, 'params') and hasattr(self.env_core, 'np_random'):
                    apple.randomize(self.env_core.params, self.env_core.np_random)
                self.env_core.entities.append(apple)
                self.apple_entities.append(apple)
            
            portal = Portal()
            portal.pos = np.array([
                np.random.uniform(min_x, max_x),
                0,
                np.random.uniform(min_z, max_z)
            ])
            portal.dir = 0
            if hasattr(self.env_core, 'params') and hasattr(self.env_core, 'np_random'):
                portal.randomize(self.env_core.params, self.env_core.np_random)
            self.env_core.entities.append(portal)
            self.portal_entity = portal
    
    def _clear_objects(self):
        for apple in self.apple_entities:
            if apple in self.env_core.entities:
                self.env_core.entities.remove(apple)
        self.apple_entities = []
        
        if self.portal_entity and self.portal_entity in self.env_core.entities:
            self.env_core.entities.remove(self.portal_entity)
        self.portal_entity = None
    
    def _render_with_objects(self):
        try:
            return self.env_core.render()
        except Exception:
            return self.env.render()
    
    def _check_apple_pickup(self, agent_pos):
        reward = 0.0
        collected = []
        
        for apple in self.apple_entities:
            dist = np.linalg.norm(agent_pos - apple.pos)
            if dist < self.PICKUP_RADIUS:
                reward += self.APPLE_REWARD
                collected.append(apple)
        
        for apple in collected:
            if apple in self.env_core.entities:
                self.env_core.entities.remove(apple)
            self.apple_entities.remove(apple)
        
        return reward
    
    def _check_portal_entry(self, agent_pos):
        if self.portal_entity is None:
            return False
        dist = np.linalg.norm(agent_pos - self.portal_entity.pos)
        return dist < self.PICKUP_RADIUS

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
            except Exception:
                pass
            self._spawn_objects()
            obs = self._render_with_objects()

        if self.step_count >= self.max_steps:
            truncated = True
        
        info['apples_remaining'] = len(self.apple_entities)
        return obs, reward, terminated, truncated, info
