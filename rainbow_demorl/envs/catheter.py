import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from catheter_vasculature_solver import RodConfig, XPBDRodSolver


class CatheterReachVectorEnv(gym.Env):
    """Vectorized 'reach a target sphere' task using the bare XPBDRodSolver.

    Success: tip within `success_threshold` (sphere radius) of target center.
    Reward: -squared_distance, + success bonus.
    """

    def __init__(self, num_envs: int, max_episode_steps: int = 200,
                 success_threshold: float = 0.05, device: str = "cuda"):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold  # sphere radius
        self.device = device

        self.config = RodConfig()

        act_dim, obs_dim = 2, 9
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(num_envs, act_dim), dtype=np.float32)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(num_envs, obs_dim), dtype=np.float32)

        self.solver = None
        self.targets = None
        self.step_count = None
        self.episode_return = None
        self._build_solver()

    def _build_solver(self):
        self.solver = XPBDRodSolver(self.config, num_envs=self.num_envs, floor_z=None)

    def _sample_targets(self):
        # reachable region: sphere around the rod's rough natural workspace
        # (rod is 1m long, root fixed near origin, gravity sags it along -Y)
        r = torch.rand(self.num_envs, device=self.device) * 0.3 + 0.1
        theta = torch.rand(self.num_envs, device=self.device) * 2 * np.pi
        phi = torch.rand(self.num_envs, device=self.device) * np.pi
        x = r * torch.sin(phi) * torch.cos(theta) + 0.5
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi) + 0.5
        return torch.stack([x, y, z], dim=-1)

    def _get_tip_positions(self):
        return self.solver.positions[:, -1, :]

    def _get_obs(self):
        tip = self._get_tip_positions()
        return torch.cat([tip, self.targets, tip - self.targets], dim=-1)

    def reset(self, *, seed=None, options=None):
        self._build_solver()
        self.targets = self._sample_targets()
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        return self._get_obs(), {}

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device)

        push = actions[:, 0] * 0.4
        rotate = actions[:, 1] * 1.0

        dt = self.config.solver.dt
        self.solver.apply_proximal_control_gpu(push, rotate, dt)
        self.solver.step(dt)

        tip = self._get_tip_positions()
        sq_dist = torch.sum((tip - self.targets) ** 2, dim=-1)
        dist = torch.sqrt(sq_dist)
        success = dist < self.success_threshold  # tip within sphere radius
        reward = -sq_dist + success.float() * 10.0

        self.episode_return += reward
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        terminated = success
        dones = terminated | truncated

        obs = self._get_obs()
        infos = {
            "success": success,
            "_final_info": dones,
            "final_info": {
                "episode": {
                    "success_once": success.float(),
                    "return": self.episode_return.clone(),
                }
            },
            "final_observation": obs,
        }
        return obs, reward, terminated, truncated, infos


def make_envs(args, run_name: str):
    env_kwargs = dict(num_envs=args.num_envs, max_episode_steps=args.num_steps)
    envs = CatheterReachVectorEnv(num_envs=args.num_envs, max_episode_steps=args.num_steps)
    eval_envs = CatheterReachVectorEnv(num_envs=args.num_eval_envs, max_episode_steps=args.num_steps)
    return envs, eval_envs, env_kwargs