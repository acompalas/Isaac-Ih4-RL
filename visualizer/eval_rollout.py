import argparse
import numpy as np
import torch

from rainbow_demorl.envs.catheter import CatheterReachVectorEnv
from rainbow_demorl.agents.td3 import TD3Agent
from rainbow_demorl.utils.common import Args

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--out", type=str, default="visualizer/eval_trajectories.npz")
cli_args = parser.parse_args()

args = Args()
args.env_id = "CatheterReach-v0"
args.checkpoint = None
args.num_envs = cli_args.num_episodes
device = torch.device("cuda")

env = CatheterReachVectorEnv(num_envs=cli_args.num_episodes, max_episode_steps=cli_args.max_steps)
args.obs_dim = env.single_observation_space.shape[0]
args.action_dim = env.single_action_space.shape[0]

agent = TD3Agent(env, device, args)
agent.load_model(cli_args.checkpoint)
agent.eval_mode()

obs, _ = env.reset()
targets = env.targets.clone()

trajectory = []
with torch.no_grad():
    for step in range(cli_args.max_steps):
        trajectory.append(env.solver.positions.cpu().numpy().copy())
        action = agent.get_eval_action(obs)
        obs, reward, terminated, truncated, infos = env.step(action)
        if (terminated | truncated).all():
            break

trajectory = np.stack(trajectory)  # (T, num_episodes, num_points, 3)
success = infos["success"].cpu().numpy()
final_dist = torch.norm(env._get_tip_positions() - targets, dim=-1).cpu().numpy()

print("Success per episode:", success)
print("Final distance per episode:", final_dist)

np.savez(cli_args.out,
         trajectory=trajectory,
         targets=targets.cpu().numpy(),
         success=success,
         final_dist=final_dist)
print(f"Saved {cli_args.out}")