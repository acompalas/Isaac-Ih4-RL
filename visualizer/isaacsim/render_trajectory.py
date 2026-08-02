"""
Loads eval_trajectories.npz (rod + target positions over time, multiple episodes)
and renders them as a video using Isaac Sim, spawning small spheres along each
rod's segments and a larger sphere at each episode's target.

UNTESTED: first use of Isaac Sim's scripting API in this project. Expect to
debug this against the actual pod environment.

Run with Isaac Sim's own Python (not the rl_env venv), since this needs the
omni/isaacsim modules that only exist in Isaac Sim's bundled environment:
    /workspace/isaaclab/_isaac_sim/python.sh visualizer/isaacsim/render_trajectory.py
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="visualizer/eval_trajectories.npz")
parser.add_argument("--episodes", type=int, nargs="+", default=None,
                     help="Which episode indices to render; default = all successes, else all")
parser.add_argument("--out", type=str, default="visualizer/catheter_eval.mp4")
parser.add_argument("--headless", action="store_true", default=True)
cli_args = parser.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": cli_args.headless})

import omni.usd
from pxr import UsdGeom, Gf, Sdf
from omni.isaac.core import World
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.viewports import set_camera_view
import omni.kit.commands

data = np.load(cli_args.data)
trajectory = data["trajectory"]   # (T, num_episodes, num_points, 3)
targets = data["targets"]         # (num_episodes, 3)
success = data["success"]

if cli_args.episodes is not None:
    episode_ids = cli_args.episodes
elif success.any():
    episode_ids = np.where(success)[0].tolist()
else:
    episode_ids = list(range(trajectory.shape[1]))

print(f"Rendering episodes: {episode_ids}")

world = World()
world.scene.add_default_ground_plane()

# spawn one small sphere per rod segment, per selected episode, plus one target sphere
num_points = trajectory.shape[2]
rod_spheres = {}   # (episode_id, point_idx) -> VisualSphere
target_spheres = {}

colors = [
    np.array([0.9, 0.1, 0.1]), np.array([0.1, 0.9, 0.1]), np.array([0.1, 0.1, 0.9]),
    np.array([0.9, 0.9, 0.1]), np.array([0.9, 0.1, 0.9]), np.array([0.1, 0.9, 0.9]),
]

for i, ep in enumerate(episode_ids):
    color = colors[i % len(colors)]
    for p in range(num_points):
        prim_path = f"/World/rod_ep{ep}_pt{p}"
        sphere = VisualSphere(prim_path=prim_path, radius=0.008, color=color)
        world.scene.add(sphere)
        rod_spheres[(ep, p)] = sphere

    target_sphere = VisualSphere(
        prim_path=f"/World/target_ep{ep}",
        radius=0.05,
        color=np.array([1.0, 0.0, 0.0]),
    )
    world.scene.add(target_sphere)
    target_spheres[ep] = target_sphere

world.reset()

set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.5, 0.0, 0.5])

for i, ep in enumerate(episode_ids):
    target_spheres[ep].set_world_pose(position=targets[ep])

# step through recorded trajectory, updating primitive positions each frame
num_frames = trajectory.shape[0]
for t in range(num_frames):
    for ep in episode_ids:
        for p in range(num_points):
            pos = trajectory[t, ep, p]
            rod_spheres[(ep, p)].set_world_pose(position=pos)
    world.step(render=True)

# NOTE: actual video capture (frame -> mp4) still needs to be wired in here,
# e.g. via omni.kit.capture or repeated viewport screenshots + ffmpeg.
# Leaving as a documented next step since it's the least-verified part of this script.

print("Playback complete. Video capture step still needs implementation/testing.")
sim_app.close()