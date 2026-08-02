# /workspace/isaaclab/Isaac-Ih4-RL/diagnostic_gpu_check.py
import time
import torch
from catheter_vasculature_solver import RodConfig, XPBDRodSolver

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

config = RodConfig()

for n in [1, 32, 128, 512]:
    solver = XPBDRodSolver(config, num_envs=n, floor_z=None)
    print(f"\nnum_envs={n}")
    print("  positions device:", solver.positions.device)
    print("  positions dtype:", solver.positions.dtype)
    print("  positions shape:", solver.positions.shape)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        solver.step(config.solver.dt)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  100 steps took {elapsed:.3f}s ({elapsed/100*1000:.2f} ms/step)")