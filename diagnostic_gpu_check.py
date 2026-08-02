import time
import sys
import torch
from catheter_vasculature_solver import RodConfig, XPBDRodSolver

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
sys.stdout.flush()

config = RodConfig()

for n in [1, 32, 128, 512]:
    print(f"\n=== num_envs={n} ===")
    print("  building solver...", flush=True)
    solver = XPBDRodSolver(config, num_envs=n, floor_z=None)
    print("  positions device:", solver.positions.device, flush=True)
    print("  positions shape:", solver.positions.shape, flush=True)

    print("  running first step (includes JIT compile + CUDA graph capture, expect this to be slow)...", flush=True)
    t0 = time.perf_counter()
    solver.step(config.solver.dt)
    torch.cuda.synchronize()
    print(f"  first step took {time.perf_counter() - t0:.3f}s", flush=True)

    print("  running 99 more steps (should be fast, replaying CUDA graph)...", flush=True)
    start = time.perf_counter()
    for i in range(99):
        solver.step(config.solver.dt)
        if (i + 1) % 20 == 0:
            print(f"    step {i+1}/99", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  99 steps took {elapsed:.3f}s ({elapsed/99*1000:.2f} ms/step)", flush=True)