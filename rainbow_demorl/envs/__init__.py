from .maniskill import find_max_episode_steps_value, make_envs as make_maniskill_envs
from .catheter import make_envs as make_catheter_envs


def make_envs(args, run_name: str):
    if args.env_id.startswith("Catheter"):
        return make_catheter_envs(args, run_name)
    return make_maniskill_envs(args, run_name)