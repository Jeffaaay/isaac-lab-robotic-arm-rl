"""Train SO101 pick-and-place task with PPO."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
import sys
import os
from datetime import datetime

sys.path.append("C:/dev/isaac/so101_project")
from pick_place_env import SO101PickPlaceEnv, SO101PickPlaceEnvCfg

from rsl_rl.runners import OnPolicyRunner


class ObsDict(dict):
    def to(self, device):
        return ObsDict({k: v.to(device) for k, v in self.items()})


class RslRlWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.cfg.num_observations
        self.num_privileged_obs = None
        self.num_actions = env.cfg.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return ObsDict({"policy": self.obs})

    def reset(self):
        obs_dict, info = self.env.reset()
        self.obs = obs_dict["policy"]
        return ObsDict({"policy": self.obs}), {"observations": {}}

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        self.obs = obs_dict["policy"]
        dones = terminated | truncated
        return ObsDict({"policy": self.obs}), rewards, dones, info


# ── Environment ──
cfg = SO101PickPlaceEnvCfg()
cfg.scene.num_envs = 512
env = RslRlWrapper(SO101PickPlaceEnv(cfg))
print(f"Environment created: {env.num_envs} envs, {env.num_obs} obs, {env.num_actions} actions")
env.reset()

# ── PPO Config ──
# Same proven settings from reaching, but bigger network for harder task
train_cfg = {
    "algorithm_class_name": "PPO",
    "policy_class_name": "ActorCritic",
    "num_steps_per_env": 24,
    "max_iterations": 10000,      # longer — pick-and-place is harder
    "save_interval": 100,
    "log_interval": 1,
    "experiment_name": "so101_pick_place",
    "run_name": "",
    "resume": True,
    "load_run": "C:/dev/isaac/so101_project/logs_pick_place/2026-03-01_16-00-57",
    "checkpoint": 3999,
    "record_interval": -1,
    "obs_groups": {},
    "policy": {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 1.5,
    },
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "init_member_classes": {},
    "seed": 42,
}

log_dir = "C:/dev/isaac/so101_project/logs_pick_place/2026-03-01_16-00-57"

os.makedirs(log_dir, exist_ok=True)

runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device="cuda:0")

# ── Resume from checkpoint ──
if train_cfg["resume"]:
    checkpoint_path = os.path.join(log_dir, "model_3999.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)

print(f"Training for {train_cfg['max_iterations']} iterations...")
print(f"Logs: {log_dir}")

runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)

env.env.close()
app.close()
