"""Play trained SO101 reaching policy visually."""
from isaacsim import SimulationApp
app = SimulationApp({"headless":False})

import torch
import sys
import os

sys.path.append("C:/dev/isaac/so101_project")
from reach_env import SO101ReachEnv, SO101ReachEnvCfg

from rsl_rl.runners import OnPolicyRunner


class ObsDict(dict):
    def to(self, device):
        return ObsDict({k: v.to(device) for k, v in self.items()})


class RslRlWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = 22
        self.num_privileged_obs = None
        self.num_actions = 6
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


CHECKPOINT = "C:/dev/isaac/so101_project/logs/2026-02-28_18-42-35/model_1999.pt"
NUM_ENVS = 4

cfg = SO101ReachEnvCfg()
cfg.scene.num_envs = NUM_ENVS
env = RslRlWrapper(SO101ReachEnv(cfg))
env.reset()

train_cfg = {
    "algorithm_class_name": "PPO",
    "policy_class_name": "ActorCritic",
    "num_steps_per_env": 24,
    "max_iterations": 2000,
    "save_interval": 50,
    "log_interval": 1,
    "experiment_name": "so101_reach",
    "run_name": "",
    "resume": False,
    "load_run": -1,
    "checkpoint": -1,
    "record_interval": -1,
    "obs_groups": {},
    "policy": {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 1.0,
    },
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.005,
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

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/play", device="cuda:0")

print(f"Loading checkpoint: {CHECKPOINT}")
runner.load(CHECKPOINT)
print(dir(runner.alg))
# Set noise to 0 for deterministic evaluation
with torch.no_grad():
    runner.alg.policy.std.fill_(0.0)
print("Noise std set to 0.0 (deterministic mode)")

obs, _ = env.reset()
step_count = 0
episode_count = 0

print("\nWatching arms reach for red spheres...\n")

while episode_count < 20:
    with torch.no_grad():
        actions = runner.alg.policy.act_inference(obs)
    obs_dict, rewards, dones, info = env.step(actions)
    obs = obs_dict
    step_count += 1

    if dones.any():
        episode_count += dones.sum().item()
        print(f"Episodes completed: {episode_count}/20")

print("\nDone!")
env.env.close()
app.close()
