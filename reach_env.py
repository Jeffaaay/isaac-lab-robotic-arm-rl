"""SO-ARM101 Reaching Task — move end effector to random XYZ targets."""

import math
import torch
from dataclasses import MISSING
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Import our robot config
import sys
sys.path.append("C:/dev/isaac/so101_project")
from so101_robot_cfg import SO101_CFG


@configclass
class SO101ReachEnvCfg(DirectRLEnvCfg):
    """Configuration for SO101 reaching task."""

    # Sim settings
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    decimation = 2

    # Task dimensions
    num_observations = 16
    num_actions = 6
    episode_length_s = 5.0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=1.5,
        replicate_physics=False,
    )
    

    # Robot
    robot_cfg: ArticulationCfg = SO101_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Target randomization bounds (meters)
    target_min = [-0.15, -0.15, 0.02]
    target_max = [0.15, 0.15, 0.25]

    # Reward weights
    reward_reaching = 0.1
    reward_close_bonus = 0.05
    reward_action_penalty = -0.0001

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.num_observations,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))


class SO101ReachEnv(DirectRLEnv):
    """Reaching environment — arm must touch random XYZ targets."""

    cfg: SO101ReachEnvCfg

    def __init__(self, cfg: SO101ReachEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices for the 6 joints
        self.arm_joint_ids, _ = self.robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        )

        # Find the end effector body
        self.ee_body_id, _ = self.robot.find_bodies(["gripper_frame_link"])

        # Create target positions buffer
        self.targets = torch.zeros(self.num_envs, 3, device=self.device)

        # Randomize targets on start
        self._randomize_targets(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):

        """Add robot to the scene."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/light", light_cfg)

        


    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions from policy."""
        # Scale actions from [-1, 1] to joint position targets
        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[..., 0]
        upper = joint_limits[..., 1]

        # Map [-1, 1] to [lower, upper] for each joint
        self._targets = 0.5 * (actions + 1.0) * (upper[:, :6] - lower[:, :6]) + lower[:, :6]

    def _apply_action(self):
        """Send joint targets to robot — called every physics substep."""
        self.robot.set_joint_position_target(self._targets, joint_ids=self.arm_joint_ids)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[..., 0][:, :6]
        upper = joint_limits[..., 1][:, :6]
        joint_pos_norm = 2.0 * (joint_pos - lower) / (upper - lower) - 1.0

        joint_vel_scaled = joint_vel * 0.1

        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id[0], :]

        # Local coordinates — every env sees the same scale
        ee_to_target = self.targets - ee_pos
        distance = torch.norm(ee_to_target, dim=-1, keepdim=True)

        obs = torch.cat([
        joint_pos_norm,      # 6 - where are my joints
        joint_vel_scaled,    # 6 - how fast are they moving
        ee_to_target,        # 3 - direction to target
        distance,            # 1 - how far away
        ], dim=-1)              # total: 16
 

        return {"policy": obs}
        
 

    def _get_rewards(self) -> torch.Tensor:
        """Reward for reaching the target."""
        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id[0], :]
        distance = torch.norm(self.targets - ee_pos, dim=-1)

        # Negative distance reward — closer = higher reward
        reaching_reward = self.cfg.reward_reaching * (1.0 - torch.tanh(5.0 * distance))

        # Bonus for being very close (< 1cm)
        close_bonus = self.cfg.reward_close_bonus * (distance < 0.01).float()

        # Small penalty for large actions to encourage smooth motion
        action_penalty = self.cfg.reward_action_penalty * torch.sum(
            self.robot.data.joint_vel[:, :6] ** 2, dim=-1
        )

        return reaching_reward + close_bonus + action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_limit = self.episode_length_buf >= self.max_episode_length

        # No early termination for reaching — let the arm learn
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return died, time_limit

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments that are done."""
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset robot to home position with small random offset
        num_envs = len(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros(num_envs, self.robot.num_joints, device=self.device)

        # Clamp randomization to joint limits
        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[env_ids, :, 0]
        upper = joint_limits[env_ids, :, 1]
        joint_pos += torch.randn(num_envs, self.robot.num_joints, device=self.device) * 0.1
        joint_pos = torch.clamp(joint_pos, lower, upper)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # New random targets
        self._randomize_targets(env_ids)
        
    def _randomize_targets(self, env_ids: torch.Tensor):
        num = len(env_ids)
        low = torch.tensor(self.cfg.target_min, device=self.device)
        high = torch.tensor(self.cfg.target_max, device=self.device)

        local_targets = low + (high - low) * torch.rand(num, 3, device=self.device)
        self.targets[env_ids] = local_targets + self.scene.env_origins[env_ids]

 