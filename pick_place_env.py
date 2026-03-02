"""SO-ARM101 Pick and Place Task — reach, grasp, lift, and place a cube."""

import math
import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

import sys
sys.path.append("C:/dev/isaac/so101_project")
from so101_robot_cfg import SO101_CFG


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@configclass
class SO101PickPlaceEnvCfg(DirectRLEnvCfg):
    """Config for pick-and-place task."""

    # Sim
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    decimation = 2

    # Task dimensions
    num_observations = 30
    num_actions = 6
    episode_length_s = 8.0  # longer than reaching — arm needs time to grab + move

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=1.5,
        replicate_physics=False,
    )

    # Robot
    robot_cfg: ArticulationCfg = SO101_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Cube — 2cm blue cube, light (10 grams)
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.0, 0.01)),
    )

    # Where the cube spawns (local coords — on the table surface)
    cube_min = [0.05, -0.10, 0.01]
    cube_max = [0.15, 0.10, 0.01]  # z fixed at table height

    # Where to place the cube (local coords — elevated to force lifting)
    goal_min = [-0.05, -0.10, 0.08]
    goal_max = [0.10, 0.10, 0.15]

    # ── Reward weights ──
    # These are graduated: placing is worth 16x more than reaching
    reward_reach = 1       # (1x)  get ee close to cube
    reward_grasp = 2       # (5x)  close gripper when near cube
    reward_lift = 1.5        # (15x) lift cube above table
    reward_goal = 1.6        # (16x) move cube to goal position
    reward_action_penalty = -0.0001

    # Thresholds
    grasp_dist = 0.04       # ee must be within 2.5cm of cube for grasp reward
    lift_height = 0.04       # cube must be 4cm above start to count as "lifted"

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.num_observations,)
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class SO101PickPlaceEnv(DirectRLEnv):
    """Pick and place — reach the cube, grab it, lift it, put it at the goal."""

    cfg: SO101PickPlaceEnvCfg

    def __init__(self, cfg: SO101PickPlaceEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices (all 6 — 5 arm + 1 gripper)
        self.arm_joint_ids, _ = self.robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        )

        # End effector body
        self.ee_body_id, _ = self.robot.find_bodies(["gripper_frame_link"])

        # Gripper joint index (last one) and its limits
        gripper_limits = self.robot.root_physx_view.get_dof_limits()[0, 5]
        self.gripper_low = gripper_limits[0].item()   # -0.17 rad (closed)
        self.gripper_high = gripper_limits[1].item()   # 1.75 rad (open)

        # Goal positions buffer
        self.goals = torch.zeros(self.num_envs, 3, device=self.device)

        # Track cube start height for lift reward
        self.cube_start_z = torch.zeros(self.num_envs, device=self.device)

        # Randomize on start
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_cube(all_ids)
        self._randomize_goals(all_ids)

    # ── Scene setup ──

    def _setup_scene(self):
        """Add robot + cube to the scene."""
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Cube
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/light", light_cfg)

        # Goal marker (green sphere)
        goal_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goals",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        self.goal_markers = VisualizationMarkers(goal_marker_cfg)

        # Cube marker (red sphere on cube for visibility)
        cube_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/cube_markers",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.cube_markers = VisualizationMarkers(cube_marker_cfg)

    # ── Actions ──

    def _pre_physics_step(self, actions: torch.Tensor):
        """Map [-1, 1] actions to joint position targets."""
        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[..., 0][:, :6]
        upper = joint_limits[..., 1][:, :6]
        self._targets = 0.5 * (actions + 1.0) * (upper - lower) + lower

    def _apply_action(self):
        """Send targets to robot."""
        self.robot.set_joint_position_target(self._targets, joint_ids=self.arm_joint_ids)

    # ── Observations (30 dim) ──

    def _get_observations(self) -> dict:
        # Joint state (includes gripper as joint 6)
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[..., 0][:, :6]
        upper = joint_limits[..., 1][:, :6]
        joint_pos_norm = 2.0 * (joint_pos - lower) / (upper - lower) - 1.0
        joint_vel_scaled = joint_vel * 0.1

        # Positions in local coordinates
        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id[0], :]
        ee_pos_local = ee_pos - self.scene.env_origins

        cube_pos = self.cube.data.root_pos_w
        cube_pos_local = cube_pos - self.scene.env_origins

        goal_pos_local = self.goals - self.scene.env_origins

        # Relative vectors
        ee_to_cube = cube_pos_local - ee_pos_local
        cube_to_goal = goal_pos_local - cube_pos_local

        # Distances
        dist_ee_cube = torch.norm(ee_to_cube, dim=-1, keepdim=True)
        dist_cube_goal = torch.norm(cube_to_goal, dim=-1, keepdim=True)

        # Cube height above its starting position (useful for lift stage)
        cube_height = (cube_pos_local[:, 2:3] - self.cube_start_z.unsqueeze(-1))

        obs = torch.cat([
            joint_pos_norm,      # 6  — where are the joints
            joint_vel_scaled,    # 6  — how fast are they moving
            ee_pos_local,        # 3  — where is the gripper
            cube_pos_local,      # 3  — where is the cube
            goal_pos_local,      # 3  — where should the cube go
            ee_to_cube,          # 3  — vector from gripper to cube
            cube_to_goal,        # 3  — vector from cube to goal
            dist_ee_cube,        # 1  — how far is gripper from cube
            dist_cube_goal,      # 1  — how far is cube from goal
            cube_height,         # 1  — how high has the cube been lifted
        ], dim=-1)               # total: 30

        # Update markers
        if hasattr(self, "goal_markers"):
            self.goal_markers.visualize(self.goals)
        if hasattr(self, "cube_markers"):
            self.cube_markers.visualize(self.cube.data.root_pos_w)

        return {"policy": obs}

    # ── Rewards (graduated) ──

    def _get_rewards(self) -> torch.Tensor:
        # Positions
        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id[0], :]
        cube_pos = self.cube.data.root_pos_w

        # Distances
        dist_ee_cube = torch.norm(ee_pos - cube_pos, dim=-1)
        dist_cube_goal = torch.norm(cube_pos - self.goals, dim=-1)

        # Gripper openness: 0 = closed, 1 = fully open
        gripper_pos = self.robot.data.joint_pos[:, 5]
        gripper_open = (gripper_pos - self.gripper_low) / (self.gripper_high - self.gripper_low)
        gripper_open = gripper_open.clamp(0, 1)

        # Cube height above start
        cube_height = (cube_pos[:, 2] - self.scene.env_origins[:, 2]) - self.cube_start_z

        # ── Stage 1: Reach the cube ──
        # Always active — get the end effector close to the cube
        reach_reward = self.cfg.reward_reach * (1.0 - torch.tanh(5.0 * dist_ee_cube))

        # ── Stage 2: Grasp ──
        # When ee is near the cube, reward closing the gripper
        near_cube = (dist_ee_cube < self.cfg.grasp_dist).float()
        grasp_reward = self.cfg.reward_grasp * near_cube * (1.0 - gripper_open)

        # ── Stage 3: Lift ──
        # Reward the cube being above the table (only when gripper is mostly closed)
        gripper_closed = (gripper_open < 0.3).float()
        lift_reward = self.cfg.reward_lift * gripper_closed * torch.tanh(
            5.0 * torch.clamp(cube_height, min=0.0)
        )

        # ── Stage 4: Goal placement ──
        # When cube is lifted, reward moving it toward the goal
        cube_lifted = (cube_height > self.cfg.lift_height).float()
        goal_reward = self.cfg.reward_goal * cube_lifted * (1.0 - torch.tanh(5.0 * dist_cube_goal))

        # Action smoothness penalty
        action_penalty = self.cfg.reward_action_penalty * torch.sum(
            self.robot.data.joint_vel[:, :6] ** 2, dim=-1
        )

        total = reach_reward + grasp_reward + lift_reward + goal_reward + action_penalty
        return total

    # ── Termination ──

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_limit = self.episode_length_buf >= self.max_episode_length

        # Terminate if cube falls off the table (below ground)
        cube_pos_local = self.cube.data.root_pos_w - self.scene.env_origins
        cube_fell = cube_pos_local[:, 2] < -0.05

        return cube_fell, time_limit

    # ── Resets ──

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)
        num = len(env_ids)

        # Reset robot joints (small random offset from home)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros(num, self.robot.num_joints, device=self.device)

        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        lower = joint_limits[env_ids, :, 0]
        upper = joint_limits[env_ids, :, 1]
        joint_pos += torch.randn(num, self.robot.num_joints, device=self.device) * 0.1
        joint_pos = torch.clamp(joint_pos, lower, upper)

        # Start gripper open
        joint_pos[:, 5] = self.gripper_high * 0.8  # 80% open

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset cube to random position on table
        self._randomize_cube(env_ids)

        # New goal
        self._randomize_goals(env_ids)

    def _randomize_cube(self, env_ids: torch.Tensor):
        """Put the cube at a random spot on the table."""
        num = len(env_ids)
        low = torch.tensor(self.cfg.cube_min, device=self.device)
        high = torch.tensor(self.cfg.cube_max, device=self.device)

        local_pos = low + (high - low) * torch.rand(num, 3, device=self.device)
        world_pos = local_pos + self.scene.env_origins[env_ids]

        # Save start height for lift reward
        self.cube_start_z[env_ids] = local_pos[:, 2]

        # Write cube state — position + zero velocity
        zeros = torch.zeros(num, 3, device=self.device)
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num, -1)

        self.cube.write_root_pose_to_sim(
            torch.cat([world_pos, default_quat], dim=-1), env_ids=env_ids
        )
        self.cube.write_root_velocity_to_sim(
            torch.cat([zeros, zeros], dim=-1), env_ids=env_ids
        )

    def _randomize_goals(self, env_ids: torch.Tensor):
        """Random goal position — elevated to require lifting."""
        num = len(env_ids)
        low = torch.tensor(self.cfg.goal_min, device=self.device)
        high = torch.tensor(self.cfg.goal_max, device=self.device)

        local_goals = low + (high - low) * torch.rand(num, 3, device=self.device)
        self.goals[env_ids] = local_goals + self.scene.env_origins[env_ids]
