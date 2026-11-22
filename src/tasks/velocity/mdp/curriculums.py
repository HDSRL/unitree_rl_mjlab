from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])

def hand_training_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[dict],
) -> torch.Tensor:
  """Move through ResGAC-style training stages as training progresses.

  Each stage (a dict) starts at a training `step` and sets:
    standing:               fraction of robots told to stand still
    lin_vel_x/y, ang_vel_z: walking command ranges (optional)
    pos_var, pos_l1:        hand position reward width (track_hand_position_exp)
    rot_var:                hand orientation reward width
  The last stage whose `step` has passed is active. Called whenever episodes
  reset, so a new stage takes effect at the next resets.
  Returns the active stage index (logged during training).
  """
  del env_ids  # Unused: stages depend on training progress, not on which robots reset.
  active = 0
  for index, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      active = index
  stage = stages[active]

  # Reward widths: the reward manager reads `params` on every call, so changing
  # them here changes the reward from the next step on.
  rewards = env.reward_manager
  rewards.get_term_cfg("hand_position").params["var"] = stage["pos_var"]
  rewards.get_term_cfg("hand_position").params["l1_scale"] = stage["pos_l1"]
  rewards.get_term_cfg("hand_orientation").params["var"] = stage["rot_var"]

  # Walking: standing fraction and command ranges (used at the next resample).
  twist = env.command_manager.get_term("twist").cfg
  twist.rel_standing_envs = stage["standing"]
  if "lin_vel_x" in stage:
    twist.ranges.lin_vel_x = stage["lin_vel_x"]
    twist.ranges.lin_vel_y = stage["lin_vel_y"]
    twist.ranges.ang_vel_z = stage["ang_vel_z"]
  return torch.tensor(float(active))


class penalty_curriculum:
  """Start the style penalties weak, ramp them up as the robot gets good.

  Holosoma's trick: strong penalties early punish every clumsy move, and the
  policy may never learn the task (what happened to v1). So the listed terms
  start at `initial_scale` (10%) of their weight. Each time episodes end, we
  look at how long robots lasted (moving average, in policy steps):
    longer than level_up_steps    -> scale *= 1 + rate  (grow toward 1.0)
    shorter than level_down_steps -> scale *= 1 - rate  (back off)
  The scale stays in [initial_scale, 1]. Keep safety terms (joint limits,
  self-collision, falling) OUT of penalty_names: they stay at full strength.

  A class, not a function, because it must remember the original weights and
  the current scale between calls. mjlab creates one instance at start-up.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    names = cfg.params["penalty_names"]
    self.original_weights = {name: env.reward_manager.get_term_cfg(name).weight for name in names}
    self.scale = cfg.params["initial_scale"]
    self.average_length = 0.0  # moving average of episode length [policy steps]
    self._apply(env)

  def _apply(self, env: ManagerBasedRlEnv) -> None:
    for name, weight in self.original_weights.items():
      env.reward_manager.get_term_cfg(name).weight = weight * self.scale

  def __call__(
    self, env, env_ids, penalty_names, initial_scale, level_up_steps, level_down_steps, rate
  ) -> torch.Tensor:
    del penalty_names  # Already used in __init__.
    # Called just BEFORE these robots reset, so their episode lengths are still there.
    lengths = env.episode_length_buf[env_ids]
    if lengths.numel() > 0:
      self.average_length = 0.99 * self.average_length + 0.01 * lengths.float().mean().item()
      if self.average_length > level_up_steps:
        self.scale *= 1.0 + rate
      elif self.average_length < level_down_steps:
        self.scale *= 1.0 - rate
      self.scale = min(max(self.scale, initial_scale), 1.0)
      self._apply(env)
    return torch.tensor(self.scale)

