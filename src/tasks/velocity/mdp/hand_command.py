"""Palm position command for whole-body hand tracking (v0: position only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class HandPositionCommand(CommandTerm):
  """Palm targets for both hands in the ground-attached heading frame.

  Heading frame: origin = pelvis xy on the ground (z = 0), rotation = pelvis
  yaw only, so pelvis roll, pitch and height do not move the targets. Goals are
  drawn from a library of reachable palm positions. The observed target moves
  toward the goal at max_speed, so it never jumps.
  """

  cfg: HandPositionCommandCfg

  def __init__(self, cfg: HandPositionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.site_ids, _ = self.robot.find_sites(cfg.site_names, preserve_order=True)

    library = torch.load(cfg.library_path)
    self.library = library["targets"].to(self.device)  # (N, 6): [left xyz, right xyz]
    self.default_target = library["default"].to(self.device)  # (6,)

    self.goal_h = self.default_target.repeat(self.num_envs, 1)  # (E, 6)
    self.target_h = self.goal_h.clone()  # (E, 6), what the policy sees.

    self.metrics["error_hand_pos"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_h

  def _heading_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
    origin_w = self.robot.data.root_link_pos_w.clone()
    origin_w[:, 2] = 0.0
    return origin_w, yaw_quat(self.robot.data.root_link_quat_w)

  def palm_pos_h(self) -> torch.Tensor:
    """Current palm positions in the heading frame. Shape (E, 2, 3)."""
    origin_w, quat_yaw = self._heading_frame()
    rel_w = self.robot.data.site_pos_w[:, self.site_ids] - origin_w.unsqueeze(1)
    return quat_apply_inverse(quat_yaw.unsqueeze(1).expand(-1, 2, -1), rel_w)

  def palm_error(self) -> torch.Tensor:
    """Distance from each palm to its target [m]. Shape (E, 2)."""
    return torch.norm(self.palm_pos_h() - self.target_h.view(-1, 2, 3), dim=-1)

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_hand_pos"] += self.palm_error().mean(dim=-1) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    idx = torch.randint(0, self.library.shape[0], (len(env_ids),), device=self.device)
    self.goal_h[env_ids] = self.library[idx]
    # After a reset, restart the moving target from the default palm pose.
    just_reset = env_ids[self.command_counter[env_ids] == 0]
    self.target_h[just_reset] = self.default_target

  def _update_command(self) -> None:
    # Move each hand's target toward its goal at no more than max_speed.
    delta = (self.goal_h - self.target_h).view(-1, 2, 3)
    dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
    frac = torch.clamp(self.cfg.max_speed * self._env.step_dt / dist, max=1.0)
    self.target_h += (delta * frac).view(-1, 6)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Red spheres = targets, green spheres = palms."""
    origin_w, quat_yaw = self._heading_frame()
    target_w = quat_apply(
      quat_yaw.unsqueeze(1).expand(-1, 2, -1), self.target_h.view(-1, 2, 3)
    ) + origin_w.unsqueeze(1)
    target_w = target_w.cpu().numpy()
    palm_w = self.robot.data.site_pos_w[:, self.site_ids].cpu().numpy()
    for i in visualizer.get_env_indices(self.num_envs):
      for h in range(2):
        visualizer.add_sphere(target_w[i, h], 0.03, (1.0, 0.2, 0.2, 0.8))
        visualizer.add_sphere(palm_w[i, h], 0.02, (0.2, 0.8, 0.2, 0.8))


@dataclass(kw_only=True)
class HandPositionCommandCfg(CommandTermCfg):
  entity_name: str
  library_path: str
  site_names: tuple[str, ...] = ("left_palm", "right_palm")
  max_speed: float = 0.5
  """Max speed [m/s] of the observed target. Also a hardware safety knob."""

  def build(self, env: ManagerBasedRlEnv) -> HandPositionCommand:
    return HandPositionCommand(self, env)

def track_hand_position(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward in [0, 1] per env: mean over both hands of exp(-error² / std²)."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, HandPositionCommand)
  error = term.palm_error()  # (E, 2): distance to target per hand [m]
  return torch.exp(-torch.square(error) / std**2).mean(dim=-1)

