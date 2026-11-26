"""Palm position command for whole-body hand tracking (v0: position only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_from_matrix,
  quat_inv,
  quat_mul,
  yaw_quat,
)


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


# ---------------------------------------------------------------------------
# v1: position AND orientation
# ---------------------------------------------------------------------------


class HandPoseCommand(HandPositionCommand):
  """Palm position AND orientation targets for both hands (v1).

  Builds on HandPositionCommand: everything about positions works exactly as
  before. This class adds one orientation target per hand, which rotates toward
  its goal at no more than max_angular_speed (like positions glide at max_speed).

  What the policy observes (18 numbers):
    [0:6]   palm positions [left x, y, z, right x, y, z], heading frame
    [6:12]  left palm orientation, "6D" = first two columns of its rotation matrix
    [12:18] right palm orientation, "6D"
  Why 6D: it changes smoothly with the rotation. A quaternion has two values
  (q and -q) for the same rotation, and Euler angles jump at +-180 degrees.

  Internally, orientations are stored as quaternions (w, x, y, z), because
  rotating "a bit toward the goal" is easy with them.
  """

  cfg: HandPoseCommandCfg

  def __init__(self, cfg: HandPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)  # positions: library, goal_h, target_h (unchanged)

    library = torch.load(cfg.library_path)
    # The library stores 3x3 rotation matrices; convert them to quaternions once.
    self.library_quat = quat_from_matrix(library["targets_rotmat"]).to(self.device)  # (N, 2, 4)
    self.default_quat = quat_from_matrix(library["default_rotmat"]).to(self.device)  # (2, 4)

    # goal = where each palm should end up; target = what the policy sees now.
    self.goal_quat = self.default_quat.repeat(self.num_envs, 1, 1)  # (E, 2, 4)
    self.target_quat = self.goal_quat.clone()  # (E, 2, 4)
    # Feet, for the robot-style pelvis-height estimate (palm_pose_estimate_h).
    self.foot_site_ids, _ = self.robot.find_sites(cfg.foot_site_names, preserve_order=True)


    self.metrics["error_hand_rot"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """The 18 numbers the policy observes (see the class docstring)."""
    rotmat = matrix_from_quat(self.target_quat)  # (E, 2, 3, 3)
    # 6D = column 0 then column 1 of each rotation matrix: [c0x c0y c0z c1x c1y c1z].
    # (The third column is their cross product, so it carries no extra information.)
    six_d = torch.cat([rotmat[:, :, :, 0], rotmat[:, :, :, 1]], dim=-1)  # (E, 2, 6)
    return torch.cat([self.target_h, six_d.reshape(self.num_envs, 12)], dim=-1)  # (E, 18)

  def palm_quat_h(self) -> torch.Tensor:
    """Current palm orientations in the heading frame. Shape (E, 2, 4)."""
    _, quat_yaw = self._heading_frame()
    palm_quat_w = self.robot.data.site_quat_w[:, self.site_ids]  # world frame, (E, 2, 4)
    # The heading frame is the world turned by the pelvis yaw. To express a
    # world rotation in it, undo that yaw first: q_heading = q_yaw^-1 * q_world.
    quat_yaw_inv = quat_inv(quat_yaw).unsqueeze(1).expand(-1, 2, -1)  # same for both hands
    return quat_mul(quat_yaw_inv, palm_quat_w)

  def palm_rotation_error(self) -> torch.Tensor:
    """Angle [rad] between each palm's orientation and its target. Shape (E, 2).

    "How many radians would I have to turn the palm to match the target?"
    quat_error_magnitude handles q vs -q, so the answer is always in [0, pi].
    """
    return quat_error_magnitude(self.palm_quat_h(), self.target_quat)
  
  def palm_pose_estimate_h(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Palm poses in the heading frame, computed the way the REAL ROBOT can.

    The robot measures joint angles (-> forward kinematics) and the IMU (-> roll,
    pitch), so it knows where the palms and feet are RELATIVE TO THE PELVIS.
    It does not measure the pelvis height above the floor, so we estimate it:
    the lower foot stands on the floor. Training uses this same estimate, so
    the policy sees exactly what it will see on the robot.

    Returns (positions (E, 2, 3), orientations (E, 2, 4)).
    """
    data = self.robot.data
    _, quat_yaw = self._heading_frame()
    quat_yaw_both = quat_yaw.unsqueeze(1).expand(-1, 2, -1)  # same yaw for both hands

    # Palms relative to the pelvis, turned into the heading frame.
    pelvis_w = data.root_link_pos_w.unsqueeze(1)  # (E, 1, 3), broadcasts over 2 hands
    palm_rel_h = quat_apply_inverse(quat_yaw_both, data.site_pos_w[:, self.site_ids] - pelvis_w)

    # Feet height relative to the pelvis (a yaw turn doesn't change heights).
    feet_rel_z = data.site_pos_w[:, self.foot_site_ids, 2] - data.root_link_pos_w[:, 2:3]  # (E, 2)
    # The lower foot is on the floor, so the pelvis is this high above it:
    pelvis_height = self.cfg.foot_site_height - feet_rel_z.min(dim=1).values  # (E,)

    palm_pos = palm_rel_h.clone()  # clone: don't modify a tensor we might reuse
    palm_pos[..., 2] += pelvis_height.unsqueeze(1)  # heights measured from the floor

    # Orientation needs no estimate: forward kinematics + IMU give it fully.
    return palm_pos, self.palm_quat_h()


  def _update_metrics(self) -> None:
    super()._update_metrics()  # position error metric (unchanged)
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_hand_rot"] += self.palm_rotation_error().mean(dim=-1) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # One random library entry per env: its position AND its orientation together,
    # so the pair stays feasible (both came from the same joint angles).
    idx = torch.randint(0, self.library.shape[0], (len(env_ids),), device=self.device)
    self.goal_h[env_ids] = self.library[idx]
    self.goal_quat[env_ids] = self.library_quat[idx]
    # After a reset, restart the moving target from the default palm pose.
    just_reset = env_ids[self.command_counter[env_ids] == 0]
    self.target_h[just_reset] = self.default_target
    self.target_quat[just_reset] = self.default_quat

  def _update_command(self) -> None:
    super()._update_command()  # positions glide toward their goals (unchanged)

    # Orientations rotate toward their goals at no more than max_angular_speed.
    # 1. The rotation still to go, from target to goal: delta = target^-1 * goal.
    delta = quat_mul(quat_inv(self.target_quat), self.goal_quat)  # (E, 2, 4)
    # 2. Write it as "axis * angle": a 3-vector whose length is the angle [rad].
    axis_angle = axis_angle_from_quat(delta)  # (E, 2, 3)
    angle = torch.norm(axis_angle, dim=-1)  # (E, 2)
    # 3. This step, turn by at most max_angular_speed * dt around the same axis.
    max_step = self.cfg.max_angular_speed * self._env.step_dt
    step_angle = torch.clamp(angle, max=max_step)
    # Unit axis. The clamp avoids 0/0 when no rotation is left (then the axis is
    # 0, step_angle is 0, and the step below is "no rotation").
    axis = axis_angle / angle.clamp(min=1e-9).unsqueeze(-1)
    step = quat_from_angle_axis(step_angle, axis)  # (E, 2, 4)
    # 4. Apply the step. Re-normalise so tiny rounding errors don't build up.
    new_quat = quat_mul(self.target_quat, step)
    self.target_quat = new_quat / torch.norm(new_quat, dim=-1, keepdim=True)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Spheres (from the parent class) + one arrow per hand along the palm normal.

    Red arrow = target, green arrow = actual palm (same colors as the spheres).
    When a hand's orientation matches its target, the two arrows point the same way.
    """
    super()._debug_vis_impl(visualizer)  # the red/green position spheres
    origin_w, quat_yaw = self._heading_frame()
    quat_yaw_both = quat_yaw.unsqueeze(1).expand(-1, 2, -1)  # same yaw for both hands

    # Targets live in the heading frame; turn them into the world for drawing:
    # position: rotate by the pelvis yaw, then shift; orientation: q_world = q_yaw * q_heading.
    target_pos_w = quat_apply(quat_yaw_both, self.target_h.view(-1, 2, 3)) + origin_w.unsqueeze(1)
    target_quat_w = quat_mul(quat_yaw_both, self.target_quat)
    palm_pos_w = self.robot.data.site_pos_w[:, self.site_ids]
    palm_quat_w = self.robot.data.site_quat_w[:, self.site_ids]

    # Palm normal, written in the hand's own frame: its sideways (y) axis, signed so
    # both palms face the body at the default pose (left hand -y, right hand +y).
    # If the arrow comes out of the back of the hand, flip both signs.
    normal_local = torch.tensor(
      [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device
    ).expand(self.num_envs, 2, 3)
    target_normal_w = quat_apply(target_quat_w, normal_local)  # rotate into the world
    palm_normal_w = quat_apply(palm_quat_w, normal_local)

    # The viewer wants NumPy arrays on the CPU.
    tp, tn, pp, pn = (
      x.cpu().numpy() for x in (target_pos_w, target_normal_w, palm_pos_w, palm_normal_w)
    )
    for i in visualizer.get_env_indices(self.num_envs):
      for h in range(2):  # 0 = left, 1 = right
        visualizer.add_arrow(tp[i, h], tp[i, h] + 0.22 * tn[i, h], color=(1.0, 0.2, 0.2, 0.9), width=0.012)
        visualizer.add_arrow(pp[i, h], pp[i, h] + 0.27 * pn[i, h], color=(0.2, 0.8, 0.2, 0.9), width=0.008)

@dataclass(kw_only=True)
class HandPoseCommandCfg(HandPositionCommandCfg):
  max_angular_speed: float = 1.0
  """Max rotation speed [rad/s] of the observed orientation target."""
  foot_site_names: tuple[str, ...] = ("left_foot", "right_foot")
  foot_site_height: float = 0.0
  """Height [m] of the foot sites above the floor when a foot stands flat.
  Calibrated in sim: the sites sit at the sole, so 0.0."""

  def build(self, env: ManagerBasedRlEnv) -> HandPoseCommand:
    return HandPoseCommand(self, env)


def track_hand_orientation(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward in [0, 1] per env: mean over both hands of exp(-angle^2 / std^2).

  angle = how far [rad] each palm's orientation is from its target.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, HandPoseCommand)
  angle = term.palm_rotation_error()  # (E, 2)
  return torch.exp(-torch.square(angle) / std**2).mean(dim=-1)



def hand_pose_error(
  env: ManagerBasedRlEnv, command_name: str, pos_noise: float = 0.0, rot_noise: float = 0.0
) -> torch.Tensor:
  """How far each palm is from its target, as the policy sees it. Shape (E, 24).

  Per hand (12 numbers):
    [0:3]   position error = target - palm, heading frame [m]
    [3:6]   tanh(100 * position error): ~linear below 1 cm, saturates near
            +-1 beyond ~3 cm. Magnifies small errors (ResGAC's precision trick).
    [6:12]  rotation error as 6D: R_err = R_palm^T * R_target, the rotation still
            needed, seen from the palm. Aligned -> [1, 0, 0, 0, 1, 0].
  Uses the robot-style estimate, so the real robot can compute the same numbers.

  pos_noise [m] and rot_noise [rad]: random measurement error (training only).
  Without it the policy learned to react very strongly to errors of a millimetre,
  which is far below what real sensors can give, and shook on hardware. The noise
  enters BEFORE tanh, exactly where real measurement noise would.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, HandPoseCommand)
  palm_pos, palm_quat = term.palm_pose_estimate_h()  # (E, 2, 3), (E, 2, 4)

  pos_err = term.target_h.view(-1, 2, 3) - palm_pos  # (E, 2, 3)
  rot_err_quat = quat_mul(quat_inv(palm_quat), term.target_quat)  # (E, 2, 4)
  if pos_noise > 0.0:
    pos_err = pos_err + (2 * torch.rand_like(pos_err) - 1) * pos_noise  # +- pos_noise per axis
  if rot_noise > 0.0:  # a small random turn about a random axis
    axis = torch.randn_like(pos_err)
    angle = (2 * torch.rand(*pos_err.shape[:2], device=pos_err.device) - 1) * rot_noise
    rot_err_quat = quat_mul(rot_err_quat, quat_from_angle_axis(angle, axis))
  rot_err = matrix_from_quat(rot_err_quat)  # (E, 2, 3, 3)
  rot_err_6d = torch.cat([rot_err[:, :, :, 0], rot_err[:, :, :, 1]], dim=-1)  # (E, 2, 6)

  per_hand = torch.cat([pos_err, torch.tanh(100.0 * pos_err), rot_err_6d], dim=-1)  # (E, 2, 12)
  return per_hand.reshape(term.num_envs, 24)


def track_hand_position_exp(
  env: ManagerBasedRlEnv, command_name: str, var: float, l1_scale: float = 0.0
) -> torch.Tensor:
  """ResGAC-style position reward in [0, 1], mean over both hands:

      exp(-e^2 / var - e / l1_scale)     (the L1 part only if l1_scale > 0)

  e = true distance [m] from each palm to its target. `var` is a VARIANCE:
  0.02 -> ~14 cm wide (early curriculum: a signal even far away),
  0.001 -> ~3 cm (final). The L1 part (l1_scale 0.01) keeps a strong pull
  even inside the last centimetre, where e^2 alone becomes flat.
  The curriculum changes var and l1_scale during training.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, HandPositionCommand)
  e = term.palm_error()  # (E, 2)
  exponent = torch.square(e) / var
  if l1_scale > 0.0:
    exponent = exponent + e / l1_scale
  return torch.exp(-exponent).mean(dim=-1)


def track_hand_orientation_trace(
  env: ManagerBasedRlEnv, command_name: str, var: float
) -> torch.Tensor:
  """ResGAC-style orientation reward in [0, 1], mean over both hands:

      exp(-trace_error / var),  trace_error = 3 - trace(R_target^T R_palm)
                                            = 2 (1 - cos(angle))

  For small angles trace_error ~ angle^2, so var also acts like a variance:
  0.2 -> ~26 deg wide (early), 0.02 -> ~8 deg (final).
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, HandPoseCommand)
  angle = term.palm_rotation_error()  # (E, 2) radians, ground truth
  trace_error = 2.0 * (1.0 - torch.cos(angle))
  return torch.exp(-trace_error / var).mean(dim=-1)
