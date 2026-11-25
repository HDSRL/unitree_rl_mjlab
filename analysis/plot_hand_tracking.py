"""Plot how well the hands track their targets, from a g1_ctrl deployment log (.csv).

What this script does, in plain words:
  1. Read the log: joint angles, IMU, hand goals and targets (50 rows per second).
  2. Work out where the palms REALLY were, with the same math the robot code uses
     (hand_pose_error in hand_observations.hpp):
       joint angles -> forward kinematics (MuJoCo, the training XML) -> pelvis frame
       -> turn by the IMU tilt (yaw removed), add the pelvis height -> heading frame
  3. Compare with the target the policy was given. Print a table, save two images.

Run from the repo root (mjlab environment):
  python analysis/plot_hand_tracking.py                 # newest log
  python analysis/plot_hand_tracking.py <path to .csv>  # a specific log

Output: analysis/plots/<log name>_tracking.png and ..._errors.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths and colours
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file is in analysis/)
LOGS = ROOT / "deploy/robots/g1/log/deployment"
XML = ROOT / "src/assets/robots/unitree_g1/xmls/g1_27dof.xml"
DEPLOY_YAML = ROOT / "deploy/robots/g1/config/policy/hands/v21_27dof/params/deploy.yaml"
OUT = ROOT / "analysis/plots"

LEFT, RIGHT = "#2a78d6", "#eb6834"  # one colour per hand, same in every plot
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"


# ---------------------------------------------------------------------------
# 1. Read the log
# ---------------------------------------------------------------------------

def load_log(path: Path) -> dict:
  """Read the CSV into arrays. Column names come from State_RLBase.cpp."""
  with open(path) as f:
    rows = list(csv.DictReader(f))

  def column(name):
    return np.array([float(r[name]) for r in rows])

  def block(prefix, count):
    return np.stack([column(f"{prefix}_{i}") for i in range(count)], axis=1)

  side = ("Lx", "Ly", "Lz", "Rx", "Ry", "Rz")  # left xyz, then right xyz
  return dict(
    t=column("timestep"),
    q=block("joint_pos", 27),  # policy order
    quat=np.stack([column(f"imu_quat_{c}") for c in "wxyz"], axis=1),  # w x y z
    goal=np.stack([column(f"hand_goal_{s}") for s in side], axis=1),
    target=np.stack([column(f"hand_target_{s}") for s in side], axis=1),
  )


# ---------------------------------------------------------------------------
# 2. Where the palms really were (same math as hand_pose_error)
# ---------------------------------------------------------------------------

def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
  """(N, 4) quaternions (w x y z) -> (N, 3, 3) rotation matrices."""
  w, x, y, z = (quat / np.linalg.norm(quat, axis=1, keepdims=True)).T
  return np.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ], axis=1).reshape(-1, 3, 3)


def tilt_matrix(quat: np.ndarray) -> np.ndarray:
  """Pelvis -> heading rotation: the IMU orientation with its yaw taken out."""
  full = quat_to_matrix(quat)
  yaw = np.arctan2(full[:, 1, 0], full[:, 0, 0])
  c, s, zero, one = np.cos(yaw), np.sin(yaw), np.zeros_like(yaw), np.ones_like(yaw)
  yaw_only = np.stack([c, -s, zero, s, c, zero, zero, zero, one], axis=1).reshape(-1, 3, 3)
  return np.einsum("nji,njk->nik", yaw_only, full)  # yaw_only transposed, times full


def forward_kinematics(q: np.ndarray):
  """Palm positions (N, 2, 3), palm rotations (N, 2, 3, 3) and foot heights (N, 2, 3),
  all relative to the pelvis, from the joint angles (policy order)."""
  model = mujoco.MjModel.from_xml_path(str(XML))
  data = mujoco.MjData(model)
  joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt) if model.jnt_type[j] != 0]  # skip the free joint
  address = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
  site = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

  palm, rot, foot = [np.zeros((len(q), 2) + shape) for shape in ((3,), (3, 3), (3,))]
  for k, angles in enumerate(q):
    data.qpos[:] = 0
    data.qpos[3] = 1.0  # pelvis at the origin, not rotated
    data.qpos[address] = angles
    mujoco.mj_kinematics(model, data)
    for s, side in enumerate(("left", "right")):
      palm[k, s] = data.site_xpos[site(f"{side}_palm")]
      rot[k, s] = data.site_xmat[site(f"{side}_palm")].reshape(3, 3)
      foot[k, s] = data.site_xpos[site(f"{side}_foot")]
  return palm, rot, foot


def palms_in_heading_frame(log: dict):
  """Palm positions and rotations where the policy sees them, plus the pelvis height."""
  palm, rot, foot = forward_kinematics(log["q"])
  R = tilt_matrix(log["quat"])  # pelvis -> heading, (N, 3, 3)
  foot_z = np.einsum("nij,nsj->nsi", R, foot)[..., 2]  # (N, 2)
  height = -foot_z.min(axis=1)  # the lower foot is on the floor (foot sites at z = 0)
  palm_h = np.einsum("nij,nsj->nsi", R, palm)
  palm_h[..., 2] += height[:, None]
  rot_h = np.einsum("nij,nsjk->nsik", R, rot)
  return palm_h, rot_h, height


# ---------------------------------------------------------------------------
# 3. Targets: which preset is active, and what orientation it asks for
# ---------------------------------------------------------------------------

def load_presets() -> dict:
  """name -> (positions (6,), quaternions (2, 4)) from deploy.yaml (w x y z)."""
  params = yaml.safe_load(open(DEPLOY_YAML))["observations"]["hand_pose_command"]["params"]
  presets = {n: (np.array(v["pos"]), np.array(v["quat"]).reshape(2, 4)) for n, v in params["presets"].items()}
  presets["default"] = (np.array(params["default"]), np.array(params["default_quat"]).reshape(2, 4))
  return presets


def preset_of_each_row(goal: np.ndarray, presets: dict) -> np.ndarray:
  """Name of the preset being requested at each moment (goals are copied from presets)."""
  name = np.full(len(goal), "", dtype=object)
  for n, (pos, _) in presets.items():
    name[np.linalg.norm(goal - pos, axis=1) < 2e-3] = n  # 2 mm: goals are exact copies
  return name


def rotation_angles(target_quats: np.ndarray, rot: np.ndarray) -> np.ndarray:
  """Angle [deg] between rot (N, 3, 3) and the quaternions target_quats (N, 4), row by row."""
  target = quat_to_matrix(target_quats)
  trace = np.einsum("nij,nij->n", rot, target)
  return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def slerp_step(q: np.ndarray, goal: np.ndarray, max_angle: float) -> np.ndarray:
  """Turn one quaternion toward another by at most max_angle (the robot's glideQuat).
  Taking the short way: if the dot product is negative, use -goal (the same rotation)."""
  if np.dot(q, goal) < 0:
    goal = -goal
  angle = 2 * np.arccos(np.clip(abs(np.dot(q, goal)), 0, 1))  # 0 to pi
  fraction = 1.0 if angle <= max_angle else max_angle / angle
  if angle < 1e-6:
    return goal
  half = angle / 2
  q = (np.sin((1 - fraction) * half) * q + np.sin(fraction * half) * goal) / np.sin(half)
  return q / np.linalg.norm(q)


def rebuild_target_orientation(names: np.ndarray, presets: dict, steps_per_second=50) -> np.ndarray:
  """The orientation target at every moment, (N, 2, 4), w x y z. The log does not store
  it, so we run the same glide the robot ran: start at the default, turn toward the
  current preset's orientation at max_angular_speed."""
  params = yaml.safe_load(open(DEPLOY_YAML))["observations"]["hand_pose_command"]["params"]
  max_angle = params["max_angular_speed"] / steps_per_second  # radians per log row
  target = np.repeat(presets["default"][1][None], len(names), axis=0).copy()
  for k in range(1, len(names)):
    goal = presets[names[k]][1] if names[k] else presets["default"][1]
    target[k] = [slerp_step(target[k - 1, h], goal[h], max_angle) for h in range(2)]
  return target


# ---------------------------------------------------------------------------
# 4. Numbers: errors, settled moments
# ---------------------------------------------------------------------------

def compute_errors(log, palm, rot, presets):
  """Everything the plots and the table need."""
  t = log["t"] - log["t"][0]
  target = log["target"].reshape(-1, 2, 3)
  pos_err = np.linalg.norm(target - palm, axis=2) * 100  # [cm], one per hand
  names = preset_of_each_row(log["goal"], presets)
  moving = np.linalg.norm(log["goal"] - log["target"], axis=1) > 1e-3  # still gliding
  # The orientation target glides too, at its own speed. It is not in the log, so rebuild it.
  target_quat = rebuild_target_orientation(names, presets)
  goal_quat = np.stack([presets[n][1] if n else presets["default"][1] for n in names])
  turn_left = np.stack([rotation_angles(target_quat[:, h], quat_to_matrix(goal_quat[:, h])) for h in range(2)], axis=1).max(axis=1)
  moving = moving | (turn_left > 1.0)  # [deg] the orientation has not arrived yet
  # Settled: the same goal for over 1 s, and the target has arrived.
  settled = np.zeros(len(t), bool)
  still = 0
  for k in range(len(t)):
    same_goal = k > 0 and names[k] == names[k - 1]
    still = still + 1 if (not moving[k] and same_goal) else 0
    settled[k] = still > 50  # 50 rows = 1 s
  # Rotation error: how far each palm is from its (rebuilt, gliding) orientation target.
  rot_err = np.stack([rotation_angles(target_quat[:, h], rot[:, h]) for h in range(2)], axis=1)
  w, x, y, z = log["quat"].T
  norm = np.linalg.norm(log["quat"], axis=1)
  tilt = np.degrees(np.arccos(np.clip(1 - 2 * ((x / norm) ** 2 + (y / norm) ** 2), -1, 1)))
  return dict(t=t, pos_err=pos_err, names=names, moving=moving, settled=settled, rot_err=rot_err, tilt=tilt)


def print_table(log, e, presets):
  print(f"\nrun length {e['t'][-1]:.0f} s | tilt max {e['tilt'].max():.1f} deg (safety limit 15)")
  print(f"position error, worse hand: median {np.median(e['pos_err'].max(1)):.2f} cm, p90 {np.quantile(e['pos_err'].max(1), .9):.2f} cm"
        f" (settled only: {np.median(e['pos_err'][e['settled']].max(1)):.2f} cm)")
  print("\npreset    held   pos err L / R [cm]     rot err L / R [deg]")
  for n in list(presets):
    use = e["settled"] & (e["names"] == n)
    if use.sum() > 10:
      print(f"{n:8s} {use.sum() / 50:5.1f} s   {np.median(e['pos_err'][use, 0]):4.2f} / {np.median(e['pos_err'][use, 1]):4.2f}"
            f"          {np.nanmedian(e['rot_err'][use, 0]):4.1f} / {np.nanmedian(e['rot_err'][use, 1]):4.1f}")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def style(ax, ylabel):
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.set_ylabel(ylabel, color=MUTED)
  ax.tick_params(colors=MUTED)


def mark_presets(ax, t, names, where_label=True):
  """Thin vertical line at each goal change. Labels alternate between two heights so
  short segments don't overlap."""
  start, row = 0, 0
  for k in range(1, len(t) + 1):
    if k == len(t) or names[k] != names[start]:
      if start > 0:
        ax.axvline(t[start], color=GRID, linewidth=1.2, zorder=0)
      if where_label and names[start]:
        ax.text((t[start] + t[k - 1]) / 2, 1.02 + 0.055 * (row % 2), names[start],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=MUTED)
        row += 1
      start = k


def plot_tracking(log, e, palm, path):
  t = e["t"]
  fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
  for h, (hand, color) in enumerate((("left hand", LEFT), ("right hand", RIGHT))):
    for a, axis in enumerate("xyz"):
      ax = axes[a, h]
      style(ax, f"{axis} [m]" if h == 0 else "")
      ax.plot(t, log["goal"][:, 3 * h + a], color="#b9b8b2", linewidth=1.5, label="goal (preset)")
      ax.plot(t, log["target"][:, 3 * h + a], color=INK, linewidth=1.2, linestyle="--", label="target (gliding)")
      ax.plot(t, palm[:, h, a], color=color, linewidth=2.0, label=f"palm, {hand} (measured)")
      mark_presets(ax, t, e["names"], where_label=(a == 0))
      if a == 0:
        ax.set_title(hand, color=INK, fontsize=11, pad=24)
    axes[2, h].set_xlabel("time [s]", color=MUTED)
  # One legend: goal and target once, then each hand's palm in its own colour.
  h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[0, 1].get_legend_handles_labels()
  fig.legend(h0[:2] + [h0[2], h1[2]], l0[:2] + [l0[2], l1[2]], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
  fig.suptitle("Hand tracking in the heading frame", color=INK, fontsize=13, y=0.99)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def plot_errors(e, path):
  t = e["t"]
  fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
  for ax, title in zip(axes, ("Position error", "Orientation error", "Pelvis tilt")):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=24 if title == "Position error" else 6)
  for ax in axes:
    style(ax, "")
  # 1. position error
  axes[0].set_ylabel("position error [cm]", color=MUTED)
  for h, (label, color) in enumerate((("left", LEFT), ("right", RIGHT))):
    axes[0].plot(t, e["pos_err"][:, h], color=color, linewidth=1.6, label=label)
  axes[0].axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
  axes[0].text(0.5, 0.94, "dotted line: the sim prediction, 0.5 cm", transform=axes[0].transAxes, ha="center", va="top", fontsize=8, color=MUTED)
  axes[0].set_ylim(0, max(2.5, e["pos_err"].max() * 1.05))
  # 2. orientation error: only defined where the wanted orientation is known (settled)
  axes[1].set_ylabel("orientation error [deg]", color=MUTED)
  axes[1].text(0.995, 0.94, "against the gliding orientation target", transform=axes[1].transAxes, ha="right", va="top", fontsize=8, color=MUTED)
  for h, (label, color) in enumerate((("left", LEFT), ("right", RIGHT))):
    axes[1].plot(t, e["rot_err"][:, h], color=color, linewidth=1.6, label=label)
  axes[1].set_ylim(0, max(6, np.quantile(e["rot_err"], 0.995) * 1.1))
  # 3. tilt
  axes[2].set_ylabel("tilt [deg]", color=MUTED)
  axes[2].plot(t, e["tilt"], color=INK, linewidth=1.4, label="pelvis tilt")
  axes[2].set_ylim(0, max(3, e["tilt"].max() * 1.2))
  axes[2].text(0.995, 0.92, "safety limit: 15 deg", transform=axes[2].transAxes, ha="right", fontsize=8, color=MUTED)
  axes[2].set_xlabel("time [s]", color=MUTED)
  for ax in axes:
    # Grey bands = the target (position or orientation) is still gliding.
    start = None
    for k in range(len(t)):
      if e["moving"][k] and start is None:
        start = k
      if start is not None and (not e["moving"][k] or k == len(t) - 1):
        ax.axvspan(t[start], t[k], color="#f0efea", zorder=0, linewidth=0)
        start = None
    mark_presets(ax, t, e["names"], where_label=(ax is axes[0]))
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("csv", nargs="?", help="log to plot (default: the newest one)")
  args = parser.parse_args()
  path = Path(args.csv) if args.csv else max(LOGS.glob("velocity_*.csv"), key=lambda p: p.stat().st_mtime)

  log = load_log(path)
  presets = load_presets()
  palm, rot, _ = palms_in_heading_frame(log)
  e = compute_errors(log, palm, rot, presets)

  OUT.mkdir(parents=True, exist_ok=True)
  tracking_png = OUT / f"{path.stem}_tracking.png"
  errors_png = OUT / f"{path.stem}_errors.png"
  plot_tracking(log, e, palm, tracking_png)
  plot_errors(e, errors_png)
  print(f"log: {path.name}")
  print_table(log, e, presets)
  print(f"\nsaved {tracking_png}\nsaved {errors_png}")


if __name__ == "__main__":
  main()
