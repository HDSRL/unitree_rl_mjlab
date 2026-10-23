"""Build a library of reachable palm targets for the G1 27-DoF (offline, CPU).

Samples arm joint angles, runs forward kinematics, rejects self-collisions and
targets outside a front manipulation workspace. Saves palm positions in the
ground-attached heading frame (pelvis xy at origin, yaw = 0, z up from ground).
"""

from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "src/assets/robots/unitree_g1/xmls/g1_27dof.xml"
OUT = ROOT / "src/assets/robots/unitree_g1/hand_targets_27dof.pt"

N_SAMPLES = 100_000
LIMIT_FRAC = 0.8  # Sample within 80% of each arm joint range.
PELVIS_HEIGHT = 0.79  # Nominal standing pelvis height [m].
# Must match HOME_KEYFRAME in g1_27dof_constants.py.
DEFAULT_POSE = {
  "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
  "left_knee_joint": 0.3, "right_knee_joint": 0.3,
  "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
  "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
  "left_elbow_joint": 0.87, "right_elbow_joint": 0.87,
  "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
}
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")


def in_workspace(p: np.ndarray) -> bool:
  """p: (2, 3) palms [left, right]. In front, waist-to-chest height, no crossing."""
  return bool(
    np.all(p[:, 0] > -0.05)
    and np.all((p[:, 2] > 0.6) & (p[:, 2] < 1.3))
    and p[0, 1] > 0.0
    and p[1, 1] < 0.0
  )


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(XML))
  d = mujoco.MjData(m)

  def joint_id(name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

  q0 = m.qpos0.copy()
  q0[2] = PELVIS_HEIGHT
  for name, value in DEFAULT_POSE.items():
    q0[m.jnt_qposadr[joint_id(name)]] = value

  arm_ids = [joint_id(f"{s}_{j}_joint") for s in ("left", "right") for j in ARM_JOINTS]
  arm_adr = m.jnt_qposadr[arm_ids]
  lo, hi = m.jnt_range[arm_ids].T
  mid, half = (lo + hi) / 2, (hi - lo) / 2 * LIMIT_FRAC

  palm_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_palm")
              for s in ("left", "right")]
  arm_geoms = {
    g for g in range(m.ngeom)
    if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
           for k in ("shoulder", "elbow", "wrist", "hand"))
  }

  # Palm positions at the default pose: where targets restart after a reset.
  d.qpos[:] = q0
  mujoco.mj_forward(m, d)
  default = d.site_xpos[palm_ids].reshape(-1).copy()

  rng = np.random.default_rng(0)
  targets = []
  while len(targets) < N_SAMPLES:
    d.qpos[:] = q0
    d.qpos[arm_adr] = mid + half * rng.uniform(-1.0, 1.0, size=len(arm_adr))
    mujoco.mj_forward(m, d)
    if any(d.contact[i].geom1 in arm_geoms or d.contact[i].geom2 in arm_geoms
           for i in range(d.ncon)):
      continue  # Arm touches the body or the other arm.
    palms = d.site_xpos[palm_ids]
    if in_workspace(palms):
      targets.append(palms.reshape(-1).copy())

  torch.save(
    {
      "targets": torch.tensor(np.array(targets), dtype=torch.float32),  # (N, 6)
      "default": torch.tensor(default, dtype=torch.float32),  # (6,)
    },
    OUT,
  )
  print(f"Saved {len(targets)} targets to {OUT}")


if __name__ == "__main__":
  main()
