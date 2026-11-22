"""Build a library of reachable palm POSES (position + orientation) for the G1 27-DoF.

Offline, CPU only. What this script does, in plain words:
  1. Put the robot model in its default standing pose.
  2. Many times: pick random angles for the 14 arm joints (7 per arm).
  3. Compute where the two palms end up AND how they are rotated
     (forward kinematics).
  4. Throw the sample away if an arm touches the body or the other arm,
     or if the palms are outside the front "manipulation workspace".
  5. Save the surviving palm poses. Training samples hand goals from them,
     so every goal (position and orientation) is reachable without
     self-collision.

Why sample joint angles instead of palm poses directly? A random palm pose may
be unreachable; a palm pose computed FROM joint angles is reachable by
construction.

Frame of the saved poses: the ground-attached heading frame. The pelvis is at
x = y = 0 and faces +x with no yaw, z is height above the floor.

Output file (a dict saved with torch.save):
  "targets":        (N, 6)       float32, [left x, y, z, right x, y, z] in metres
  "targets_rotmat": (N, 2, 3, 3) float32, rotation matrix of [left, right] palm
  "default":        (6,)         float32, palm positions in the default pose
  "default_rotmat": (2, 3, 3)    float32, palm rotations in the default pose

History: v0 and v0.5 were trained on hand_targets_27dof.pt (positions only,
all arm joints sampled within 80% of their range), made by this script at
commit 6aa2261.

Run from the repo root:  python scripts/build_hand_targets.py   (~1.5 min)
"""

from pathlib import Path

import mujoco
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file is in scripts/)
XML = ROOT / "src/assets/robots/unitree_g1/xmls/g1_27dof.xml"
OUT = ROOT / "src/assets/robots/unitree_g1/hand_pose_targets_27dof.pt"

N_SAMPLES = 200_000  # more than v0's 100k: position + orientation is a bigger space
LIMIT_FRAC = 0.8  # shoulders and elbow: sample within the middle 80% of the range
WRIST_LIMIT_FRAC = 0.6  # wrists: narrower, for natural hand orientations and
                        # to keep the wrists away from their limits
PELVIS_HEIGHT = 0.79  # nominal standing pelvis height above the floor [m]
RANDOM_SEED = 0  # fixed seed: running the script twice gives the same file

# Default standing pose. Must match HOME_KEYFRAME in g1_27dof_constants.py.
# Joints not listed here stay at 0.
DEFAULT_POSE = {
  "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
  "left_knee_joint": 0.3, "right_knee_joint": 0.3,
  "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
  "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
  "left_elbow_joint": 0.87, "right_elbow_joint": 0.87,
  "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
}

# The 7 joints of one arm, from shoulder to wrist.
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")

# Words that appear in the names of the arms' collision shapes.
ARM_WORDS = ("shoulder", "elbow", "wrist", "hand")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def joint_id(model: mujoco.MjModel, name: str) -> int:
  """MuJoCo's index number for a joint, looked up by its name."""
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def site_id(model: mujoco.MjModel, name: str) -> int:
  """Index of a "site": a named point attached to a body (here: the palm centre)."""
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def default_standing_qpos(model: mujoco.MjModel) -> np.ndarray:
  """The full position vector (qpos) of the robot standing in its default pose.

  qpos layout: 7 numbers for the floating base (x, y, z, then a quaternion
  w, x, y, z), then one number per joint. Each joint knows where its number
  lives in qpos via model.jnt_qposadr ("qpos address").
  """
  qpos = model.qpos0.copy()  # model.qpos0 = everything at 0, base upright
  qpos[2] = PELVIS_HEIGHT  # index 2 = base height z
  for name, angle in DEFAULT_POSE.items():
    address = model.jnt_qposadr[joint_id(model, name)]
    qpos[address] = angle
  return qpos


def arm_joint_sampling_ranges(model: mujoco.MjModel):
  """Where the 14 arm joints live in qpos, and the range to sample each one from.

  Returns (addresses, centre, half_width): a random angle for joint i is
  centre[i] + half_width[i] * u with u uniform in [-1, 1].
  """
  arm_joint_ids = []
  fractions = []  # how much of each joint's range we sample from
  for side in ("left", "right"):  # left arm first, then right arm
    for joint in ARM_JOINTS:
      arm_joint_ids.append(joint_id(model, f"{side}_{joint}_joint"))
      fractions.append(WRIST_LIMIT_FRAC if joint.startswith("wrist") else LIMIT_FRAC)
  addresses = model.jnt_qposadr[arm_joint_ids]
  lower = model.jnt_range[arm_joint_ids, 0]  # joint limits from the XML
  upper = model.jnt_range[arm_joint_ids, 1]
  centre = (lower + upper) / 2
  half_width = (upper - lower) / 2 * np.array(fractions)  # stay away from limits
  return addresses, centre, half_width


def arm_collision_geoms(model: mujoco.MjModel) -> set:
  """IDs of every collision shape ("geom") that belongs to an arm.

  Used to check whether an arm touches the body or the other arm.
  """
  arm_geoms = set()  # a set: no duplicates, and "x in arm_geoms" is fast
  for geom_id in range(model.ngeom):  # MuJoCo numbers every shape 0, 1, 2, ...
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None:
      # Visual meshes usually have no name, and they can't collide anyway.
      continue
    for word in ARM_WORDS:
      if word in name:  # e.g. "elbow" is inside "left_elbow_yaw_collision"
        arm_geoms.add(geom_id)
        break  # one matching word is enough; go to the next shape
  return arm_geoms


def arm_is_touching_something(data: mujoco.MjData, arm_geoms: set) -> bool:
  """True if any current contact involves an arm shape.

  The XML has no floor, so every contact here is a self-contact:
  arm vs body, or arm vs the other arm.
  """
  for i in range(data.ncon):  # data.ncon = number of contacts right now
    contact = data.contact[i]  # each contact is between two geoms
    if contact.geom1 in arm_geoms or contact.geom2 in arm_geoms:
      return True
  return False


def palms_in_workspace(palms: np.ndarray) -> bool:
  """Keep only targets useful for manipulation in front of the robot.

  palms: 2x3 array. Row 0 = left palm, row 1 = right palm; columns = x, y, z.
  """
  left, right = palms[0], palms[1]
  both_in_front = left[0] > -0.05 and right[0] > -0.05  # not behind the body
  both_at_working_height = (0.6 < left[2] < 1.3) and (0.6 < right[2] < 1.3)
  hands_not_crossed = left[1] > 0.0 and right[1] < 0.0  # left hand stays left
  return bool(both_in_front and both_at_working_height and hands_not_crossed)


def palm_rotation_matrices(data: mujoco.MjData, palm_sites: list) -> np.ndarray:
  """The 3x3 rotation matrix of each palm, shape (2, 3, 3).

  MuJoCo stores each site's rotation matrix flattened as 9 numbers
  (site_xmat), so we reshape it back to 3x3. Indexing with a LIST returns a
  copy, so the result is not overwritten by the next mj_forward.
  """
  return data.site_xmat[palm_sites].reshape(2, 3, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  # MjModel = the robot description (fixed). MjData = the current state
  # (joint angles, computed positions, contacts), which we overwrite each sample.
  model = mujoco.MjModel.from_xml_path(str(XML))
  data = mujoco.MjData(model)

  standing_qpos = default_standing_qpos(model)
  arm_addresses, centre, half_width = arm_joint_sampling_ranges(model)
  palm_sites = [site_id(model, "left_palm"), site_id(model, "right_palm")]
  arm_geoms = arm_collision_geoms(model)

  # Palm poses in the default pose: where the hand targets start after a reset.
  # mj_forward computes all body/site positions AND detects contacts.
  data.qpos[:] = standing_qpos
  mujoco.mj_forward(model, data)
  default_palms = data.site_xpos[palm_sites].reshape(-1).copy()  # (6,)
  default_rotmats = palm_rotation_matrices(data, palm_sites)  # (2, 3, 3)

  rng = np.random.default_rng(RANDOM_SEED)
  targets = []  # palm positions, one (6,) row per kept sample
  target_rotmats = []  # palm rotations, one (2, 3, 3) entry per kept sample
  while len(targets) < N_SAMPLES:
    # 1. Start from the standing pose, then randomise only the 14 arm joints.
    data.qpos[:] = standing_qpos
    random_unit = rng.uniform(-1.0, 1.0, size=len(arm_addresses))  # 14 numbers
    data.qpos[arm_addresses] = centre + half_width * random_unit

    # 2. Forward kinematics + contact detection for this pose.
    mujoco.mj_forward(model, data)

    # 3. Reject poses where an arm hits the body or the other arm.
    if arm_is_touching_something(data, arm_geoms):
      continue

    # 4. Keep the palm poses if the palms are inside the front workspace.
    # Indexing with a LIST (palm_sites) makes NumPy return a COPY. A slice such as
    # site_xpos[2:4] would instead return a "view" into MuJoCo's memory, which
    # changes on the next mj_forward.
    palms = data.site_xpos[palm_sites]  # 2x3 copy: world position of each palm
    if palms_in_workspace(palms):
      # .copy() is defensive: not needed with the list index above, but it keeps
      # this correct if that indexing is ever changed to a slice (a view).
      targets.append(palms.reshape(-1).copy())
      # The orientation of the same palms, from the same joint angles:
      # feasible by construction, just like the position.
      target_rotmats.append(palm_rotation_matrices(data, palm_sites))

  torch.save(
    {
      "targets": torch.tensor(np.array(targets), dtype=torch.float32),  # (N, 6)
      "targets_rotmat": torch.tensor(np.array(target_rotmats), dtype=torch.float32),  # (N, 2, 3, 3)
      "default": torch.tensor(default_palms, dtype=torch.float32),  # (6,)
      "default_rotmat": torch.tensor(default_rotmats, dtype=torch.float32),  # (2, 3, 3)
    },
    OUT,
  )
  print(f"Saved {len(targets)} palm poses to {OUT}")


if __name__ == "__main__":
  main()
