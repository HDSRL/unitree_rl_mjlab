"""Unitree G1 velocity environment configurations."""
from dataclasses import replace

from src.assets.robots import (
  G1_27DOF_ACTION_SCALE,
  get_g1_27dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.observation_manager import ObservationTermCfg
from src import SRC_PATH
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.rewards import action_rate_l2_joints, body_upright, variable_posture
from src.tasks.velocity.mdp.hand_command import (
  HandPoseCommandCfg,
  HandPositionCommandCfg,
  hand_pose_error,
  track_hand_orientation,
  track_hand_orientation_trace,
  track_hand_position,
  track_hand_position_exp,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from src.tasks.velocity.mdp.curriculums import hand_training_stages, penalty_curriculum


HAND_TARGETS_PATH = SRC_PATH / "assets/robots/unitree_g1/hand_targets_27dof.pt"
HAND_POSE_TARGETS_PATH = SRC_PATH / "assets/robots/unitree_g1/hand_pose_targets_27dof.pt"

HAND_POSE_TARGETS_NATURAL_PATH = SRC_PATH / "assets/robots/unitree_g1/hand_pose_targets_27dof_natural.pt"

# Joint groups (regex). Each joint matches exactly one pattern in each group.
ARM_JOINTS = (r".*_shoulder_.*", r".*_elbow_.*", r".*_wrist_.*")
LEG_WAIST_JOINTS = (r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*", r"waist_.*")

# v2 training stages (ResGAC's A-E, compressed for PPO). Steps = iterations *
# STEPS_PER_ITERATION, which must match num_steps_per_env in rl_cfg.py (24).
STEPS_PER_ITERATION = 24
FULL_WALK = dict(lin_vel_x=(-0.5, 0.8), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8))
HAND_STAGES = [
  # A: stand, broad kernels (position ~14 cm, orientation ~26 deg wide).
  dict(step=0, standing=1.0, pos_var=0.02, pos_l1=0.0, rot_var=0.2),
  # B: stand, tighter.
  dict(step=1500 * STEPS_PER_ITERATION, standing=1.0, pos_var=0.01, pos_l1=0.0, rot_var=0.1),
  # C: start walking slowly; loosen position again (walking makes it harder).
  dict(step=3000 * STEPS_PER_ITERATION, standing=0.3,
       lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5),
       pos_var=0.02, pos_l1=0.0, rot_var=0.1),
  # D: full walking ranges, tighten.
  dict(step=6000 * STEPS_PER_ITERATION, standing=0.3, **FULL_WALK,
       pos_var=0.005, pos_l1=0.0, rot_var=0.05),
  # E: final precision, ResGAC's final kernels (~3 cm + L1, ~8 deg).
  dict(step=10000 * STEPS_PER_ITERATION, standing=0.3, **FULL_WALK,
       pos_var=0.001, pos_l1=0.01, rot_var=0.02),
]
# Style penalties ramped by penalty_curriculum. Safety terms (joint_pos_limits,
# self_collisions, is_terminated) are deliberately NOT here.
STYLE_PENALTIES = [
  "action_rate_legs", "action_rate_arms", "joint_acc_l2", "joint_acc_arms", "body_ang_vel",
  "angular_momentum", "foot_clearance", "foot_slip", "soft_landing", "stand_still",
]

# v2.1 resumes from v2's model_10000 (end of stage D). A resumed run restarts the
# env's step counter at 0, so the normal stage list would start again at stage A.
# Instead: only the final stage, active from step 0.
HAND_STAGES_FINAL_ONLY = [
  dict(step=0, standing=0.3, **FULL_WALK, pos_var=0.001, pos_l1=0.01, rot_var=0.02),
]



def unitree_g1_27dof_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_27dof_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_27DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_27dof_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_27dof_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

def unitree_g1_27dof_hands_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat velocity task plus palm position tracking (v0: position only)."""
  cfg = unitree_g1_27dof_flat_env_cfg(play=play)

  cfg.commands["hands"] = HandPositionCommandCfg(
    entity_name="robot",
    resampling_time_range=(2.0, 4.0),
    library_path=str(HAND_TARGETS_PATH),
    max_speed=0.5,
    debug_vis=True,
  )
  # Appended last, so the deploy obs layout only grows at the end.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["hand_command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "hands"}
    )

  # Walking speeds for manipulation (ResGAC uses up to 0.7), and more standing.
  # The velocity curriculum is removed: it would widen these ranges up to 2 m/s.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.3
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)
  cfg.curriculum.pop("command_vel", None)

  # Hand tracking: the coarse kernel pulls from far away, the fine one sharpens.
  # Total weight 2.0 = 2x velocity tracking, a similar ratio to ResGAC.
  cfg.rewards["track_hands"] = RewardTermCfg(
    func=track_hand_position,
    weight=1.0,
    params={"command_name": "hands", "std": 0.1},
  )
  cfg.rewards["track_hands_fine"] = RewardTermCfg(
    func=track_hand_position,
    weight=1.0,
    params={"command_name": "hands", "std": 0.03},
  )

  # Arms are free now: only legs and waist must stay still when standing.
  cfg.rewards["stand_still"].params["asset_cfg"].joint_names = (
    r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*", r"waist_.*",
  )

  # v0.5: arms must reach targets but keep a natural, symmetric posture.
  # v0 used 1.0 rad for every arm joint and drifted into contorted,
  # asymmetric poses. Wrists barely move the palm, so they get the tightest pull.
  ARM_STD, WRIST_STD = 0.5, 0.3
  # Dict keys are regex patterns matched against joint names. Each joint must
  # match exactly ONE key, otherwise mjlab raises a ValueError. So replace
  # entries instead of adding overlapping ones.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_hip_.*": 0.05, r".*_knee_.*": 0.05, r".*_ankle_.*": 0.05, r"waist_.*": 0.05,
    r".*_shoulder_.*": ARM_STD, r".*_elbow_.*": ARM_STD, r".*_wrist_.*": WRIST_STD,
  }
  for regime in ("std_walking", "std_running"):
    stds = cfg.rewards["pose"].params[regime]
    for key in stds:
      if "wrist" in key:
        stds[key] = WRIST_STD
      elif "shoulder" in key or "elbow" in key:
        stds[key] = ARM_STD

  # Joint-limit penalty starts at 85% of range, leaving a margin before the
  # deploy safety trip at 90%.
  # Why replace(): in Python, `a = b` makes both names point to ONE object.
  # robot.articulation IS the module-level G1_27DOF_ARTICULATION, shared by
  # every G1-27DoF task, so editing it in place would also change the
  # locomotion task. replace() builds a modified copy and leaves the shared
  # one untouched.
  robot = cfg.scene.entities["robot"]
  robot.articulation = replace(robot.articulation, soft_joint_pos_limit_factor=0.85)


  return cfg


def unitree_g1_27dof_hands_pose_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v1: palm position AND orientation tracking.

  Builds on the v0.5 hands task (one more layer on top). Everything not changed
  here is inherited: velocity commands, posture fixes, 85% soft joint limits.
  """
  cfg = unitree_g1_27dof_hands_flat_env_cfg(play=play)

  # Replace the position-only command with position + orientation.
  # The "hand_command" observation reads this command, so the policy now sees
  # 18 numbers instead of 6 (actor observation 98 -> 110).
  cfg.commands["hands"] = HandPoseCommandCfg(
    entity_name="robot",
    resampling_time_range=(2.0, 4.0),
    library_path=str(HAND_POSE_TARGETS_PATH),
    max_speed=0.5,  # palm position glide [m/s], same as v0.5
    max_angular_speed=1.0,  # palm orientation glide [rad/s]
    debug_vis=True,
  )

  # Orientation tracking, next to v0.5's position rewards (weights 1.0 + 1.0).
  # Coarse kernel (std 0.5 rad ~ 29 deg) pulls from far away; fine kernel
  # (std 0.2 rad ~ 11 deg) sharpens. Position : orientation = 2.0 : 1.5,
  # close to ResGAC's 4 : 3.
  cfg.rewards["track_hands_rot"] = RewardTermCfg(
    func=track_hand_orientation,
    weight=0.75,
    params={"command_name": "hands", "std": 0.5},
  )
  cfg.rewards["track_hands_rot_fine"] = RewardTermCfg(
    func=track_hand_orientation,
    weight=0.75,
    params={"command_name": "hands", "std": 0.2},
  )

  # The wrist angles are now decided by the orientation target, so v0.5's strong
  # pull toward the default wrist angles (std 0.3) would fight the task. Loosen
  # it. Shoulders and elbow keep 0.5: they choose the elbow swivel, the one
  # motion the task leaves free. Rule: regularize only what the task leaves free.
  for regime in ("std_standing", "std_walking", "std_running"):
    stds = cfg.rewards["pose"].params[regime]
    for key in stds:
      if "wrist" in key:
        stds[key] = 1.0

  return cfg


def unitree_g1_27dof_hands_v2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v2: ResGAC-style end-to-end hand pose tracking.

  Builds on v1 (position + orientation command). Changes, following ResGAC:
  pose-error observation, their reward forms, an upright reward, split posture
  and smoothness terms, natural targets. The values here are curriculum
  stage A (standing, broad kernels); the curriculum moves them on.
  """
  cfg = unitree_g1_27dof_hands_pose_flat_env_cfg(play=play)

  # Natural target library: v1's had too many twisted hand poses.
  cfg.commands["hands"].library_path = str(HAND_POSE_TARGETS_NATURAL_PATH)

  # Observation: palm pose error, computed the way the robot can (24 numbers).
  # Appended after hand_command, so the deploy obs layout only grows at the end.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["hand_pose_error"] = ObservationTermCfg(
      func=hand_pose_error, params={"command_name": "hands"}
    )

  # Hand rewards in ResGAC's form, replacing v1's four kernels. Start broad
  # (var 0.02 = ~14 cm, 0.2 = ~26 deg) so the policy gets a signal from far
  # away; the curriculum tightens them. Manipulation (2.0 + 1.5) : walking
  # (1.0 + 1.0) ~ 2 : 1, the same ratio as ResGAC.
  for name in ("track_hands", "track_hands_fine", "track_hands_rot", "track_hands_rot_fine"):
    del cfg.rewards[name]
  cfg.rewards["hand_position"] = RewardTermCfg(
    func=track_hand_position_exp,
    weight=2.0,
    params={"command_name": "hands", "var": 0.02, "l1_scale": 0.0},
  )
  cfg.rewards["hand_orientation"] = RewardTermCfg(
    func=track_hand_orientation_trace,
    weight=1.5,
    params={"command_name": "hands", "var": 0.2},
  )

  # Upright: ResGAC's pelvis-orientation reward replaces our tiny tilt penalty
  # (training used to end episodes only at 70 deg, the robot trips at 15).
  # Pelvis only: with the waist roll/pitch locked, the torso tilts exactly
  # like the pelvis, so a torso term would just duplicate it.
  del cfg.rewards["body_orientation_l2"]
  cfg.rewards["upright"] = RewardTermCfg(
    func=body_upright,
    weight=1.0,
    params={"var": 0.04, "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )

  # Posture split. Legs and waist keep their posture reward unchanged; the
  # arms get a separate, WEAK one (weight 0.3) whose only job is to pick a
  # comfortable elbow angle, so it no longer competes with hand tracking.
  # The std dicts must only contain keys for the joints in asset_cfg (each
  # regex key must match a joint), so drop the arm keys from the legs' dicts.
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"].joint_names = LEG_WAIST_JOINTS
  for regime in ("std_standing", "std_walking", "std_running"):
    pose.params[regime] = {
      key: std
      for key, std in pose.params[regime].items()
      if not any(word in key for word in ("shoulder", "elbow", "wrist"))
    }
  arm_std = {r".*_shoulder_.*": 0.5, r".*_elbow_.*": 0.5, r".*_wrist_.*": 1.0}
  cfg.rewards["pose_arms"] = RewardTermCfg(
    func=variable_posture,
    weight=0.3,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
      "command_name": "twist",
      "std_standing": arm_std,
      "std_walking": arm_std,
      "std_running": arm_std,
      "walking_threshold": 0.1,
      "running_threshold": 1.5,
    },
  )

  # Smoothness split: the arms get their own action-rate penalty, at half the
  # legs' weight (ResGAC's ratio), so the gait doesn't pay for the arms or
  # the other way round.
  del cfg.rewards["action_rate_l2"]
  cfg.rewards["action_rate_legs"] = RewardTermCfg(
    func=action_rate_l2_joints,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_WAIST_JOINTS)},
  )
  cfg.rewards["action_rate_arms"] = RewardTermCfg(
    func=action_rate_l2_joints,
    weight=-0.15,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
  )
  # Extra acceleration penalty on the arms only; with the global joint_acc_l2 (-2.5e-7)
  # the arms total -2.5e-6. Legs keep -2.5e-7, so the gait is not affected.
  cfg.rewards["joint_acc_arms"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.25e-6,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
  )


  # Guide terms: broad kernels that never tighten. Once the main terms are
  # tight, a hand far from its target still gets some signal (multi-scale).
  # A documented deviation from ResGAC (they have GAC doing the coarse work).
  cfg.rewards["hand_position_guide"] = RewardTermCfg(
    func=track_hand_position_exp,
    weight=0.5,
    params={"command_name": "hands", "var": 0.02, "l1_scale": 0.0},
  )
  cfg.rewards["hand_orientation_guide"] = RewardTermCfg(
    func=track_hand_orientation_trace,
    weight=0.25,
    params={"command_name": "hands", "var": 0.2},
  )

  if play:
    # Playing a trained policy: no training stages, use the final stage's walking.
    cfg.commands["twist"].rel_standing_envs = 0.3
    return cfg

  # Training: start at stage A (standing); hand_training_stages moves on from there.
  cfg.commands["twist"].rel_standing_envs = 1.0
  cfg.curriculum["hand_stages"] = CurriculumTermCfg(
    func=hand_training_stages, params={"stages": HAND_STAGES}
  )
  # Holosoma's PPO values: start at 10%, grow while episodes last > 750 of 1000 steps.
  cfg.curriculum["penalty_scale"] = CurriculumTermCfg(
    func=penalty_curriculum,
    params={
      "penalty_names": STYLE_PENALTIES,
      "initial_scale": 0.1,
      "level_up_steps": 750,
      "level_down_steps": 150,
      "rate": 0.00025,
    },
  )
  return cfg

def unitree_g1_27dof_hands_v21_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v2.1: v2 with a multi-scale final stage

  Finding from v2: its final stage (tight kernels only) reached 0.6 cm median
  but lost 15% of goals, because a hand more than ~5 cm off got almost no
  reward signal. The ONE change here: position rewards at three widths at once,
  so every distance has a useful signal:
    tight (~3 cm + L1, ResGAC's)  - precision near the target (curriculum term)
    mid   (~7 cm, new)            - bridges the gap
    broad (~14 cm, guide 0.5->1.5) - pulls far-away hands back
  """
  cfg = unitree_g1_27dof_hands_v2_flat_env_cfg(play=play)

  cfg.rewards["hand_position_mid"] = RewardTermCfg(
    func=track_hand_position_exp,
    weight=1.0,
    params={"command_name": "hands", "var": 0.005, "l1_scale": 0.0},
  )
  cfg.rewards["hand_position_guide"].weight = 1.5

  if play:
    return cfg

  # Measurement noise on the hand error: +-5 mm and +-1.7 deg (training only).
  cfg.observations["actor"].terms["hand_pose_error"].params.update(pos_noise=0.005, rot_noise=0.03)
  
  # v2.4: v2.3 nearly stopped walking (hand rewards ~6 vs velocity 1 + 1).
  # Make walking worth more. Arm penalties stay as in v2.3.
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 1.5

  # Resume from v2.3's model_15000: the step counter restarts at 0 on resume, so skip
  # the stage curriculum and apply the style penalties at full strength.
  cfg.curriculum["hand_stages"].params["stages"] = HAND_STAGES_FINAL_ONLY
  cfg.curriculum["penalty_scale"].params["initial_scale"] = 1.0
  
  return cfg
