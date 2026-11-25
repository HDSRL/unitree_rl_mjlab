#ifndef HAND_OBSERVATIONS_HPP
#define HAND_OBSERVATIONS_HPP

#include "isaaclab/envs/mdp/observations/observations.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "palm_kinematics.hpp"
#include "param.h"


// Observations for the whole-body hand policy (v2.1). Each REGISTER_OBSERVATION
// function runs once per policy step and returns the numbers the policy sees.
namespace isaaclab
{
namespace mdp
{

// Turn "target" toward "goal" by at most maxAngle [rad] (same math as training).
// angularDistance: the angle between two orientations, always between 0 and pi.
// slerp(t, goal): go the fraction t of the way along the shortest rotation.
inline Eigen::Quaternionf glideQuat(const Eigen::Quaternionf& target,
                                    const Eigen::Quaternionf& goal, float maxAngle)
{
    float angle = target.angularDistance(goal);
    float t = (angle > maxAngle) ? maxAngle / angle : 1.0f;  // 1 = arrive at the goal
    Eigen::Quaternionf next = target.slerp(t, goal);
    next.normalize();  // keeps rounding errors from building up
    return next;
}

// Append a rotation as "6D": column 0, then column 1 of its matrix (6 numbers).
// The third column is the cross product of the first two, so it adds nothing.
inline void append6D(const Eigen::Quaternionf& q, std::vector<float>& out)
{
    Eigen::Matrix3f R = q.toRotationMatrix();
    for (int col = 0; col < 2; col++) {
        for (int row = 0; row < 3; row++) {
            out.push_back(R(row, col));
        }
    }
}

// Read 4 numbers (w x y z) starting at v[i] as an Eigen quaternion.
inline Eigen::Quaternionf readQuat(const std::vector<float>& v, int i)
{
    return Eigen::Quaternionf(v[i], v[i + 1], v[i + 2], v[i + 3]);  // (w, x, y, z)
}

// 18 numbers: [L xyz, R xyz, L 6D, R 6D], the hand targets in the heading frame.
REGISTER_OBSERVATION(hand_pose_command)
{
    // 1. First call after a reset: start at the default palms (matches training).
    //    Also check the config once, so a typo stops the program at startup, not mid-run.
    if (env->hand_target.empty()) {
        env->hand_target = params["default"].as<std::vector<float>>();
        env->hand_target_quat = params["default_quat"].as<std::vector<float>>();
        env->hand_goal = env->hand_target;
        env->hand_goal_quat = env->hand_target_quat;
        for (std::string name : {"up", "down", "left", "right"}) {
            if (params["presets"][name]["pos"].as<std::vector<float>>().size() != 6 ||
                params["presets"][name]["quat"].as<std::vector<float>>().size() != 8) {
                std::cerr << "hand_pose_command: preset " << name << " needs 6 pos and 8 quat numbers" << std::endl;
                exit(1);
            }
        }
    }

    // 2. A held button picks a preset goal: position and orientation together, so
    //    the pair stays feasible. Use .pressed: .on_pressed lasts one LowState
    //    message (~2 ms) and this function runs every 20 ms.
    auto & joy = *env->robot->data.joystick;
    std::string preset = "";
    if (joy.up.pressed)    preset = "up";
    if (joy.down.pressed)  preset = "down";
    if (joy.left.pressed)  preset = "left";
    if (joy.right.pressed) preset = "right";
    if (preset != "") {
        env->hand_goal = params["presets"][preset]["pos"].as<std::vector<float>>();
        env->hand_goal_quat = params["presets"][preset]["quat"].as<std::vector<float>>();
    }
    if (joy.Y.pressed) {  // Y: back to the default palms
        env->hand_goal = params["default"].as<std::vector<float>>();
        env->hand_goal_quat = params["default_quat"].as<std::vector<float>>();
    }

    // 3. Clamp the goal positions to the training workspace box.
    const auto box_min = params["box_min"].as<std::vector<float>>();
    const auto box_max = params["box_max"].as<std::vector<float>>();
    for (size_t i = 0; i < env->hand_goal.size(); ++i) {
        env->hand_goal[i] = std::clamp(env->hand_goal[i], box_min[i], box_max[i]);
    }

    // 4. Glide each hand along a straight line toward its goal. The speed limit
    //    applies to the hand's 3D distance (as in training), not to each axis.
    const float step = params["max_speed"].as<float>() * env->step_dt;
    for (size_t h = 0; h < 2; h++) {
        float delta[3];
        float dist_sq = 0.0f;
        for (size_t i = 0; i < 3; i++) {
            delta[i] = env->hand_goal[h * 3 + i] - env->hand_target[h * 3 + i];
            dist_sq += delta[i] * delta[i];
        }
        const float dist = std::sqrt(dist_sq);
        const float fraction = (dist > step) ? (step / dist) : 1.0f;  // 1 = arrive at the goal
        for (size_t i = 0; i < 3; i++) {
            env->hand_target[h * 3 + i] += delta[i] * fraction;
        }
    }

    // 5. Rotate each hand's target toward its goal orientation, at most
    //    max_angular_speed radians per second.
    const float max_angle = params["max_angular_speed"].as<float>() * env->step_dt;
    for (int h = 0; h < 2; h++) {
        Eigen::Quaternionf target = readQuat(env->hand_target_quat, 4 * h);
        Eigen::Quaternionf goal = readQuat(env->hand_goal_quat, 4 * h);
        Eigen::Quaternionf next = glideQuat(target, goal, max_angle);
        env->hand_target_quat[4 * h + 0] = next.w();
        env->hand_target_quat[4 * h + 1] = next.x();
        env->hand_target_quat[4 * h + 2] = next.y();
        env->hand_target_quat[4 * h + 3] = next.z();
    }

    // 6. The 18 numbers the policy sees: both positions, then both 6D rotations.
    std::vector<float> obs = env->hand_target;
    for (int h = 0; h < 2; h++) {
        append6D(readQuat(env->hand_target_quat, 4 * h), obs);
    }
    return obs;
}

// The policy's 27 joints, in POLICY order (the order of joint_pos).
inline std::vector<std::string> policyJointNames()
{
    return {
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};
}

// How far each palm is from its target: 24 numbers, 12 per hand (left, then right):
//   [0:3]   position error = target - palm [m], heading frame
//   [3:6]   tanh(100 * position error): magnifies small errors, saturates at +-1
//   [6:12]  rotation error as 6D: R_palm^T * R_target, the rotation still needed,
//           seen from the palm. Aligned -> 1 0 0 0 1 0
// Kept separate from the observation below so it can be tested without a robot.
inline std::vector<float> computeHandPoseError(const PalmKinematics& fk, const Eigen::Quaternionf& rootQuat,
    const std::vector<float>& handTarget, const std::vector<float>& handTargetQuat, float footSiteHeight)
{
    // 1. Pelvis tilt from the IMU: roll and pitch only. The heading frame removes the
    //    yaw (both target and palm are relative to the robot's own heading).
    Eigen::Quaternionf q = rootQuat;
    if (q.norm() < 1e-6f) {
        q = Eigen::Quaternionf(1, 0, 0, 0);  // no IMU reading yet: treat as level
    }
    q.normalize();
    float yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));
    Eigen::Quaternionf yawQuat(std::cos(yaw / 2), 0, 0, std::sin(yaw / 2));  // (w, x, y, z)
    Eigen::Matrix3d R = (yawQuat.conjugate() * q).toRotationMatrix().cast<double>();  // pelvis -> heading

    // 2. Pelvis height above the floor: the lower foot stands on the floor.
    double leftFootZ = (R * fk.footPos[0]).z();
    double rightFootZ = (R * fk.footPos[1]).z();
    double pelvisHeight = footSiteHeight - std::min(leftFootZ, rightFootZ);

    // 3. One hand at a time.
    std::vector<float> obs;
    for (int h = 0; h < 2; h++) {
        Eigen::Vector3d palmPos = R * fk.palmPos[h];  // turned into the heading frame
        palmPos.z() += pelvisHeight;                 // measured from the floor
        Eigen::Matrix3d palmRot = R * fk.palmRot[h];  // same turn for the orientation

        Eigen::Vector3d targetPos(handTarget[3 * h], handTarget[3 * h + 1], handTarget[3 * h + 2]);
        Eigen::Matrix3d targetRot = readQuat(handTargetQuat, 4 * h).toRotationMatrix().cast<double>();

        Eigen::Vector3d posError = targetPos - palmPos;                 // [m]
        Eigen::Matrix3d rotError = palmRot.transpose() * targetRot;     // rotation still needed

        for (int i = 0; i < 3; i++) obs.push_back(posError[i]);
        for (int i = 0; i < 3; i++) obs.push_back(std::tanh(100 * posError[i]));
        for (int col = 0; col < 2; col++) {          // 6D: column 0, then column 1
            for (int row = 0; row < 3; row++) {
                obs.push_back(rotError(row, col));
            }
        }
    }
    return obs;
}

// 24 numbers: how far each palm is from its target, as the policy sees it.
REGISTER_OBSERVATION(hand_pose_error)
{
    const std::vector<std::string> jointNames = policyJointNames();
    if (env->robot->data.joint_pos.size() != (int)jointNames.size()) {
        std::cerr << "hand_pose_error: expected " << jointNames.size() << " joints, got "
                  << env->robot->data.joint_pos.size() << std::endl;
        exit(1);
    }
    // hand_pose_command (the target) must run earlier in the observation list.
    if (env->hand_target_quat.empty()) {
        std::cerr << "hand_pose_error: put hand_pose_command before it in deploy.yaml" << std::endl;
        exit(1);
    }

    // The robot model is loaded ONCE. "static" means: built on the first call and kept.
    // That first call is at startup, so a wrong path stops the program there.
    const std::string modelPath = (param::proj_dir / params["model_path"].as<std::string>()).string();
    static PalmKinematics fk(modelPath, jointNames);

    fk.compute(env->robot->data.joint_pos);
    return computeHandPoseError(fk, env->robot->data.root_quat_w, env->hand_target, env->hand_target_quat,
                                params["foot_site_height"].as<float>());
}


}  // namespace mdp
}  // namespace isaaclab

#endif
