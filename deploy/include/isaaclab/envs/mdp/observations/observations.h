// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

REGISTER_OBSERVATION(hand_command)
{
    // 1. First call after reset: start at the default palms (matches training).
    if (env->hand_target.empty()) {
        env->hand_target = params["default"].as<std::vector<float>>();
        env->hand_goal = env->hand_target;
    }

    // 2. A held button picks a preset goal. Use .pressed: .on_pressed lasts one
    //    LowState message (~2 ms) and this function runs every 20 ms.
    auto & joy = *env->robot->data.joystick;
    const auto presets = params["presets"];
    if (joy.up.pressed)    env->hand_goal = presets["up"].as<std::vector<float>>();
    if (joy.down.pressed)  env->hand_goal = presets["down"].as<std::vector<float>>();
    if (joy.left.pressed)  env->hand_goal = presets["left"].as<std::vector<float>>();
    if (joy.right.pressed) env->hand_goal = presets["right"].as<std::vector<float>>();
    if (joy.Y.pressed)     env->hand_goal = params["default"].as<std::vector<float>>();

    // 3. Clamp the goal to the training workspace box.
    const auto box_min = params["box_min"].as<std::vector<float>>();
    const auto box_max = params["box_max"].as<std::vector<float>>();
    for (size_t i = 0; i < env->hand_goal.size(); ++i) {
        env->hand_goal[i] = std::clamp(env->hand_goal[i], box_min[i], box_max[i]);
    }   
    
    // 4. Glide each hand along a straight line toward its goal. The speed limit
    //    applies to the hand's 3D distance (as in training), not to each axis.
    const float step = params["max_speed"].as<float>() * env->step_dt;
    for (size_t h = 0; h < 2; h++)
    {
        float delta[3];
        float dist_sq = 0.0f;
        for (size_t i = 0; i < 3; i++)
        {
            delta[i] = env->hand_goal[h * 3 + i] - env->hand_target[h * 3 + i];
            dist_sq += delta[i] * delta[i];
        }
        const float dist = std::sqrt(dist_sq);
        const float fraction = (dist > step) ? (step / dist) : 1.0f; //1 = arrive at the goal, <1 = glide toward the goal
        for (size_t i = 0; i < 3; i++)
        {
            env->hand_target[h * 3 + i] += delta[i] * fraction;
        }

    }
    


    return env->hand_target;
}

}
}