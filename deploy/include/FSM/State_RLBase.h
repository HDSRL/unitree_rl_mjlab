// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        //Calling reset to avoid initial joint target positions being set to zero, which can cause a large initial torque spike.
        env->reset();
        // Do not mix timing results from separate Velocity sessions.
        max_env_step_ms_ = 0.0;
        max_data_log_ms_ = 0.0;
        max_policy_work_ms_ = 0.0;
        policy_work_overruns_ = 0;
        max_policy_wake_lateness_ms_ = 0.0;
        policy_wake_late_over_1ms_count_ = 0;
        // reset safety counters
        safety_tau_over_count_ = 0;
        // Open the CSV before the policy thread begins producing samples.
        open_data_log();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
            // Initialize timing
            auto sleepTill = clock::now() + dt;

            while (policy_thread_running)
            {
                const auto work_start = clock::now();
                env->step();
                const auto after_env_step = clock::now();

                // The policy loop is 50 Hz, so log one sample per policy step.
                write_data_log_sample();
                const auto work_end = clock::now();
                const double env_step_ms =
                    std::chrono::duration<double, std::milli>(
                        after_env_step - work_start).count();
                const double data_log_ms =
                    std::chrono::duration<double, std::milli>(
                        work_end - after_env_step).count();
                const double policy_work_ms =
                    std::chrono::duration<double, std::milli>(
                        work_end - work_start).count();

                if (env_step_ms > max_env_step_ms_) {
                    max_env_step_ms_ = env_step_ms;
                }
                if (data_log_ms > max_data_log_ms_) {
                    max_data_log_ms_ = data_log_ms;
                }
                if (policy_work_ms > max_policy_work_ms_) {
                    max_policy_work_ms_ = policy_work_ms;
                }

                // The 50 Hz policy budget is 20 ms.
                const bool work_overran = work_end > sleepTill;
                if (work_overran) {
                    ++policy_work_overruns_;
                }

                std::this_thread::sleep_until(sleepTill);
                const auto wake_time = clock::now();

                // Do not call a compute overrun a scheduler wake-up delay.
                if (!work_overran && wake_time > sleepTill) {
                    const double wake_lateness_ms =
                        std::chrono::duration<double, std::milli>(
                            wake_time - sleepTill).count();

                    if (wake_lateness_ms > max_policy_wake_lateness_ms_) {
                        max_policy_wake_lateness_ms_ = wake_lateness_ms;
                    }

                    // Ignore sub-millisecond OS scheduling variation.
                    if (wake_lateness_ms > 1.0) {
                        ++policy_wake_late_over_1ms_count_;
                    }
                }

                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        spdlog::info(
            "Velocity timing: max env->step={:.3f} ms, "
            "max data log={:.3f} ms, max policy work={:.3f} ms, "
            "work overruns={}, max wake lateness={:.3f} ms, "
            "wakes >1 ms={}",
            max_env_step_ms_,
            max_data_log_ms_,
            max_policy_work_ms_,
            policy_work_overruns_,
            max_policy_wake_lateness_ms_,
            policy_wake_late_over_1ms_count_);
        // The policy thread has stopped, so no more samples can be written.
        close_data_log();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    // Velocity-state CSV data log constant 50 Hz.
    // Open one CSV when Velocity starts; write from the 50 Hz policy thread.
    void open_data_log();
    // Store one synchronized policy/state sample for later plotting.
    void write_data_log_sample();
    // Close and flush the CSV after the policy thread stops.
    void close_data_log();
    bool data_log_enabled_ = false;
    std::filesystem::path data_log_output_dir_;
    std::ofstream data_log_file_;
    std::chrono::steady_clock::time_point data_log_start_time_;
    std::uint64_t data_log_sequence_ = 0;
    // Worst-case timing observed during the current Velocity session.
    double max_env_step_ms_ = 0.0;
    double max_data_log_ms_ = 0.0;
    double max_policy_work_ms_ = 0.0;
    std::uint64_t policy_work_overruns_ = 0;
    // Scheduler delay after sleep_until(); count only meaningful delays.
    double max_policy_wake_lateness_ms_ = 0.0;
    std::uint64_t policy_wake_late_over_1ms_count_ = 0;

    // Per-joint safety limits in SDK motor order, checked at 1 kHz by CtrlFSM.
    bool safety_limits_exceeded();
    std::vector<float> safety_q_min_;
    std::vector<float> safety_q_max_;
    std::vector<float> safety_dq_max_;
    std::vector<float> safety_tau_max_;
    int safety_tau_debounce_ticks_ = 0;
    int safety_tau_over_count_ = 0;

};

REGISTER_FSM(State_RLBase)
