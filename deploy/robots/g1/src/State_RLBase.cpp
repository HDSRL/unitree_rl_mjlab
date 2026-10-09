#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <stdexcept>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <array>
#include <mutex>
#include <cmath>
namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    // Data logging is optional and configured separately for each FSM state.
    // Mimic, for example, has no data_log block and will leave logging disabled.
    const auto data_log_cfg = cfg["data_log"];

    if (data_log_cfg && data_log_cfg["enabled"].as<bool>()) {
        data_log_enabled_ = true;
        data_log_output_dir_ =
            data_log_cfg["output_dir"].as<std::string>();
        // Keep log placement independent of the terminal's current directory.
        // param::proj_dir is deploy/robots/g1 at runtime.
        if (data_log_output_dir_.is_relative()) {
            data_log_output_dir_ = param::proj_dir / data_log_output_dir_;
        }
    }
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    //Check for Per-joint pos/vel/torque limits -> Passive
    const auto safety_cfg = cfg["safety"];
    if (safety_cfg && safety_cfg["enabled"].as<bool>()) {
        safety_q_min_ = safety_cfg["q_min"].as<std::vector<float>>();
        safety_q_max_ = safety_cfg["q_max"].as<std::vector<float>>();
        safety_dq_max_ = safety_cfg["dq_max"].as<std::vector<float>>();
        safety_tau_max_ = safety_cfg["tau_max"].as<std::vector<float>>();
        safety_tau_debounce_ticks_ = safety_cfg["tau_debounce_ms"].as<int>();

        // A short table would silently leave joints unchecked; fail before the robot moves.
        const auto joint_count = env->robot->data.joint_ids_map.size();
        if (safety_q_min_.size() != joint_count || safety_q_max_.size() != joint_count ||
            safety_dq_max_.size() != joint_count || safety_tau_max_.size() != joint_count) {
            throw std::runtime_error("safety: each limit array needs one entry per joint");
        }

        // Front of the list: a limit violation must win over any joystick transition.
        registered_checks.emplace(
            registered_checks.begin(),
            [this]() -> bool { return safety_limits_exceeded(); },
            FSMStringMap.right.at("Passive"));
        spdlog::info("Safety limits enabled for {}", state_string);
    } else {
        spdlog::warn("Safety limits DISABLED for {}", state_string);
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_RLBase::open_data_log()
{
    // Logging is optional. Disabled states, such as Mimic, do nothing here.
    if (!data_log_enabled_) {
        return;
    }

    try {
        std::filesystem::create_directories(data_log_output_dir_);

        // Human-readable timestamp makes each Velocity run a separate file.
        const auto wall_time = std::chrono::system_clock::now();
        const std::time_t time_value =
            std::chrono::system_clock::to_time_t(wall_time);

        std::tm local_time{};
        localtime_r(&time_value, &local_time);

        std::ostringstream filename;
        filename << "velocity_"
                 << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S")
                 << ".csv";

        const auto csv_path = data_log_output_dir_ / filename.str();
        data_log_file_.open(csv_path);

        // Logging failure must not prevent the controller from running.
        if (!data_log_file_) {
            spdlog::error(
                "Could not open data-log file: {}", csv_path.string());
            data_log_enabled_ = false;
            return;
        }

        // Shared names match the evaluation rollout.parquet schema where
        // equivalent signals exist. Extra deployment-only fields follow them.
        data_log_file_
            << "timestep,sequence,"
            << "command_vx,command_vy,command_yaw_rate,"
            << "root_com_ang_vel_b_x,"
            << "root_com_ang_vel_b_y,"
            << "root_com_ang_vel_b_z,"
            << "imu_quat_w,imu_quat_x,imu_quat_y,imu_quat_z,"
            << "joystick_lx,joystick_ly,joystick_rx";

        const auto joint_count = env->robot->data.joint_ids_map.size();

        // These names are identical to the evaluation joint-state columns.
        for (std::size_t index = 0; index < joint_count; ++index) {
            data_log_file_ << ",joint_pos_" << index;
        }
        for (std::size_t index = 0; index < joint_count; ++index) {
            data_log_file_ << ",joint_vel_" << index;
        }

        // Keep the hardware/simulator torque-estimate name explicit.
        for (std::size_t index = 0; index < joint_count; ++index) {
            data_log_file_ << ",joint_tau_est_" << index;
        }

        // Deployment-only: this is the processed position target sent to the robot.
        for (std::size_t index = 0; index < joint_count; ++index) {
            data_log_file_ << ",target_joint_pos_" << index;
        }

        data_log_file_ << '\n';

        // steady_clock is monotonic, so it is correct for timing plots.
        data_log_start_time_ = std::chrono::steady_clock::now();
        data_log_sequence_ = 0;

        spdlog::info("Velocity data log: {}", csv_path.string());
    }
    catch (const std::exception& error) {
        // Disk/path issues are diagnostic only; never stop control for logging.
        spdlog::error("Could not start Velocity data log: {}", error.what());
        data_log_enabled_ = false;
    }
}

void State_RLBase::write_data_log_sample()
{
    // Logging is optional and must never interfere with the policy itself.
    if (!data_log_enabled_ || !data_log_file_.is_open()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double timestep =
        std::chrono::duration<double>(now - data_log_start_time_).count();

    const auto joint_count = env->robot->data.joint_ids_map.size();

    // These are the processed position targets that State_RLBase::run()
    // publishes as LowCmd motor q targets.
    const auto target_joint_pos =
        env->action_manager->processed_actions();

    std::array<float, 4> imu_quat{};
    std::array<float, 3> root_ang_vel_b{};
    std::vector<float> joint_pos(joint_count);
    std::vector<float> joint_vel(joint_count);
    std::vector<float> joint_tau_est(joint_count);

    float joystick_lx = 0.0f;
    float joystick_ly = 0.0f;
    float joystick_rx = 0.0f;

    {
        // Take q, dq, tau_est, IMU, and joystick from one LowState snapshot.
        std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);

        const auto& imu = FSMState::lowstate->msg_.imu_state();

        for (std::size_t axis = 0; axis < 4; ++axis) {
            imu_quat[axis] = imu.quaternion()[axis];
        }

        for (std::size_t axis = 0; axis < 3; ++axis) {
            root_ang_vel_b[axis] = imu.gyroscope()[axis];
        }

        joystick_lx = FSMState::lowstate->joystick.lx();
        joystick_ly = FSMState::lowstate->joystick.ly();
        joystick_rx = FSMState::lowstate->joystick.rx();

        const auto& motor_state = FSMState::lowstate->msg_.motor_state();

        for (std::size_t index = 0; index < joint_count; ++index) {
            const auto motor_id = static_cast<std::size_t>(
                env->robot->data.joint_ids_map[index]);

            joint_pos[index] = motor_state[motor_id].q();
            joint_vel[index] = motor_state[motor_id].dq();
            joint_tau_est[index] = motor_state[motor_id].tau_est();
        }
    }

    // This is intentionally the same mapping as velocity_commands() in
    // observations.h, using the exact joystick snapshot recorded above.
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];

    const float command_vx = std::clamp(
        joystick_ly,
        ranges["lin_vel_x"][0].as<float>(),
        ranges["lin_vel_x"][1].as<float>());

    const float command_vy = std::clamp(
        -joystick_lx,
        ranges["lin_vel_y"][0].as<float>(),
        ranges["lin_vel_y"][1].as<float>());

    const float command_yaw_rate = std::clamp(
        -joystick_rx,
        ranges["ang_vel_z"][0].as<float>(),
        ranges["ang_vel_z"][1].as<float>());

    data_log_file_ << std::setprecision(9)
                   << timestep << ','
                   << data_log_sequence_++ << ','
                   << command_vx << ','
                   << command_vy << ','
                   << command_yaw_rate;

    for (const float value : root_ang_vel_b) {
        data_log_file_ << ',' << value;
    }

    for (const float value : imu_quat) {
        data_log_file_ << ',' << value;
    }

    data_log_file_ << ','
                   << joystick_lx << ','
                   << joystick_ly << ','
                   << joystick_rx;

    for (const float value : joint_pos) {
        data_log_file_ << ',' << value;
    }

    for (const float value : joint_vel) {
        data_log_file_ << ',' << value;
    }

    for (const float value : joint_tau_est) {
        data_log_file_ << ',' << value;
    }

    for (const float value : target_joint_pos) {
        data_log_file_ << ',' << value;
    }

    data_log_file_ << '\n';
}

void State_RLBase::close_data_log()
{
    if (!data_log_file_.is_open()) {
        return;
    }

    // Flush only while leaving Velocity, never in the policy loop.
    data_log_file_.flush();
    data_log_file_.close();
    spdlog::info("Velocity data log closed.");
}

bool State_RLBase::safety_limits_exceeded()
{
    // Read the freshest LowState (1 kHz), not env->robot->data (50 Hz, policy thread).
    std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
    const auto& motor_state = FSMState::lowstate->msg_.motor_state();

    int tau_motor = -1;
    for (std::size_t m = 0; m < safety_q_min_.size(); ++m) {
        const float q = motor_state[m].q();
        const float dq = motor_state[m].dq();
        const float tau = motor_state[m].tau_est();

        // Written as !(inside) so a NaN reading also counts as a violation.
        if (!(q >= safety_q_min_[m] && q <= safety_q_max_[m])) {
            spdlog::critical("SAFETY: motor {} position {:.3f} outside [{:.3f}, {:.3f}] -> Passive",
                             m, q, safety_q_min_[m], safety_q_max_[m]);
            return true;
        }
        if (!(std::fabs(dq) <= safety_dq_max_[m])) {
            spdlog::critical("SAFETY: motor {} velocity {:.2f} exceeds {:.2f} rad/s -> Passive",
                             m, dq, safety_dq_max_[m]);
            return true;
        }
        if (tau_motor < 0 && !(std::fabs(tau) <= safety_tau_max_[m])) {
            tau_motor = static_cast<int>(m);
        }
    }

    // Only sustained torque trips; single-tick spikes reset the count.
    safety_tau_over_count_ = (tau_motor >= 0) ? safety_tau_over_count_ + 1 : 0;
    if (safety_tau_over_count_ > safety_tau_debounce_ticks_) {
        spdlog::critical("SAFETY: motor {} torque {:.1f} over {:.1f} Nm for {} ms -> Passive",
                         tau_motor, motor_state[tau_motor].tau_est(),
                         safety_tau_max_[tau_motor], safety_tau_over_count_);
        return true;
    }
    return false;
}
