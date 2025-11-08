#include "px4_accel_env.hpp"
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

PX4AccelEnv::PX4AccelEnv(double rate_hz, double a_max, double ep_time_s,
                         double target_up_m, double target_lateral_m,
                         double start_up_m, double takeoff_kp, double takeoff_kd,
                         double takeoff_tol, int settle_steps, double takeoff_max_time,
                         bool safety_verbose)
    : rate_hz_(rate_hz),
      dt_(1.0 / rate_hz),
      a_max_(a_max),
      max_steps_(static_cast<int>(ep_time_s * rate_hz)),
      target_up_(target_up_m),
      target_lateral_(target_lateral_m),
      start_up_m_(start_up_m),
      takeoff_kp_(takeoff_kp),
      takeoff_kd_(takeoff_kd),
      takeoff_tol_(takeoff_tol),
      settle_steps_(settle_steps),
      takeoff_max_time_(takeoff_max_time),
      safety_verbose_(safety_verbose),
      // Reward weights (matching Python exactly)
      w_terminal_(100.0),
      w_improve_(10.0),
      w_height_(5.0),
      w_xdrift_(3.0),
      w_overshoot_(4.0),
      w_action_(0.01),
      w_vel_limit_(2.0),
      w_stagnation_(8.0),
      w_oscillation_(6.0),
      w_state_fault_(50.0),
      w_backward_(2.0),
      w_action_slew_(0.02),
      w_hold_bonus_(0.5),
      // Safety thresholds
      max_abs_vxy_(6.0),
      soft_abs_vxy_(4.5),
      max_abs_vz_(3.0),
      max_abs_x_(2.0),
      max_abs_z_(1.5),
      max_abs_y_(5.0),
      stale_timeout_s_(0.5),
      no_progress_horizon_(25),
      min_improvement_eps_(0.01),
      oscillation_window_(20),
      oscillation_flips_(6),
      max_action_slew_rate_(20.0),
      // PD gains
      hold_kp_(2.0),
      hold_kd_(1.0),
      node_(nullptr),
      spawn_xyz_({0.0f, 0.0f, 0.7f}),
      hover_z_ned_(0.0f),
      step_(0),
      episode_running_(false),
      ready_for_rl_(false),
      prev_action_(0.0),
      freq_monitor_start_time_(0.0),
      freq_monitor_step_count_(0),
      last_step_wall_time_(0.0)
{
    episode_origin_ned_.fill(0.0f);
    current_global_ned_.fill(0.0f);
    current_local_ned_.fill(0.0f);
    reset_reward_tracking();
}

PX4AccelEnv::~PX4AccelEnv() {
    close();
}

void PX4AccelEnv::reset_reward_tracking() {
    episode_reward_sum_ = 0.0;
    episode_improvement_reward_ = 0.0;
    episode_overshoot_penalty_ = 0.0;
    episode_action_penalty_ = 0.0;
    episode_height_penalty_ = 0.0;
    episode_x_drift_penalty_ = 0.0;
    episode_backward_penalty_ = 0.0;
    episode_action_slew_penalty_ = 0.0;
    episode_soft_vel_penalty_ = 0.0;
    episode_stagnation_penalty_ = 0.0;
    episode_oscillation_penalty_ = 0.0;
    episode_state_fault_penalty_ = 0.0;
    episode_hold_bonus_ = 0.0;
    last_distance_to_target_ = -1.0;
    last_lateral_position_ = 0.0;
    last_vy_local_ = 0.0;
    no_progress_steps_ = 0;
    vy_sign_history_.clear();
}

void PX4AccelEnv::ensure_node() {
    if (node_ == nullptr) {
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<PX4Node>(rate_hz_);
    }
}

std::string PX4AccelEnv::pose_to_str(const std::array<float, 3>& ned) const {
    char buf[100];
    snprintf(buf, sizeof(buf), "x=%.2f y=%.2f z=%.2f", ned[0], ned[1], ned[2]);
    return std::string(buf);
}

void PX4AccelEnv::disarm_and_stop() {
    if (node_ == nullptr) return;
    node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
    node_->wait_until([this]() { return !node_->is_armed(); }, 3.0);
}

torch::Tensor PX4AccelEnv::read_observation() {
    auto local = node_->get_last_local();
    if (local == nullptr) {
        // No data yet → return neutral observation
        return torch::zeros({6}, torch::kFloat32);
    }

    std::array<float, 3> global_ned = {local->x, local->y, local->z};
    current_global_ned_ = global_ned;

    if (episode_origin_ned_[0] == 0.0f && episode_origin_ned_[1] == 0.0f && 
        episode_origin_ned_[2] == 0.0f && step_ == 0) {
        episode_origin_ned_ = global_ned;
    }

    current_local_ned_[0] = global_ned[0] - episode_origin_ned_[0];
    current_local_ned_[1] = global_ned[1] - episode_origin_ned_[1];
    current_local_ned_[2] = global_ned[2] - episode_origin_ned_[2];

    float vx_ned = local->vx;
    float vy_ned = local->vy;
    float vz_ned = local->vz;

    auto obs = torch::zeros({6}, torch::kFloat32);
    obs[0] = current_local_ned_[0];
    obs[1] = current_local_ned_[1];
    obs[2] = current_local_ned_[2];
    obs[3] = vx_ned;
    obs[4] = vy_ned;
    obs[5] = vz_ned;

    return obs;
}

bool PX4AccelEnv::arm_offboard() {
    const int max_takeoff_retries = 3;
    
    for (int takeoff_attempt = 0; takeoff_attempt < max_takeoff_retries; ++takeoff_attempt) {
        if (takeoff_attempt > 0) {
            std::cout << "[STATE] Takeoff retry " << (takeoff_attempt + 1) 
                      << "/" << max_takeoff_retries << "..." << std::endl;
        }

        // Warmup Offboard: publish position mode + hold current position
        node_->wait_for_local(3.0);
        auto warm_local = node_->get_last_local();
        float hold_x = 0.0f, hold_y = 0.0f, hold_z = 0.0f;
        if (warm_local) {
            hold_x = warm_local->x;
            hold_y = warm_local->y;
            hold_z = warm_local->z;
        }
        for (int i = 0; i < 30; ++i) {
            node_->publish_offboard_heartbeat(true, false, false, false, false);
            node_->publish_position_setpoint(hold_x, hold_y, hold_z, 0.0f);
            rclcpp::spin_some(node_->get_node_base_interface());
            node_->sleep_dt();
        }

        // Switch to OFFBOARD
        std::cout << "[STATE] Entering OFFBOARD mode..." << std::endl;
        node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
        bool offboard_ok = node_->wait_until([this]() {
            return node_->get_nav_state() == px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD;
        }, 3.0);
        if (!offboard_ok) {
            std::cout << "[WARN] Failed to confirm OFFBOARD mode. Retrying takeoff sequence..." << std::endl;
            continue;
        }

        // Arm
        std::cout << "[STATE] Arming vehicle..." << std::endl;
        int arm_attempts = 0;
        const int max_arm_attempts = 3;
        bool armed = false;
        while (arm_attempts < max_arm_attempts && !armed) {
            node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
            node_->wait_until([this]() { return node_->is_armed(); }, 3.0);
            if (node_->is_armed()) {
                std::cout << "[STATE] Armed successfully." << std::endl;
                armed = true;
                break;
            } else {
                std::cout << "[WARN] Arm attempt " << (arm_attempts + 1) 
                          << " failed. Disarming, waiting 2s, and retrying..." << std::endl;
                node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
                node_->wait_until([this]() { return !node_->is_armed(); }, 2.0);
                std::this_thread::sleep_for(2s);
                arm_attempts++;
            }
        }
        if (!armed) {
            std::cout << "[ERROR] Failed to arm after multiple attempts." << std::endl;
            if (takeoff_attempt < max_takeoff_retries - 1) {
                continue;
            } else {
                return false;
            }
        }

        // Smooth climb using position setpoints
        node_->wait_for_local(10.0);
        auto local0 = node_->get_last_local();
        if (local0 == nullptr) {
            std::cout << "[ERROR] No VehicleLocalPosition messages for takeoff." << std::endl;
            if (takeoff_attempt < max_takeoff_retries - 1) {
                std::cout << "[STATE] Landing and disarming before retry..." << std::endl;
                node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
                std::this_thread::sleep_for(2s);
                node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
                node_->wait_until([this]() { return !node_->is_armed(); }, 3.0);
                std::this_thread::sleep_for(2s);
                continue;
            } else {
                return false;
            }
        }

        float pre_z0 = local0->z;
        float target_z_ned = pre_z0 - start_up_m_;  // up in NED is negative
        int steps = static_cast<int>(3.0 / dt_);
        std::cout << "[STATE] Performing smooth takeoff to " << start_up_m_ << " m..." << std::endl;
        
        float hold_x0 = episode_origin_ned_[0];
        float hold_y0 = episode_origin_ned_[1];
        for (int i = 0; i < steps; ++i) {
            float alpha = static_cast<float>(i + 1) / steps;
            float interp_z = pre_z0 + alpha * (target_z_ned - pre_z0);
            node_->publish_offboard_heartbeat(true, false, false, false, false);
            node_->publish_position_setpoint(hold_x0, hold_y0, interp_z, 0.0f);
            rclcpp::spin_some(node_->get_node_base_interface());
            node_->sleep_dt();
        }

        // Hold hover for stabilization
        std::cout << "[STATE] Hovering to stabilize..." << std::endl;
        auto hover_t0 = std::chrono::steady_clock::now();
        auto stable_start = std::chrono::steady_clock::time_point();
        bool hover_success = false;
        
        while (true) {
            node_->publish_offboard_heartbeat(true, false, false, false, false);
            node_->publish_position_setpoint(hold_x0, hold_y0, target_z_ned, 0.0f);
            rclcpp::spin_some(node_->get_node_base_interface());
            node_->sleep_dt();
            
            auto now = std::chrono::steady_clock::now();
            auto local_now = node_->get_last_local();
            if (local_now != nullptr) {
                float z_err = std::abs(local_now->z - target_z_ned);
                if (z_err <= 0.5f) {
                    if (stable_start == std::chrono::steady_clock::time_point()) {
                        stable_start = now;
                    }
                    if (std::chrono::duration<double>(now - stable_start).count() >= 1.0) {
                        std::cout << "[STATE] Hover stabilized within 0.5 m for 1.0 s." << std::endl;
                        hover_success = true;
                        break;
                    }
                } else {
                    stable_start = std::chrono::steady_clock::time_point();
                }
            }
            if (std::chrono::duration<double>(now - hover_t0).count() >= 10.0) {
                std::cout << "[WARN] Hover stabilization timeout (10 s)." << std::endl;
                break;
            }
        }

        if (hover_success) {
            // Re-zero origin
            node_->wait_for_local(1.0);
            auto last_local = node_->get_last_local();
            if (last_local != nullptr) {
                episode_origin_ned_[0] = last_local->x;
                episode_origin_ned_[1] = last_local->y;
                episode_origin_ned_[2] = last_local->z;
                current_global_ned_ = episode_origin_ned_;
                current_local_ned_.fill(0.0f);
            }
            hover_z_ned_ = target_z_ned;
            std::cout << "[STATE] Hover complete — switching to lateral RL control." << std::endl;
            ready_for_rl_ = true;
            return true;
        } else {
            if (takeoff_attempt < max_takeoff_retries - 1) {
                std::cout << "[STATE] Landing and disarming before retry..." << std::endl;
                node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
                std::this_thread::sleep_for(2s);
                node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
                node_->wait_until([this]() { return !node_->is_armed(); }, 3.0);
                std::this_thread::sleep_for(2s);
            } else {
                std::cout << "[ERROR] Failed to stabilize hover after all retry attempts." << std::endl;
                return false;
            }
        }
    }
    return false;
}

torch::Tensor PX4AccelEnv::reset() {
    ensure_node();
    current_global_ned_.fill(0.0f);
    current_local_ned_.fill(0.0f);
    episode_origin_ned_.fill(0.0f);
    reset_reward_tracking();

    std::cout << "[STATE] Waiting for PX4 local position..." << std::endl;
    bool ok = node_->wait_for_local(10.0);
    if (!ok || node_->get_last_local() == nullptr) {
        throw std::runtime_error("No VehicleLocalPosition messages received. Check bridge/topic.");
    }

    // Capture spawn pose once (global reference) for consistent episode resets
    if (!spawn_captured_) {
        auto first = node_->get_last_local();
        spawn_xyz_[0] = first->x;
        spawn_xyz_[1] = first->y;
        spawn_xyz_[2] = first->z;
        spawn_captured_ = true;
        std::cout << "[STATE] Captured spawn global NED: " << pose_to_str(spawn_xyz_) << std::endl;
    }

    // If previous episode ended mid-air, command land and wait until grounded
    auto last_local_pre = node_->get_last_local();
    if (last_local_pre && std::abs(last_local_pre->z - spawn_xyz_[2]) > 0.3f) {
        std::cout << "[STATE] Episode reset: landing to ground reference before starting new episode..." << std::endl;
        node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
        if (!wait_until_grounded(12.0)) {
            std::cout << "[WARN] Grounding timeout; proceeding with best available position." << std::endl;
        } else {
            std::cout << "[STATE] Grounded and stabilized." << std::endl;
        }
    }

    // Ensure disarmed before new cycle
    disarm_and_stop();
    std::cout << "[STATE] Disarmed prior to new episode." << std::endl;

    // Grab starting origin
    node_->wait_for_local(2.0);
    auto local = node_->get_last_local();
    if (local == nullptr) {
        throw std::runtime_error("No VehicleLocalPosition messages received.");
    }
    // Use spawn global reference to zero local coordinates, ensuring consistent reset
    episode_origin_ned_[0] = spawn_xyz_[0];
    episode_origin_ned_[1] = spawn_xyz_[1];
    episode_origin_ned_[2] = spawn_xyz_[2];
    current_global_ned_[0] = local->x;
    current_global_ned_[1] = local->y;
    current_global_ned_[2] = local->z;
    current_local_ned_[0] = current_global_ned_[0] - episode_origin_ned_[0];
    current_local_ned_[1] = current_global_ned_[1] - episode_origin_ned_[1];
    current_local_ned_[2] = current_global_ned_[2] - episode_origin_ned_[2];
    hover_z_ned_ = 0.0f;

    // Arm and takeoff
    ready_for_rl_ = false;
    ok = arm_offboard();
    if (!ok || !ready_for_rl_) {
        throw std::runtime_error("Failed to arm/hover before RL.");
    }

    step_ = 0;
    episode_running_ = true;

    auto obs = read_observation();
    std::cout << "[STATE] Start pose:" << std::endl;
    std::cout << "         Global NED: " << pose_to_str(current_global_ned_) << std::endl;
    std::cout << "         Local NED:  " << pose_to_str(current_local_ned_) << std::endl;
    std::cout << "[STATE] RL control begins." << std::endl;

    // Reset frequency monitor
    freq_monitor_start_time_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    freq_monitor_step_count_ = 0;
    last_step_wall_time_ = freq_monitor_start_time_;

    return obs;
}

bool PX4AccelEnv::wait_until_grounded(double timeout_s, double vz_tol, double stable_s) {
    auto start = std::chrono::steady_clock::now();
    auto stable_start = std::chrono::steady_clock::time_point();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < timeout_s) {
        node_->publish_offboard_heartbeat(true, false, false, false, false);
        rclcpp::spin_some(node_->get_node_base_interface());
        node_->sleep_dt();
        auto lp = node_->get_last_local();
        if (lp) {
            float dz = std::abs(lp->z - spawn_xyz_[2]);
            float vz = std::abs(lp->vz);
            if (dz < 0.25f && vz < vz_tol) {
                if (stable_start == std::chrono::steady_clock::time_point()) {
                    stable_start = std::chrono::steady_clock::now();
                }
                double stable_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - stable_start).count();
                if (stable_elapsed >= stable_s) {
                    return true;
                }
            } else {
                stable_start = std::chrono::steady_clock::time_point();
            }
        }
    }
    return false;
}

StepResult PX4AccelEnv::step(const torch::Tensor& action) {
    if (!episode_running_) {
        throw std::runtime_error("Episode not running. Call reset().");
    }
    if (!ready_for_rl_) {
        throw std::runtime_error("RL step() called before PX4 was ready for RL control.");
    }

    StepResult result;

    // Parse action with safety checks (match Python env behavior)
    bool fault_action = false;
    torch::Tensor action_checked;
    if (action.defined()) {
        action_checked = action.to(torch::kCPU).flatten();
    }
    if (!action_checked.defined() || action_checked.numel() < 1) {
        fault_action = true;
        action_checked = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        action_checked = action_checked.to(torch::kFloat32);
        auto invalid_mask = torch::isnan(action_checked) | torch::isinf(action_checked);
        if (invalid_mask.any().item<bool>()) {
            fault_action = true;
            action_checked = torch::zeros_like(action_checked);
        }
    }
    if (fault_action && safety_verbose_) {
        std::cout << "[SAFETY] Invalid or NaN action detected; substituting zero command." << std::endl;
    }
    float raw_action = action_checked[0].item<float>();
    float a_lat_req = std::clamp(raw_action, -static_cast<float>(a_max_), static_cast<float>(a_max_));
    if (std::abs(raw_action) > static_cast<float>(a_max_) && safety_verbose_) {
        std::cout << "[SAFETY] Action saturated from " << raw_action 
                  << " to " << a_lat_req << std::endl;
    }

    // Determine effective dt: use configured dt_ when >0, else measured wall-time since last step
    double now_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double runtime_dt = (last_step_wall_time_ > 0.0) ? (now_wall - last_step_wall_time_) : dt_;
    last_step_wall_time_ = now_wall;
    double eff_dt = (dt_ > 0.0) ? dt_ : std::max(1e-4, runtime_dt);

    // Apply slew rate limit using effective dt
    float da_allowed = static_cast<float>(max_action_slew_rate_ * eff_dt);
    float a_lat = std::clamp(a_lat_req, 
                             static_cast<float>(prev_action_ - da_allowed),
                             static_cast<float>(prev_action_ + da_allowed));
    double slew_penalty = w_action_slew_ * std::abs(a_lat - a_lat_req);

    // Publish acceleration setpoint
    node_->publish_offboard_heartbeat(false, false, true, false, false);
    auto lp = node_->get_last_local();
    float a_ned_z = -9.81f;
    bool a_z_sat = false;
    
    if (lp != nullptr && hover_z_ned_ != 0.0f) {
        float z_ned = lp->z;
        float vz_ned = lp->vz;
        float height_up = -(z_ned - hover_z_ned_);
        float vz_up = -vz_ned;
        float a_up_corr = hold_kp_ * (0.0f - height_up) + hold_kd_ * (0.0f - vz_up);
        a_ned_z = -9.81f - a_up_corr;
        if (a_ned_z < -15.0f) {
            a_ned_z = -15.0f;
            a_z_sat = true;
        } else if (a_ned_z > -5.0f) {
            a_ned_z = -5.0f;
            a_z_sat = true;
        }
    }
    
    node_->publish_accel_setpoint_lat(a_lat, a_ned_z);
    rclcpp::spin_some(node_->get_node_base_interface());
    if (rate_hz_ > 0.0) {
        node_->sleep_dt();
    }

    // Read observation
    step_++;
    auto obs = read_observation();
    float x_m = obs[0].item<float>();
    float y_m = obs[1].item<float>();
    float z_m = obs[2].item<float>();
    float vx = obs[3].item<float>();
    float vy = obs[4].item<float>();
    float vz = obs[5].item<float>();

    // Validate observation
    bool obs_fault = false;
    if (std::isnan(x_m) || std::isnan(y_m) || std::isnan(z_m) ||
        std::isnan(vx) || std::isnan(vy) || std::isnan(vz) ||
        std::isinf(x_m) || std::isinf(y_m) || std::isinf(z_m) ||
        std::isinf(vx) || std::isinf(vy) || std::isinf(vz)) {
        obs_fault = true;
        if (safety_verbose_) {
            std::cout << "[SAFETY] Observation contains NaN/Inf values." << std::endl;
        }
    }

    double stale_age = 0.0;
    if (node_->get_last_local_time() > 0.0) {
        stale_age = node_->now().seconds() - node_->get_last_local_time();
    } else {
        stale_age = std::numeric_limits<double>::infinity();
    }
    bool stale = stale_age > stale_timeout_s_;
    if (stale && safety_verbose_) {
        std::cout << "[SAFETY] Local position stale for " << stale_age 
                  << "s (> " << stale_timeout_s_ << "s)" << std::endl;
    }

    // Frequency monitor
    freq_monitor_step_count_++;
    if (freq_monitor_step_count_ % 100 == 0) {
        double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        double elapsed = now - freq_monitor_start_time_;
        if (elapsed > 0) {
            double freq = freq_monitor_step_count_ / elapsed;
            std::cout << "[FREQ] Control loop: " << freq << " Hz over last " 
                      << freq_monitor_step_count_ << " steps." << std::endl;
        }
        freq_monitor_start_time_ = now;
        freq_monitor_step_count_ = 0;
    }

    // Compute reward components
    double target_local_lateral = target_lateral_;
    double error_lateral = target_local_lateral - y_m;
    double distance_to_target = std::abs(error_lateral);
    double overshoot = std::max(0.0, static_cast<double>(y_m) - target_local_lateral);
    double height_error = std::abs(z_m);
    double x_drift = std::abs(x_m);

    // Improvement
    double improvement = 0.0;
    if (last_distance_to_target_ >= 0.0) {
        improvement = last_distance_to_target_ - distance_to_target;
    }

    // Reward terms
    double improvement_term = w_improve_ * improvement;
    double overshoot_term = w_overshoot_ * overshoot;
    double action_term = w_action_ * (a_lat * a_lat);
    double height_term = w_height_ * height_error;
    double x_drift_term = w_xdrift_ * x_drift;

    // Backward penalty
    double backward_penalty = 0.0;
    if (distance_to_target > 0.02) {
        bool dir_ok = (std::copysign(1.0, error_lateral) * vy) >= -0.02;
        if (!dir_ok) {
            backward_penalty = w_backward_ * std::abs(vy);
        }
    }

    // Hold bonus
    double hold_bonus = (distance_to_target < 0.05) ? w_hold_bonus_ : 0.0;

    // Soft velocity penalty
    double soft_vel_penalty = 0.0;
    double horiz_speed = std::sqrt(vx * vx + vy * vy);
    if (horiz_speed > soft_abs_vxy_) {
        soft_vel_penalty = w_vel_limit_ * (horiz_speed - soft_abs_vxy_);
    }

    // Stagnation
    if (improvement < min_improvement_eps_) {
        no_progress_steps_++;
    } else {
        no_progress_steps_ = 0;
    }
    double stagnation_penalty = 0.0;
    if (no_progress_steps_ > no_progress_horizon_) {
        stagnation_penalty = w_stagnation_ * (no_progress_steps_ - no_progress_horizon_);
    }

    // Oscillation
    vy_sign_history_.push_back(std::copysign(1.0, vy));
    if (static_cast<int>(vy_sign_history_.size()) > oscillation_window_) {
        vy_sign_history_.pop_front();
    }
    double oscillation_penalty = 0.0;
    if (static_cast<int>(vy_sign_history_.size()) == oscillation_window_) {
        int flips = 0;
        for (size_t i = 1; i < vy_sign_history_.size(); ++i) {
            if (vy_sign_history_[i] != vy_sign_history_[i - 1]) {
                flips++;
            }
        }
        if (flips >= oscillation_flips_) {
            oscillation_penalty = w_oscillation_ * (flips - oscillation_flips_ + 1);
            if (safety_verbose_) {
                std::cout << "[SAFETY] Oscillation detected: " << flips 
                          << " sign flips in last " << oscillation_window_ << " steps." << std::endl;
            }
        }
    }

    // Fault penalties
    double state_fault_penalty = 0.0;
    if (obs_fault || fault_action) {
        state_fault_penalty += w_state_fault_;
    }
    if (stale) {
        state_fault_penalty += w_state_fault_ * 0.5;
    }

    double reward = improvement_term - overshoot_term - action_term - height_term
                    - x_drift_term - backward_penalty - slew_penalty - soft_vel_penalty
                    - stagnation_penalty - oscillation_penalty - state_fault_penalty
                    + hold_bonus;

    // Track components
    episode_reward_sum_ += reward;
    episode_improvement_reward_ += improvement_term;
    episode_overshoot_penalty_ += overshoot_term;
    episode_action_penalty_ += action_term;
    episode_height_penalty_ += height_term;
    episode_x_drift_penalty_ += x_drift_term;
    episode_backward_penalty_ += backward_penalty;
    episode_action_slew_penalty_ += slew_penalty;
    episode_soft_vel_penalty_ += soft_vel_penalty;
    episode_stagnation_penalty_ += stagnation_penalty;
    episode_oscillation_penalty_ += oscillation_penalty;
    episode_state_fault_penalty_ += state_fault_penalty;
    episode_hold_bonus_ += hold_bonus;
    last_distance_to_target_ = distance_to_target;
    last_lateral_position_ = y_m;
    last_vy_local_ = vy;
    prev_action_ = a_lat;

    // Termination conditions
    bool terminated = false;
    if (height_error > max_abs_z_) {
        std::cout << "[SAFETY] Episode terminated: excessive height deviation (" 
                  << height_error << "m)" << std::endl;
        terminated = true;
    } else if (x_drift > max_abs_x_) {
        std::cout << "[SAFETY] Episode terminated: excessive X-axis drift (" 
                  << x_drift << "m)" << std::endl;
        terminated = true;
    } else if (std::abs(y_m) > max_abs_y_) {
        std::cout << "[SAFETY] Episode terminated: excessive lateral displacement (" 
                  << y_m << "m)" << std::endl;
        terminated = true;
    } else if (horiz_speed > max_abs_vxy_) {
        std::cout << "[SAFETY] Episode terminated: excessive horizontal speed (" 
                  << horiz_speed << "m/s)" << std::endl;
        terminated = true;
    } else if (stale) {
        std::cout << "[SAFETY] Episode terminated: stale local position (" 
                  << stale_age << "s)" << std::endl;
        terminated = true;
    } else if (obs_fault) {
        std::cout << "[SAFETY] Episode terminated: invalid observation values (NaN/Inf)." << std::endl;
        terminated = true;
    } else if (!node_->is_armed() || 
               node_->get_nav_state() != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD) {
        std::cout << "[SAFETY] Episode terminated: vehicle left OFFBOARD or disarmed." << std::endl;
        terminated = true;
    }

    bool truncated = (step_ >= max_steps_);

    // Terminal bonus
    if (terminated || truncated) {
        double terminal_bonus = -w_terminal_ * distance_to_target;
        reward += terminal_bonus;
        episode_reward_sum_ += terminal_bonus;
    }

    if (terminated || truncated) {
        std::cout << "[STATE] End pose:" << std::endl;
        std::cout << "         Global NED: " << pose_to_str(current_global_ned_) << std::endl;
        std::cout << "         Local NED:  " << pose_to_str(current_local_ned_) << std::endl;
        std::cout << "[REWARD] Episode distance to target (local Y): " 
                  << distance_to_target << " m (target=" << target_lateral_ 
                  << " m, final=" << y_m << " m)" << std::endl;
        std::cout << "[REWARD] Episode return breakdown:" << std::endl;
        std::cout << "         + Improvement reward:  " << episode_improvement_reward_ << std::endl;
        std::cout << "         - Overshoot penalty:   " << episode_overshoot_penalty_ << std::endl;
        std::cout << "         - Action penalty:      " << episode_action_penalty_ << std::endl;
        std::cout << "         - Height penalty:      " << episode_height_penalty_ << std::endl;
        std::cout << "         - X-drift penalty:     " << episode_x_drift_penalty_ << std::endl;
        std::cout << "         - Backward penalty:    " << episode_backward_penalty_ << std::endl;
        std::cout << "         - Action slew penalty: " << episode_action_slew_penalty_ << std::endl;
        std::cout << "         - Soft vel penalty:    " << episode_soft_vel_penalty_ << std::endl;
        std::cout << "         - Stagnation penalty:  " << episode_stagnation_penalty_ << std::endl;
        std::cout << "         - Oscillation penalty: " << episode_oscillation_penalty_ << std::endl;
        std::cout << "         - State fault penalty: " << episode_state_fault_penalty_ << std::endl;
        std::cout << "         + Hold bonus:          " << episode_hold_bonus_ << std::endl;
        std::cout << "         = Total return:        " << episode_reward_sum_ << std::endl;
        
        disarm_and_stop();
        episode_running_ = false;
        ready_for_rl_ = false;
    }

    result.observation = obs;
    result.reward = reward;
    result.terminated = terminated;
    result.truncated = truncated;
    result.lateral_position = y_m;
    result.x_position = x_m;
    result.z_position = z_m;
    result.distance_to_target = distance_to_target;
    result.horiz_speed = horiz_speed;
    result.a_lat = a_lat;

    return result;
}

void PX4AccelEnv::close() {
    if (node_ != nullptr) {
        try {
            disarm_and_stop();
        } catch (...) {}
        node_.reset();
        rclcpp::shutdown();
    }
}
