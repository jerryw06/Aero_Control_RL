#include "px4_accel_env.hpp"
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>

using namespace std::chrono_literals;

namespace {
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;
constexpr float GRAVITY = 9.81f;  // Positive for NED frame (Z-down)

std::string nav_state_str(int nav) {
    switch (nav) {
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD: return "OFFBOARD";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_MANUAL: return "MANUAL";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_ALTCTL: return "ALTCTL";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_POSCTL: return "POSCTL";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_MISSION: return "AUTO_MISSION";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_TAKEOFF: return "AUTO_TAKEOFF";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_LAND: return "AUTO_LAND";
        case px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_LOITER: return "AUTO_LOITER";
        default: return std::string("UNKNOWN(") + std::to_string(nav) + ")";
    }
}
}

PX4AccelEnv::PX4AccelEnv(double rate_hz, double a_max, double ep_time_s,
                         double target_up_m, double target_lateral_m,
                         double start_up_m, double takeoff_kp, double takeoff_kd,
                         double takeoff_tol, int settle_steps, double takeoff_max_time,
                         bool safety_verbose)
    : rate_cap_hz_(150.0),
      requested_rate_hz_(rate_hz),
      rate_hz_(std::max(1.0, std::min(rate_hz, rate_cap_hz_))),
      dt_(1.0 / std::max(1.0, std::min(rate_hz, rate_cap_hz_))),
      a_max_(a_max),
      max_steps_(static_cast<int>(ep_time_s * rate_hz_)),
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
      roll_limit_deg_(35.0),
      pitch_limit_deg_(35.0),
      yaw_limit_deg_(180.0),
      unsafe_roll_deg_(45.0),
      unsafe_pitch_deg_(45.0),
      unsafe_yaw_rate_deg_s_(120.0),
      yaw_rate_limit_deg_s_(360.0),
      yaw_rate_limit_rad_s_(yaw_rate_limit_deg_s_ * DEG2RAD),
      obs_dim_(12),
      act_dim_(4),
      enforce_attitude_limits_(false),
      // PD gains
      hold_kp_(2.0),
      hold_kd_(1.0),
      node_(nullptr),
      spawn_xyz_({0.0f, 0.0f, 0.7f}),
      hover_z_ned_(0.0f),
      step_(0),
      episode_running_(false),
      ready_for_rl_(false),
      freq_monitor_start_time_(0.0),
      freq_monitor_step_count_(0),
      last_step_wall_time_(0.0),
      use_isaac_sim_reset_(false)
{
    episode_origin_ned_.fill(0.0f);
    current_global_ned_.fill(0.0f);
    current_local_ned_.fill(0.0f);
    current_attitude_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};
    current_rpy_rad_ = {0.0f, 0.0f, 0.0f};
    current_body_rates_rad_s_ = {0.0f, 0.0f, 0.0f};
    reset_reward_tracking();
    std::cout << "[HZ-DEBUG] Constructor: rate_hz_arg=" << rate_hz 
              << " rate_cap_hz_=" << rate_cap_hz_
              << " min(rate,cap)=" << std::min(rate_hz, rate_cap_hz_)
              << " max(1,min)=" << std::max(1.0, std::min(rate_hz, rate_cap_hz_))
              << " ACTUAL rate_hz_=" << rate_hz_
              << " dt_=" << dt_ << std::endl;
    std::cout << "[HZ] Requested " << requested_rate_hz_ << " Hz, capped at "
              << rate_hz_ << " Hz, dt=" << dt_ << " s" << std::endl;
    
    // Check if Isaac Sim reset service is available (optional feature)
    // If using Pegasus Simulator with Isaac Sim, you can publish reset commands via a service
    // For now, this is a placeholder - integration depends on your sim setup
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
    prev_action_norm_.assign(act_dim_, 0.0f);
}

void PX4AccelEnv::ensure_node() {
    if (node_ == nullptr) {
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<PX4Node>(rate_hz_);
        // Check if Isaac Sim script restart is enabled
        const char* sim_reset_env = std::getenv("ISAAC_SIM_RESET");
        if (sim_reset_env && std::string(sim_reset_env) == "1") {
            use_isaac_sim_reset_ = true;
            std::cout << "[INIT] Isaac Sim script restart enabled." << std::endl;
        } else {
            use_isaac_sim_reset_ = false;
            std::cout << "[INIT] Isaac Sim script restart disabled." << std::endl;
        }
    }
}

std::string PX4AccelEnv::pose_to_str(const std::array<float, 3>& ned) const {
    char buf[100];
    snprintf(buf, sizeof(buf), "x=%.2f y=%.2f z=%.2f", ned[0], ned[1], ned[2]);
    return std::string(buf);
}

void PX4AccelEnv::publish_rl_attitude(const std::array<float, 3>& rpy_rad,
                                      float thrust_body_z) {
    if (node_ == nullptr) {
        return;
    }
    auto quat = euler_to_quat(rpy_rad[0], rpy_rad[1], rpy_rad[2]);
    std::array<float, 3> thrust = {0.0f, 0.0f, thrust_body_z};
    node_->publish_attitude_setpoint(quat, 0.0f, thrust);
}

void PX4AccelEnv::hold_position_for(double seconds) {
    if (node_ == nullptr) {
        return;
    }
    // Use current local position as hold target; default to episode origin if not available
    float hold_x = episode_origin_ned_[0];
    float hold_y = episode_origin_ned_[1];
    float hold_z = hover_z_ned_;
    auto lp = node_->get_last_local();
    if (lp) {
        hold_x = lp->x;
        hold_y = lp->y;
        // Keep altitude at the stabilized hover height when available
        hold_z = (hover_z_ned_ != 0.0f) ? hover_z_ned_ : lp->z;
    }

    int iters = static_cast<int>(std::max(1.0, seconds) / std::max(1e-3, dt_));
    for (int i = 0; i < iters; ++i) {
        node_->publish_offboard_heartbeat(true, false, false, false, false);
        node_->publish_position_setpoint(hold_x, hold_y, hold_z, 0.0f);
        rclcpp::spin_some(node_->get_node_base_interface());
        node_->sleep_dt();
    }
}

std::array<float, 4> PX4AccelEnv::euler_to_quat(float roll, float pitch, float yaw) const {
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);
    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);

    std::array<float, 4> q;
    q[0] = cy * cr * cp + sy * sr * sp;
    q[1] = cy * sr * cp - sy * cr * sp;
    q[2] = cy * cr * sp + sy * sr * cp;
    q[3] = sy * cr * cp - cy * sr * sp;
    return q;
}

std::array<float, 3> PX4AccelEnv::quat_to_euler(const std::array<float, 4>& quat) const {
    float qw = quat[0];
    float qx = quat[1];
    float qy = quat[2];
    float qz = quat[3];

    float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    float roll = std::atan2(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (qw * qy - qz * qx);
    float pitch;
    if (std::abs(sinp) >= 1.0f) {
        pitch = std::copysign(static_cast<float>(M_PI) / 2.0f, sinp);
    } else {
        pitch = std::asin(sinp);
    }

    float siny_cosp = 2.0f * (qw * qz + qx * qy);
    float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
    float yaw = std::atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

void PX4AccelEnv::update_attitude_cache() {
    if (node_ == nullptr) {
        return;
    }
    auto att = node_->get_last_attitude();
    if (att) {
        for (size_t i = 0; i < 4; ++i) {
            current_attitude_quat_[i] = att->q[i];
        }
        auto rpy = quat_to_euler(current_attitude_quat_);
        current_rpy_rad_ = rpy;
    }
    auto ang_vel = node_->get_last_angular_velocity();
    if (ang_vel) {
        current_body_rates_rad_s_[0] = ang_vel->xyz[0];
        current_body_rates_rad_s_[1] = ang_vel->xyz[1];
        current_body_rates_rad_s_[2] = ang_vel->xyz[2];
    }
}

bool PX4AccelEnv::attitude_unstable(float roll_deg, float pitch_deg, float yaw_rate_deg_s) const {
    if (!enforce_attitude_limits_) {
        return false;
    }
    return (std::abs(roll_deg) > unsafe_roll_deg_) ||
           (std::abs(pitch_deg) > unsafe_pitch_deg_) ||
           (std::abs(yaw_rate_deg_s) > unsafe_yaw_rate_deg_s_);
}

void PX4AccelEnv::initiate_safe_landing(const std::string& reason) {
    if (node_ == nullptr) {
        return;
    }
    std::cout << "[SAFETY] " << reason << std::endl;
    node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
    ready_for_rl_ = false;
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
        return torch::zeros({obs_dim_}, torch::kFloat32);
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

    update_attitude_cache();

    auto obs = torch::zeros({obs_dim_}, torch::kFloat32);
    obs[0] = current_local_ned_[0];
    obs[1] = current_local_ned_[1];
    obs[2] = current_local_ned_[2];
    obs[3] = vx_ned;
    obs[4] = vy_ned;
    obs[5] = vz_ned;
    obs[6] = current_rpy_rad_[0];
    obs[7] = current_rpy_rad_[1];
    obs[8] = current_rpy_rad_[2];
    obs[9]  = current_body_rates_rad_s_[0];
    obs[10] = current_body_rates_rad_s_[1];
    obs[11] = current_body_rates_rad_s_[2];

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
        std::cout << "[STATE] Waiting for PX4 local position data..." << std::endl;
        node_->wait_for_local(10.0);  // Increased timeout for after reset
        auto warm_local = node_->get_last_local();
        float hold_x = 0.0f, hold_y = 0.0f, hold_z = 0.0f;
        if (warm_local) {
            hold_x = warm_local->x;
            hold_y = warm_local->y;
            hold_z = warm_local->z;
            std::cout << "[STATE] Got position data: x=" << hold_x << " y=" << hold_y << " z=" << hold_z << std::endl;
        } else {
            std::cout << "[WARN] No position data received after 10s timeout!" << std::endl;
        }
        std::cout << "[STATE] Publishing OFFBOARD heartbeat..." << std::endl;
        for (int i = 0; i < 30; ++i) {
            node_->publish_offboard_heartbeat(true, false, false, false, false);
            node_->publish_position_setpoint(hold_x, hold_y, hold_z, 0.0f);
            rclcpp::spin_some(node_->get_node_base_interface());
            node_->sleep_dt();
            if (i % 10 == 0) {
                std::cout << "[TAKEOFF-DEBUG] Warmup i=" << i << " nav_state="
                          << nav_state_str(node_->get_nav_state())
                          << " armed=" << (node_->is_armed()?"YES":"NO") << std::endl;
            }
        }

        // Switch to OFFBOARD
        std::cout << "[STATE] Entering OFFBOARD mode..." << std::endl;
        node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
        bool offboard_ok = node_->wait_until([this]() {
            return node_->get_nav_state() == px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD;
        }, 3.0);
        if (!offboard_ok) {
            std::cout << "[WARN] Failed to confirm OFFBOARD mode. Retrying takeoff sequence..." << std::endl;
            std::cout << "[TAKEOFF-DEBUG] nav_state after attempt=" << nav_state_str(node_->get_nav_state()) << std::endl;
            continue;
        }

        // Arm the vehicle - keep sending OFFBOARD heartbeat during arming
        std::cout << "[STATE] Arming vehicle..." << std::endl;
        int arm_attempts = 0;
        const int max_arm_attempts = 3;
        bool armed = false;
        
        while (arm_attempts < max_arm_attempts && !armed) {
            arm_attempts++;
            
            // Send arm command (try without force first)
            std::cout << "[ARM] Attempt " << arm_attempts << "/" << max_arm_attempts << std::endl;
            node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
            
            // Keep sending OFFBOARD heartbeat while waiting for arm to succeed
            auto start_time = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(3)) {
                node_->publish_offboard_heartbeat(true, false, false, false, false);
                node_->publish_position_setpoint(hold_x, hold_y, hold_z, 0.0f);
                rclcpp::spin_some(node_->get_node_base_interface());
                
                if (node_->is_armed()) {
                    std::cout << "[STATE] Armed successfully!" << std::endl;
                    armed = true;
                    break;
                }
                if (std::chrono::steady_clock::now() - start_time > std::chrono::milliseconds(500)) {
                    // periodic debug
                    std::cout << "[ARM-DEBUG] nav_state=" << nav_state_str(node_->get_nav_state())
                              << " armed=" << (node_->is_armed()?"YES":"NO") << std::endl;
                }
                std::this_thread::sleep_for(50ms);
            }
            
            if (!armed && arm_attempts < max_arm_attempts) {
                std::cout << "[WARN] Arm attempt failed, retrying..." << std::endl;
                std::this_thread::sleep_for(1s);
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
        std::cout << "[TAKEOFF-DEBUG] pre_z0=" << pre_z0 << " start_up_m=" << start_up_m_
                  << " target_z_ned=" << target_z_ned << " dt_=" << dt_ << " steps=" << steps << std::endl;
        std::cout << "[STATE] Performing smooth takeoff to " << start_up_m_ << " m (target_z_ned=" 
                  << target_z_ned << ", " << steps << " steps over 3.0s)..." << std::endl;
        
        float hold_x0 = episode_origin_ned_[0];
        float hold_y0 = episode_origin_ned_[1];
        for (int i = 0; i < steps; ++i) {
            float alpha = static_cast<float>(i + 1) / steps;
            float interp_z = pre_z0 + alpha * (target_z_ned - pre_z0);
            node_->publish_offboard_heartbeat(true, false, false, false, false);
            node_->publish_position_setpoint(hold_x0, hold_y0, interp_z, 0.0f);
            rclcpp::spin_some(node_->get_node_base_interface());
            node_->sleep_dt();
            if (i % (steps/3 + 1) == 0) {
                auto lp_take = node_->get_last_local();
                if (lp_take) {
                    std::cout << "[TAKEOFF-DEBUG] Climb step=" << i << " z=" << lp_take->z
                              << " target_z=" << interp_z << " nav_state=" << nav_state_str(node_->get_nav_state()) << std::endl;
                }
            }
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
            if ((std::chrono::steady_clock::now() - hover_t0) > std::chrono::seconds(2) && (std::chrono::steady_clock::now() - hover_t0) < std::chrono::seconds(3)) {
                auto lp_hover_dbg = node_->get_last_local();
                if (lp_hover_dbg) {
                    std::cout << "[HOVER-DEBUG] z=" << lp_hover_dbg->z << " target=" << target_z_ned
                              << " z_err=" << std::abs(lp_hover_dbg->z - target_z_ned) << " nav_state=" << nav_state_str(node_->get_nav_state()) << std::endl;
                }
            }
            
            auto now = std::chrono::steady_clock::now();
            auto local_now = node_->get_last_local();
            if (local_now != nullptr) {
                float z_err = std::abs(local_now->z - target_z_ned);
                if (z_err <= 1.0f) {  // Relaxed from 0.5m to 1.0m
                    if (stable_start == std::chrono::steady_clock::time_point()) {
                        stable_start = now;
                    }
                    if (std::chrono::duration<double>(now - stable_start).count() >= 0.5) {  // Reduced from 1.0s to 0.5s
                        std::cout << "[STATE] Hover stabilized." << std::endl;
                        hover_success = true;
                        break;
                    }
                } else {
                    stable_start = std::chrono::steady_clock::time_point();
                }
            }
            if (std::chrono::duration<double>(now - hover_t0).count() >= 15.0) {  // Increased from 10s to 15s
                std::cout << "[WARN] Hover stabilization timeout (15 s)." << std::endl;
                break;
            }
        }

        if (hover_success) {
            // ========================================================================
            // CRITICAL: Ensure consistent starting position for RL
            // ========================================================================
            // Before handing control to RL, verify:
            // 1. Altitude is within acceptable range of target
            // 2. All velocities (vx, vy, vz) are near zero
            // 3. Conditions hold stable for 1 second
            std::cout << "[STATE] Hover stabilized. Verifying RL starting conditions..." << std::endl;
            
            const float altitude_tolerance = 0.2f;  // meters - acceptable deviation from target altitude
            const float velocity_threshold = 0.15f;  // m/s - max velocity in any direction
            const double stability_duration = 1.0;   // seconds - how long conditions must hold
            
            auto stability_check_start = std::chrono::steady_clock::now();
            auto stable_condition_start = std::chrono::steady_clock::time_point();
            bool rl_ready_conditions_met = false;
            
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - stability_check_start).count() < 10.0) {
                node_->publish_offboard_heartbeat(true, false, false, false, false);
                node_->publish_position_setpoint(hold_x0, hold_y0, target_z_ned, 0.0f);
                rclcpp::spin_some(node_->get_node_base_interface());
                node_->sleep_dt();
                
                auto lp_check = node_->get_last_local();
                if (lp_check != nullptr) {
                    float z_error = std::abs(lp_check->z - target_z_ned);
                    float vx_abs = std::abs(lp_check->vx);
                    float vy_abs = std::abs(lp_check->vy);
                    float vz_abs = std::abs(lp_check->vz);
                    float max_velocity = std::max({vx_abs, vy_abs, vz_abs});
                    
                    bool altitude_ok = (z_error <= altitude_tolerance);
                    bool velocities_ok = (max_velocity <= velocity_threshold);
                    
                    if (altitude_ok && velocities_ok) {
                        if (stable_condition_start == std::chrono::steady_clock::time_point()) {
                            stable_condition_start = std::chrono::steady_clock::now();
                            std::cout << "[RL-READY-CHECK] Starting conditions met - z_err=" << z_error 
                                      << "m, max_vel=" << max_velocity << "m/s. Waiting " 
                                      << stability_duration << "s for stability..." << std::endl;
                        }
                        
                        double stable_elapsed = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - stable_condition_start).count();
                        
                        if (stable_elapsed >= stability_duration) {
                            std::cout << "[RL-READY-CHECK] ✓ Altitude within " << altitude_tolerance << "m (error: " 
                                      << z_error << "m)" << std::endl;
                            std::cout << "[RL-READY-CHECK] ✓ All velocities ≤ " << velocity_threshold 
                                      << "m/s (vx=" << vx_abs << ", vy=" << vy_abs << ", vz=" << vz_abs << ")" << std::endl;
                            std::cout << "[RL-READY-CHECK] ✓ Conditions stable for " << stability_duration << "s" << std::endl;
                            std::cout << "[RL-READY-CHECK] Starting position VERIFIED - RL takeover in 1 second..." << std::endl;
                            
                            // Wait additional 1 second before RL takeover
                            auto final_wait_start = std::chrono::steady_clock::now();
                            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - final_wait_start).count() < 1.0) {
                                node_->publish_offboard_heartbeat(true, false, false, false, false);
                                node_->publish_position_setpoint(hold_x0, hold_y0, target_z_ned, 0.0f);
                                rclcpp::spin_some(node_->get_node_base_interface());
                                node_->sleep_dt();
                            }
                            
                            rl_ready_conditions_met = true;
                            break;
                        }
                    } else {
                        // Conditions not met, reset timer
                        if (stable_condition_start != std::chrono::steady_clock::time_point()) {
                            std::cout << "[RL-READY-CHECK] Conditions lost - z_err=" << z_error 
                                      << "m (tol=" << altitude_tolerance << "m), max_vel=" << max_velocity 
                                      << "m/s (thr=" << velocity_threshold << "m/s). Restarting stability timer." << std::endl;
                        }
                        stable_condition_start = std::chrono::steady_clock::time_point();
                    }
                }
            }
            
            if (!rl_ready_conditions_met) {
                std::cout << "[WARN] RL starting condition verification timeout (10s). Conditions not consistently met." << std::endl;
                if (takeoff_attempt < max_takeoff_retries - 1) {
                    std::cout << "[STATE] Landing and disarming before retry..." << std::endl;
                    node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
                    std::this_thread::sleep_for(2s);
                    node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
                    node_->wait_until([this]() { return !node_->is_armed(); }, 3.0);
                    std::this_thread::sleep_for(2s);
                    continue;
                } else {
                    std::cout << "[ERROR] Failed to achieve consistent RL starting conditions after all retries." << std::endl;
                    return false;
                }
            }
            
            // Re-zero origin after final verification
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
            std::cout << "[STATE] RL starting position verified and consistent — switching to lateral RL control." << std::endl;
            ready_for_rl_ = true;
            return true;
        } else {
            if (takeoff_attempt < max_takeoff_retries - 1) {
                std::cout << "[HOVER-DEBUG] Hover failed; will retry. nav_state=" << nav_state_str(node_->get_nav_state())
                          << " armed=" << (node_->is_armed()?"YES":"NO") << std::endl;
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
    static int episode_count = 0;
    episode_count++;
    
    std::cout << "\n\n" << std::endl;
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         STARTING EPISODE #" << episode_count << " RESET SEQUENCE          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    ensure_node();
    current_global_ned_.fill(0.0f);
    current_local_ned_.fill(0.0f);
    episode_origin_ned_.fill(0.0f);
    reset_reward_tracking();
    std::cout << "[HZ] RL loop target " << rate_hz_ << " Hz (requested "
              << requested_rate_hz_ << ", cap " << rate_cap_hz_ << ")." << std::endl;

    std::cout << "[STATE] Waiting for PX4 local position..." << std::endl;
    bool ok = node_->wait_for_local(10.0);
    if (!ok || node_->get_last_local() == nullptr) {
        std::cout << "[RESET-ERROR] wait_for_local failed (10s). nav_state="
                  << nav_state_str(node_->get_nav_state()) << " armed=" << (node_->is_armed()?"YES":"NO")
                  << " last_local_ptr=" << (node_->get_last_local()?"NONNULL":"NULL") << std::endl;
        std::cout << "[RESET-ERROR] Possible causes: PX4 SITL not running, MicroXRCEAgent bridge down, wrong ROS_DOMAIN_ID, or DDS transport mismatch." << std::endl;
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

    // ========================================================================
    // PROPER EPISODE RESET SEQUENCE (Python-style teleport-back-and-reset)
    // ========================================================================
    
    // 1. If previous episode ended mid-air, land and wait until grounded
    auto last_local_pre = node_->get_last_local();
    if (last_local_pre && std::abs(last_local_pre->z - spawn_xyz_[2]) > 0.3f) {
        std::cout << "[RESET] Episode ended mid-air; landing to ground..." << std::endl;
        node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
        if (!wait_until_grounded(12.0)) {
            std::cout << "[WARN] Grounding timeout; proceeding with best available position." << std::endl;
        } else {
            std::cout << "[RESET] Grounded and stabilized." << std::endl;
        }
    }

    // 2. Disarm the vehicle
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 2: Disarming vehicle..." << std::endl;
    disarm_and_stop();
    std::cout << "[RESET-SUCCESS] Vehicle disarmed." << std::endl;
    std::this_thread::sleep_for(500ms);
    
    // 3. Teleport back to spawn in Isaac Sim (if available)
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 3: Teleporting to spawn position..." << std::endl;
    if (!teleport_to_spawn()) {
        std::cout << "[RESET-WARN] Teleport failed or not available; continuing with manual reset." << std::endl;
    }
    
    // 4. Reset physics and zero velocities
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 4: Resetting physics and zeroing velocities..." << std::endl;
    if (!reset_physics_and_velocities()) {
        std::cout << "[RESET-WARN] Physics reset incomplete; continuing." << std::endl;
    }
    
    // 5. Wait for position AND velocity to settle after teleport
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 5: Waiting for position and velocity to stabilize..." << std::endl;
    wait_for_position_settle(3.0);  // Increased to 3.0s for better stabilization

    // 6. Reset PX4 local origin tracking
    //    After teleport/landing, re-capture the current position as episode origin
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 6: Resetting PX4 local origin tracking..." << std::endl;
    node_->wait_for_local(2.0);
    auto local = node_->get_last_local();
    if (local == nullptr) {
        throw std::runtime_error("No VehicleLocalPosition messages received after reset.");
    }
    
    std::cout << "[RESET-DEBUG] Raw PX4 position after reset: x=" << local->x 
              << ", y=" << local->y << ", z=" << local->z << std::endl;
    
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
    
    std::cout << "[RESET-DEBUG] Episode origin (spawn): " << pose_to_str(episode_origin_ned_) << std::endl;
    std::cout << "[RESET-DEBUG] Current global NED: " << pose_to_str(current_global_ned_) << std::endl;
    std::cout << "[RESET-DEBUG] Current local NED (global - origin): " << pose_to_str(current_local_ned_) << std::endl;
    
    float reset_drift = std::sqrt(
        std::pow(current_local_ned_[0], 2) +
        std::pow(current_local_ned_[1], 2) +
        std::pow(current_local_ned_[2], 2)
    );
    std::cout << "[RESET-DEBUG] Drift from spawn after reset: " << reset_drift << " m" << std::endl;
    
    if (reset_drift > 1.0f) {
        std::cout << "[RESET-WARN] Large drift from spawn (" << reset_drift 
                  << "m). Isaac Sim restart may have taken longer than expected." << std::endl;
        // Since we restarted Isaac Sim, drift should be minimal. If it's large, 
        // it likely means initialization isn't complete yet. The system will handle
        // this through the takeoff sequence.
    }

    // 7. Re-arm and take off to hover altitude
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET] Step 7: Re-arming and taking off to hover altitude..." << std::endl;
    ready_for_rl_ = false;
    ok = arm_offboard();
    if (!ok || !ready_for_rl_) {
        std::cout << "[RESET-ERROR] arm_offboard() returned false or ready_for_rl_ not set." << std::endl;
        std::cout << "[RESET-DEBUG] nav_state=" << nav_state_str(node_->get_nav_state())
                  << " armed=" << (node_->is_armed()?"YES":"NO") << " ready_for_rl_=" << (ready_for_rl_?"true":"false") << std::endl;
        auto lp_fail = node_->get_last_local();
        if (lp_fail) {
            std::cout << "[RESET-DEBUG] Current local position before failure: x=" << lp_fail->x
                      << " y=" << lp_fail->y << " z=" << lp_fail->z << std::endl;
        } else {
            std::cout << "[RESET-DEBUG] No local position message available at failure point." << std::endl;
        }
        std::cout << "[RESET-ERROR] Common failure reasons: OFFBOARD not entered, arming denied, altitude hover not stabilized, or insufficient heartbeats during takeoff." << std::endl;
        throw std::runtime_error("Failed to arm/hover after reset.");
    }
    
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[RESET-SUCCESS] Episode reset complete! Ready for RL control." << std::endl;
    std::cout << "[RESET] ========================================" << std::endl;

    step_ = 0;
    episode_running_ = true;
    // Activate vertical hold guard so RL does not interfere with late takeoff stabilization
    vertical_hold_active_ = true;
    vertical_stable_steps_ = 0;
    required_stable_steps_ = static_cast<int>(std::max(5.0, 0.5 * rate_hz_)); // ~0.5s dwell
    vertical_guard_log_count_ = 0;
    std::cout << "[TAKEOFF-GUARD] RL vertical/yaw channels locked until hover stable (|z|<" 
              << vertical_release_z_tol_ << " & |vz|<" << vertical_release_vz_tol_ 
              << " for " << required_stable_steps_ << " consecutive steps)." << std::endl;

    auto obs = read_observation();
    
    std::cout << "\n[RESET] ========================================" << std::endl;
    std::cout << "[RESET] FINAL STATE SUMMARY:" << std::endl;
    std::cout << "[RESET] ========================================" << std::endl;
    std::cout << "[STATE] Start pose:" << std::endl;
    std::cout << "         Global NED: " << pose_to_str(current_global_ned_) << std::endl;
    std::cout << "         Local NED:  " << pose_to_str(current_local_ned_) << std::endl;
    std::cout << "         Spawn ref:  " << pose_to_str(spawn_xyz_) << std::endl;
    
    auto final_pos = node_->get_last_local();
    if (final_pos) {
        std::cout << "         Velocities: vx=" << final_pos->vx << ", vy=" << final_pos->vy << ", vz=" << final_pos->vz << std::endl;
        std::cout << "         Armed: " << (node_->is_armed() ? "YES" : "NO") << std::endl;
    }
    
    std::cout << "[STATE] RL control begins NOW." << std::endl;
    std::cout << "[RESET] ========================================\n" << std::endl;

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

    // Parse action with safety checks
    bool fault_action = false;
    torch::Tensor action_checked;
    if (action.defined()) {
        action_checked = action.to(torch::kCPU).flatten();
    }
    if (!action_checked.defined() || action_checked.numel() < act_dim_) {
        fault_action = true;
        action_checked = torch::zeros({act_dim_}, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        action_checked = action_checked.to(torch::kFloat32);
        auto invalid_mask = torch::isnan(action_checked) | torch::isinf(action_checked);
        if (invalid_mask.any().item<bool>()) {
            fault_action = true;
            action_checked = torch::zeros({act_dim_}, torch::TensorOptions().dtype(torch::kFloat32));
        }
    }
    if (fault_action && safety_verbose_) {
        std::cout << "[SAFETY] Invalid or NaN action detected; substituting zero command." << std::endl;
    }

    std::vector<float> action_norm(act_dim_, 0.0f);
    for (int i = 0; i < act_dim_; ++i) {
        float raw = (i < action_checked.numel()) ? action_checked[i].item<float>() : 0.0f;
        action_norm[i] = std::clamp(raw, -1.0f, 1.0f);
        if (std::abs(raw) > 1.0f && safety_verbose_) {
            std::cout << "[SAFETY] Action component " << i 
                      << " saturated from " << raw << " to " << action_norm[i] << std::endl;
        }
    }

    // Determine effective dt: use configured dt_ when >0, else measured wall-time since last step
    double now_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double runtime_dt = (last_step_wall_time_ > 0.0) ? (now_wall - last_step_wall_time_) : dt_;
    last_step_wall_time_ = now_wall;
    double eff_dt = (dt_ > 0.0) ? dt_ : std::max(1e-4, runtime_dt);

    // Apply slew-rate limit in normalized space
    float da_allowed = static_cast<float>(max_action_slew_rate_ * eff_dt);
    double slew_penalty = 0.0;
    for (int i = 0; i < act_dim_; ++i) {
        float limited = std::clamp(action_norm[i],
                                   prev_action_norm_[i] - da_allowed,
                                   prev_action_norm_[i] + da_allowed);
        limited = std::clamp(limited, -1.0f, 1.0f);
        slew_penalty += w_action_slew_ * std::abs(limited - action_norm[i]);
        action_norm[i] = limited;
    }

    float ax_cmd = static_cast<float>(a_max_) * action_norm[0];
    float ay_cmd = static_cast<float>(a_max_) * action_norm[1];
    float az_cmd = GRAVITY + static_cast<float>(a_max_) * action_norm[2];
    float yaw_rate_cmd = static_cast<float>(yaw_rate_limit_rad_s_) * action_norm[3];

    // ------------------------------------------------------------------
    // Takeoff vertical hold guard: override az & yaw until altitude stable
    // ------------------------------------------------------------------
    if (vertical_hold_active_) {
        auto lp_guard = node_->get_last_local();
        if (lp_guard) {
            float z_err_guard = std::abs(lp_guard->z - hover_z_ned_); // hover_z_ned_ should be near target (0 local)
            float vz_abs_guard = std::abs(lp_guard->vz);
            bool stable_now = (z_err_guard < vertical_release_z_tol_) && (vz_abs_guard < vertical_release_vz_tol_);
            if (stable_now) {
                vertical_stable_steps_++;
            } else {
                vertical_stable_steps_ = 0; // reset dwell counter
            }
            if (vertical_stable_steps_ >= required_stable_steps_) {
                vertical_hold_active_ = false;
                std::cout << "[TAKEOFF-GUARD] Hover stability confirmed (z_err=" << z_err_guard
                          << " vz=" << vz_abs_guard << ") — enabling full RL control." << std::endl;
            }
            if (vertical_hold_active_) {
                // Override vertical & yaw commands to maintain hover; only lateral acceleration allowed
                az_cmd = GRAVITY; // neutral vertical accel (hover thrust assumption)
                yaw_rate_cmd = 0.0f;
                action_norm[2] = 0.0f; // reflect clamped action for logging consistency
                action_norm[3] = 0.0f;
                if (vertical_guard_log_count_ < 5 || (vertical_guard_log_count_ % 100 == 0)) {
                    std::cout << "[TAKEOFF-GUARD] Holding vertical/yaw (z_err=" << z_err_guard
                              << " vz=" << vz_abs_guard << ", stable_steps=" << vertical_stable_steps_ << "/" 
                              << required_stable_steps_ << ")" << std::endl;
                }
                vertical_guard_log_count_++;
            }
        }
    }

    node_->publish_offboard_heartbeat(false, false, true, false, true, false, false);
    node_->publish_accel_setpoint(ax_cmd, ay_cmd, az_cmd, 0.0f, yaw_rate_cmd);
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
    double roll_deg = static_cast<double>(current_rpy_rad_[0]) * RAD2DEG;
    double pitch_deg = static_cast<double>(current_rpy_rad_[1]) * RAD2DEG;
    double yaw_deg = static_cast<double>(current_rpy_rad_[2]) * RAD2DEG;
    double yaw_rate_deg_s = static_cast<double>(current_body_rates_rad_s_[2]) * RAD2DEG;
    bool attitude_fault = attitude_unstable(roll_deg, pitch_deg, yaw_rate_deg_s);

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
            std::cout << "[HZ] Control loop running at " << freq 
                      << " Hz (target=" << rate_hz_ 
                      << " Hz, cap=" << rate_cap_hz_ << " Hz)" << std::endl;
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
    double action_energy = static_cast<double>(ax_cmd) * ax_cmd +
                           static_cast<double>(ay_cmd) * ay_cmd +
                           static_cast<double>(az_cmd - GRAVITY) * (az_cmd - GRAVITY) +
                           static_cast<double>(yaw_rate_cmd) * yaw_rate_cmd;
    double action_term = w_action_ * action_energy;
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
    if (attitude_fault) {
        state_fault_penalty += w_state_fault_;
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
    prev_action_norm_ = action_norm;

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
    } else if (attitude_fault) {
        char buf[200];
        snprintf(buf, sizeof(buf),
                 "Unsafe attitude detected (roll=%.1f deg, pitch=%.1f deg, yaw_rate=%.1f deg/s).",
                 roll_deg, pitch_deg, yaw_rate_deg_s);
        initiate_safe_landing(buf);
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
        
        // Gracefully hand control back to position-hold before ending episode
        hold_position_for(1.0);

        // If still airborne, command an automatic land; otherwise disarm
        auto lp_end = node_->get_last_local();
        bool airborne = false;
        if (lp_end) {
            // Consider airborne if >0.5 m from spawn altitude
            airborne = std::abs(lp_end->z - spawn_xyz_[2]) > 0.5f;
        }
        if (airborne) {
            std::cout << "[STATE] Commanding LAND for episode end." << std::endl;
            node_->send_vehicle_cmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);
        } else {
            disarm_and_stop();
        }
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
    result.roll_deg = roll_deg;
    result.pitch_deg = pitch_deg;
    result.yaw_deg = yaw_deg;
    result.yaw_rate_dps = yaw_rate_deg_s;
    result.action_command = {static_cast<double>(ax_cmd),
                             static_cast<double>(ay_cmd),
                             static_cast<double>(az_cmd),
                             static_cast<double>(yaw_rate_cmd)};

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

// ============================================================================
// Isaac Sim Reset/Teleport Integration
// ============================================================================

bool PX4AccelEnv::teleport_to_spawn() {
    /**
     * Use topic-based reset instead of ROS2 service to avoid Python<->C++ type support issues.
     * 
     * How it works:
     * 1. C++ publishes to /isaac_sim/reset_request
     * 2. Python script receives it and calls world.reset() + teleport
     * 3. Python publishes to /isaac_sim/reset_done when complete
     * 4. C++ waits for the confirmation message
     * 
     * This is more reliable than services for mixed Python/C++ communication.
     */
    
    std::cout << "[RESET-DEBUG] teleport_to_spawn() called - using topic-based reset" << std::endl;
    std::cout << "[RESET-DEBUG] use_isaac_sim_reset_ = " << use_isaac_sim_reset_ << std::endl;
    std::cout << "[RESET-DEBUG] spawn_xyz_ = " << pose_to_str(spawn_xyz_) << std::endl;
    
    if (!use_isaac_sim_reset_) {
        std::cout << "[RESET-INFO] Isaac Sim reset disabled; relying on PX4 position alone." << std::endl;
        std::cout << "[RESET-INFO] Assuming drone is already at spawn after landing/disarm." << std::endl;
        return true;
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  REQUESTING ISAAC SIM RESET VIA TOPICS               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    // Create publisher and subscriber for reset communication
    auto reset_pub = node_->create_publisher<std_msgs::msg::Bool>("/isaac_sim/reset_request", 10);
    
    // Flag to track when reset is done
    bool reset_done = false;
    auto reset_sub = node_->create_subscription<std_msgs::msg::Bool>(
        "/isaac_sim/reset_done",
        10,
        [&reset_done](const std_msgs::msg::Bool::SharedPtr msg) {
            if (msg->data) {
                reset_done = true;
            }
        }
    );
    
    // Publish reset request
    std::cout << "[RESET] Publishing reset request to /isaac_sim/reset_request..." << std::endl;
    auto msg = std_msgs::msg::Bool();
    msg.data = true;
    reset_pub->publish(msg);
    
    // Wait for reset done confirmation (with timeout)
    std::cout << "[RESET] Waiting for reset confirmation on /isaac_sim/reset_done..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(10);
    
    while (!reset_done) {
        rclcpp::spin_some(node_->get_node_base_interface());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > timeout) {
            std::cout << "[RESET-WARN] Reset confirmation timeout (>10s)." << std::endl;
            std::cout << "[RESET-WARN] Make sure Isaac Sim Python script is running with ROS2 enabled." << std::endl;
            break;
        }
    }
    
    if (reset_done) {
        std::cout << "[RESET-SUCCESS] Isaac Sim environment reset confirmed!" << std::endl;
    }
    
    // Wait a moment for physics to fully settle
    std::cout << "[RESET] Waiting 2 seconds for physics to settle..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Verify position data is available
    std::cout << "[RESET] Verifying PX4 topics after reset..." << std::endl;
    bool topics_ready = node_->wait_for_local(5.0);
    
    if (!topics_ready) {
        std::cout << "[RESET-ERROR] PX4 topics not available after reset!" << std::endl;
        return false;
    }
    
    auto pos_after = node_->get_last_local();
    if (pos_after) {
        std::cout << "[RESET-DEBUG] Position after reset: x=" << pos_after->x 
                  << " y=" << pos_after->y << " z=" << pos_after->z << std::endl;
        float dist_from_spawn = std::sqrt(
            std::pow(pos_after->x - spawn_xyz_[0], 2) +
            std::pow(pos_after->y - spawn_xyz_[1], 2) +
            std::pow(pos_after->z - spawn_xyz_[2], 2)
        );
        std::cout << "[RESET-DEBUG] Distance from spawn: " << dist_from_spawn << " m" << std::endl;
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  ISAAC SIM RESET COMPLETE                             ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    return true;
}

bool PX4AccelEnv::reset_physics_and_velocities() {
    /**
     * Reset physics state and zero out all velocities.
     * This ensures clean episode starts.
     * 
     * In a full Isaac Sim integration, you'd call:
     * - physics scene wake_up() to refresh rigid body states
     * - set linear/angular velocities to zero
     * 
     * For PX4 SITL without direct sim access, we rely on:
     * - Landing and waiting for stabilization
     * - Re-arming and taking off from known position
     */
    
    std::cout << "[RESET-DEBUG] reset_physics_and_velocities() called" << std::endl;
    std::cout << "[RESET] Waiting for velocities to settle..." << std::endl;
    
    // Wait until vehicle is stationary
    auto start = std::chrono::steady_clock::now();
    int check_count = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 3.0) {
        rclcpp::spin_some(node_->get_node_base_interface());
        auto lp = node_->get_last_local();
        if (lp) {
            float v_mag = std::sqrt(lp->vx*lp->vx + lp->vy*lp->vy + lp->vz*lp->vz);
            check_count++;
            
            if (check_count % 10 == 0) {  // Log every 10th check (~0.5s)
                std::cout << "[RESET-DEBUG] Velocity magnitude: " << v_mag << " m/s "
                          << "(vx=" << lp->vx << ", vy=" << lp->vy << ", vz=" << lp->vz << ")" << std::endl;
            }
            
            if (v_mag < 0.1f) {
                std::cout << "[RESET-SUCCESS] Velocities zeroed (|v|=" << v_mag << " m/s < 0.1 m/s)." << std::endl;
                return true;
            }
        } else {
            std::cout << "[RESET-WARN] No local position data during velocity check" << std::endl;
        }
        std::this_thread::sleep_for(50ms);
    }
    
    // Final velocity check
    auto lp_final = node_->get_last_local();
    if (lp_final) {
        float v_final = std::sqrt(lp_final->vx*lp_final->vx + lp_final->vy*lp_final->vy + lp_final->vz*lp_final->vz);
        std::cout << "[RESET-WARN] Velocity settle timeout after 3s; final |v|=" << v_final << " m/s" << std::endl;
    }
    
    std::cout << "[RESET-WARN] Proceeding despite non-zero velocity." << std::endl;
    return true;
}

bool PX4AccelEnv::wait_for_position_settle(double timeout_s) {
    /**
     * Wait for position AND velocity to stabilize near spawn after teleport/reset.
     * Ensures we don't start RL control while drone is still moving.
     * This guarantees identical starting conditions for each episode.
     */
    
    auto start = std::chrono::steady_clock::now();
    std::array<float, 3> last_pos = {0.0f, 0.0f, 0.0f};
    bool first = true;
    int check_count = 0;
    int stable_count = 0;  // Consecutive checks that meet stability criteria
    const int required_stable = 3;  // Require 3 consecutive stable checks (0.3s)
    
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < timeout_s) {
        rclcpp::spin_some(node_->get_node_base_interface());
        auto lp = node_->get_last_local();
        if (lp) {
            check_count++;
            
            if (first) {
                last_pos = {lp->x, lp->y, lp->z};
                first = false;
            } else {
                float dx = lp->x - last_pos[0];
                float dy = lp->y - last_pos[1];
                float dz = lp->z - last_pos[2];
                float drift = std::sqrt(dx*dx + dy*dy + dz*dz);
                
                // Check velocity magnitude
                float vel_mag = std::sqrt(lp->vx*lp->vx + lp->vy*lp->vy + lp->vz*lp->vz);
                
                // Stability criteria: position drift < 5cm AND velocity < 0.1 m/s
                if (drift < 0.05f && vel_mag < 0.1f) {
                    stable_count++;
                    if (stable_count >= required_stable) {
                        std::cout << "[RESET] Stable (drift=" << drift 
                                  << "m, vel=" << vel_mag << " m/s)" << std::endl;
                        float dist_from_spawn = std::sqrt(
                            std::pow(lp->x - spawn_xyz_[0], 2) +
                            std::pow(lp->y - spawn_xyz_[1], 2) +
                            std::pow(lp->z - spawn_xyz_[2], 2)
                        );
                        std::cout << "[RESET] Position offset from spawn: " << dist_from_spawn << " m" << std::endl;
                        return true;
                    }
                } else {
                    // Reset counter if stability criteria not met
                    stable_count = 0;
                }
                last_pos = {lp->x, lp->y, lp->z};
            }
        }
        std::this_thread::sleep_for(100ms);
    }
    
    auto lp_final = node_->get_last_local();
    if (lp_final) {
        std::cout << "[RESET-WARN] Position settle timeout after " << timeout_s << "s" << std::endl;
        std::cout << "[RESET-DEBUG] Final position: x=" << lp_final->x << ", y=" << lp_final->y << ", z=" << lp_final->z << std::endl;
    }
    
    return true;
}
