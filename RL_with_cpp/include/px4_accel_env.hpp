#ifndef PX4_ACCEL_ENV_HPP
#define PX4_ACCEL_ENV_HPP

#include "px4_node.hpp"
#include <torch/torch.h>
#include <array>
#include <vector>
#include <deque>
#include <memory>

struct StepResult {
    torch::Tensor observation;  // [6]: [x, y, z, vx, vy, vz]
    double reward;
    bool terminated;
    bool truncated;
    
    // Info fields
    double lateral_position;
    double x_position;
    double z_position;
    double distance_to_target;
    double horiz_speed;
    double a_lat;
};

class PX4AccelEnv {
public:
    PX4AccelEnv(double rate_hz = 50.0,
                double a_max = 3.0,
                double ep_time_s = 6.0,
                double target_up_m = 2.0,
                double target_lateral_m = 2.0,
                double start_up_m = 2.5,
                double takeoff_kp = 2.5,
                double takeoff_kd = 1.2,
                double takeoff_tol = 0.05,
                int settle_steps = 20,
                double takeoff_max_time = 8.0,
                bool safety_verbose = true);

    ~PX4AccelEnv();

    // Gym-like API
    torch::Tensor reset();
    StepResult step(const torch::Tensor& action);
    void close();

    // Accessors
    int get_max_steps() const { return max_steps_; }
    double get_episode_reward_sum() const { return episode_reward_sum_; }

private:
    // Helper methods
    void ensure_node();
    void disarm_and_stop();
    torch::Tensor read_observation();
    bool arm_offboard();
    void reset_reward_tracking();
    std::string pose_to_str(const std::array<float, 3>& ned) const;
    bool wait_until_grounded(double timeout_s = 10.0, double vz_tol = 0.15, double stable_s = 1.0);

    // Configuration
    double rate_hz_;
    double dt_;
    double a_max_;
    int max_steps_;
    double target_up_;
    double target_lateral_;
    double start_up_m_;
    double takeoff_kp_;
    double takeoff_kd_;
    double takeoff_tol_;
    int settle_steps_;
    double takeoff_max_time_;
    bool safety_verbose_;

    // Reward weights
    double w_terminal_;
    double w_improve_;
    double w_height_;
    double w_xdrift_;
    double w_overshoot_;
    double w_action_;
    double w_vel_limit_;
    double w_stagnation_;
    double w_oscillation_;
    double w_state_fault_;
    double w_backward_;
    double w_action_slew_;
    double w_hold_bonus_;

    // Safety thresholds
    double max_abs_vxy_;
    double soft_abs_vxy_;
    double max_abs_vz_;
    double max_abs_x_;
    double max_abs_z_;
    double max_abs_y_;
    double stale_timeout_s_;
    int no_progress_horizon_;
    double min_improvement_eps_;
    int oscillation_window_;
    int oscillation_flips_;
    double max_action_slew_rate_;

    // PD gains for altitude hold
    double hold_kp_;
    double hold_kd_;

    // State
    std::shared_ptr<PX4Node> node_;
    std::array<float, 3> spawn_xyz_;
    bool spawn_captured_ = false;
    std::array<float, 3> episode_origin_ned_;
    std::array<float, 3> current_global_ned_;
    std::array<float, 3> current_local_ned_;
    float hover_z_ned_;
    
    int step_;
    bool episode_running_;
    bool ready_for_rl_;
    
    // Tracking
    double episode_reward_sum_;
    double episode_improvement_reward_;
    double episode_overshoot_penalty_;
    double episode_action_penalty_;
    double episode_height_penalty_;
    double episode_x_drift_penalty_;
    double episode_backward_penalty_;
    double episode_action_slew_penalty_;
    double episode_soft_vel_penalty_;
    double episode_stagnation_penalty_;
    double episode_oscillation_penalty_;
    double episode_state_fault_penalty_;
    double episode_hold_bonus_;
    
    double last_distance_to_target_;
    double last_lateral_position_;
    double last_vy_local_;
    int no_progress_steps_;
    std::deque<double> vy_sign_history_;
    double prev_action_;
    
    // Frequency monitor
    double freq_monitor_start_time_;
    int freq_monitor_step_count_;
    double last_step_wall_time_;
};

#endif // PX4_ACCEL_ENV_HPP
