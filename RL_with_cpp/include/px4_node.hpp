#ifndef PX4_NODE_HPP
#define PX4_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_angular_velocity.hpp>
#include <memory>
#include <functional>
#include <chrono>
#include <array>

class PX4Node : public rclcpp::Node {
public:
    PX4Node(double hz);
    ~PX4Node() = default;

    // Publishing methods
    void send_vehicle_cmd(uint16_t command, float p1 = 0.0f, float p2 = 0.0f, 
                          float p3 = 0.0f, float p4 = 0.0f, float p5 = 0.0f, 
                          float p6 = 0.0f, float p7 = 0.0f);
    
    void publish_offboard_heartbeat(bool position = false, bool velocity = false,
                                    bool acceleration = false, bool attitude = false,
                                    bool body_rate = false, bool thrust_and_torque = false,
                                    bool direct_actuator = false);
    
    void publish_accel_setpoint(float ax, float ay, float az,
                                float yaw_rad = 0.0f, float yaw_rate_rad_s = 0.0f);
    void publish_accel_setpoint_lat(float a_lat, float a_ned_z);
    void publish_attitude_setpoint(const std::array<float, 4>& quat_frn,
                                   float yaw_rate_rad_s,
                                   const std::array<float, 3>& thrust_body);

    // Publish a position setpoint in NED (meters), yaw in radians
    void publish_position_setpoint(float x_ned, float y_ned, float z_ned, float yaw_rad = 0.0f);
    
    void sleep_dt();
    
    bool wait_for_local(double timeout = 2.0);
    bool wait_until(std::function<bool()> pred, double timeout = 3.0);

    // State accessors
    bool is_armed() const { return armed_; }
    uint8_t get_nav_state() const { return nav_state_; }
    std::shared_ptr<px4_msgs::msg::VehicleLocalPosition> get_last_local() const { 
        return last_local_; 
    }
    double get_last_local_time() const { return last_local_time_; }
    std::shared_ptr<px4_msgs::msg::VehicleAttitude> get_last_attitude() const { 
        return last_attitude_; 
    }
    std::shared_ptr<px4_msgs::msg::VehicleAngularVelocity> get_last_angular_velocity() const {
        return last_ang_vel_;
    }
    double get_last_attitude_time() const { return last_attitude_time_; }
    double get_last_ang_vel_time() const { return last_ang_vel_time_; }
    double get_rate_hz() const { return rate_hz_; }
    double get_dt() const { return dt_; }

private:
    void on_status(const px4_msgs::msg::VehicleStatus::SharedPtr msg);
    void on_local(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void on_attitude(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void on_angular_velocity(const px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg);

    double rate_hz_;
    double dt_;

    // Publishers
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr pub_offboard_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_traj_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr pub_attitude_sp_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr pub_cmd_;

    // Subscribers
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr sub_status_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr sub_local_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr sub_attitude_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAngularVelocity>::SharedPtr sub_ang_vel_;

    // State
    bool armed_;
    uint8_t nav_state_;
    std::shared_ptr<px4_msgs::msg::VehicleLocalPosition> last_local_;
    std::shared_ptr<px4_msgs::msg::VehicleAttitude> last_attitude_;
    std::shared_ptr<px4_msgs::msg::VehicleAngularVelocity> last_ang_vel_;
    double last_local_time_;
    double last_attitude_time_;
    double last_ang_vel_time_;
};

#endif // PX4_NODE_HPP
