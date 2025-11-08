#include "px4_node.hpp"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

PX4Node::PX4Node(double hz)
    : Node("px4_accel_env"),
      rate_hz_(hz),
      dt_(1.0 / hz),
      armed_(false),
      nav_state_(px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_MAX),
      last_local_(nullptr),
      last_attitude_(nullptr),
      last_ang_vel_(nullptr),
      last_local_time_(0.0),
      last_attitude_time_(0.0),
      last_ang_vel_time_(0.0)
{
    // QoS profiles
    auto qos_pub = rclcpp::QoS(1)
        .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        .durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

    auto qos_sub = rclcpp::QoS(10)
        .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
        .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

    // Publishers
    pub_offboard_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "fmu/in/offboard_control_mode", qos_pub);
    pub_traj_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "fmu/in/trajectory_setpoint", qos_pub);
    pub_cmd_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
        "fmu/in/vehicle_command", qos_pub);
    pub_attitude_sp_ = this->create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>(
        "fmu/in/vehicle_attitude_setpoint", qos_pub);

    // Subscribers
    sub_status_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
        "fmu/out/vehicle_status", qos_sub,
        std::bind(&PX4Node::on_status, this, std::placeholders::_1));
    sub_local_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        "fmu/out/vehicle_local_position", qos_sub,
        std::bind(&PX4Node::on_local, this, std::placeholders::_1));
    sub_attitude_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
        "fmu/out/vehicle_attitude", qos_sub,
        std::bind(&PX4Node::on_attitude, this, std::placeholders::_1));
    sub_ang_vel_ = this->create_subscription<px4_msgs::msg::VehicleAngularVelocity>(
        "fmu/out/vehicle_angular_velocity", qos_sub,
        std::bind(&PX4Node::on_angular_velocity, this, std::placeholders::_1));
}

void PX4Node::on_status(const px4_msgs::msg::VehicleStatus::SharedPtr msg) {
    armed_ = (msg->arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED);
    nav_state_ = msg->nav_state;
}

void PX4Node::on_local(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
    last_local_ = msg;
    last_local_time_ = this->now().seconds();
}

void PX4Node::on_attitude(const px4_msgs::msg::VehicleAttitude::SharedPtr msg) {
    last_attitude_ = msg;
    last_attitude_time_ = this->now().seconds();
}

void PX4Node::on_angular_velocity(const px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg) {
    last_ang_vel_ = msg;
    last_ang_vel_time_ = this->now().seconds();
}

void PX4Node::send_vehicle_cmd(uint16_t command, float p1, float p2, float p3, 
                                float p4, float p5, float p6, float p7) {
    auto msg = px4_msgs::msg::VehicleCommand();
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    msg.command = command;
    msg.param1 = p1;
    msg.param2 = p2;
    msg.param3 = p3;
    msg.param4 = p4;
    msg.param5 = p5;
    msg.param6 = p6;
    msg.param7 = p7;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    pub_cmd_->publish(msg);
}

void PX4Node::publish_offboard_heartbeat(bool position, bool velocity,
                                         bool acceleration, bool attitude,
                                         bool body_rate, bool thrust_and_torque,
                                         bool direct_actuator) {
    auto hb = px4_msgs::msg::OffboardControlMode();
    hb.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    hb.position = position;
    hb.velocity = velocity;
    hb.acceleration = acceleration;
    hb.attitude = attitude;
    hb.body_rate = body_rate;
    hb.thrust_and_torque = thrust_and_torque;
    hb.direct_actuator = direct_actuator;
    pub_offboard_->publish(hb);
}

void PX4Node::publish_accel_setpoint(float ax, float ay, float az,
                                     float yaw_rad, float yaw_rate_rad_s) {
    auto ts = px4_msgs::msg::TrajectorySetpoint();
    ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    ts.acceleration[0] = ax;
    ts.acceleration[1] = ay;
    ts.acceleration[2] = az;
    ts.yaw = yaw_rad;
    ts.yawspeed = yaw_rate_rad_s;
    pub_traj_->publish(ts);
}

void PX4Node::publish_accel_setpoint_lat(float a_lat, float a_ned_z) {
    publish_accel_setpoint(0.0f, a_lat, a_ned_z);
}

void PX4Node::publish_attitude_setpoint(const std::array<float, 4>& quat_frn,
                                         float yaw_rate_rad_s,
                                         const std::array<float, 3>& thrust_body) {
    auto sp = px4_msgs::msg::VehicleAttitudeSetpoint();
    sp.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    sp.yaw_sp_move_rate = yaw_rate_rad_s;
    for (size_t i = 0; i < 4; ++i) {
        sp.q_d[i] = quat_frn[i];
    }
    for (size_t i = 0; i < 3; ++i) {
        sp.thrust_body[i] = thrust_body[i];
    }
    pub_attitude_sp_->publish(sp);
}

void PX4Node::publish_position_setpoint(float x_ned, float y_ned, float z_ned, float yaw_rad) {
    auto ts = px4_msgs::msg::TrajectorySetpoint();
    ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    ts.position[0] = x_ned;
    ts.position[1] = y_ned;
    ts.position[2] = z_ned;
    ts.yaw = yaw_rad;
    ts.yawspeed = 0.0f;
    pub_traj_->publish(ts);
}

void PX4Node::sleep_dt() {
    if (dt_ > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(dt_));
    }
}

bool PX4Node::wait_for_local(double timeout) {
    auto start = this->now();
    while (last_local_ == nullptr && 
           (this->now() - start).seconds() < timeout) {
        rclcpp::spin_some(this->get_node_base_interface());
        std::this_thread::sleep_for(10ms);
    }
    return last_local_ != nullptr;
}

bool PX4Node::wait_until(std::function<bool()> pred, double timeout) {
    auto start = this->now();
    while ((this->now() - start).seconds() < timeout) {
        rclcpp::spin_some(this->get_node_base_interface());
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(20ms);
    }
    return false;
}
