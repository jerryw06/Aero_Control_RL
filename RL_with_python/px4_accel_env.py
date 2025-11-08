# px4_accel_env.py
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
)

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleStatus, VehicleLocalPosition
)

# Forward-acceleration control: no frame conversion helper needed

class _PX4Node(Node):
    def __init__(self, hz: float):
        super().__init__("px4_accel_env")
        self.rate_hz = hz
        self.dt = 1.0 / hz

        # QoS: small buffers
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.pub_offboard = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', qos_pub)

        # Subscribers
        self.sub_status = self.create_subscription(VehicleStatus, 'fmu/out/vehicle_status', self._on_status, qos_sub)
        self.sub_local = self.create_subscription(VehicleLocalPosition, 'fmu/out/vehicle_local_position', self._on_local, qos_sub)

        # State mirror
        self.armed = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.last_local = None
        self.last_local_time = None  # wall-clock time when last_local updated

    def _on_status(self, msg: VehicleStatus):
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state

    def _on_local(self, msg: VehicleLocalPosition):
        self.last_local = msg
        self.last_local_time = time.time()

    def send_vehicle_cmd(self, command, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        m = VehicleCommand()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.command = int(command)
        m.param1, m.param2, m.param3, m.param4 = float(p1), float(p2), float(p3), float(p4)
        m.param5, m.param6, m.param7 = float(p5), float(p6), float(p7)
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        self.pub_cmd.publish(m)

    def publish_offboard_heartbeat(self, *, position=False, velocity=False, acceleration=False, attitude=False, body_rate=False):
        hb = OffboardControlMode()
        hb.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        hb.position     = bool(position)
        hb.velocity     = bool(velocity)
        hb.acceleration = bool(acceleration)
        hb.attitude     = bool(attitude)
        hb.body_rate    = bool(body_rate)
        self.pub_offboard.publish(hb)

    def publish_accel_setpoint(self, a_fwd: float):
        ts = TrajectorySetpoint()
        ts.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        ts.position = [0.0, 0.0, 0.0]
        ts.velocity = [0.0, 0.0, 0.0]
        # PX4 NED: forward (north) is negative X; positive a_fwd -> -X accel
        # Add gravity compensation on Z to maintain hover during forward control
        ts.acceleration[0], ts.acceleration[1], ts.acceleration[2] = -float(a_fwd), 0.0, -9.81
        ts.jerk = [0.0, 0.0, 0.0]
        ts.yaw = 0.0
        ts.yawspeed = 0.0
        self.pub_traj.publish(ts)

    def publish_accel_setpoint_fwd(self, a_fwd: float, a_ned_z: float):
        ts = TrajectorySetpoint()
        ts.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        ts.acceleration[0] = -float(a_fwd)
        ts.acceleration[1] = 0.0
        ts.acceleration[2] = float(a_ned_z)
        ts.yaw = 0.0
        ts.yawspeed = 0.0
        self.pub_traj.publish(ts)

    def publish_accel_setpoint_lat(self, a_lat: float, a_ned_z: float):
        ts = TrajectorySetpoint()
        ts.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        ts.acceleration[0] = 0.0
        ts.acceleration[1] = float(a_lat)
        ts.acceleration[2] = float(a_ned_z)
        ts.yaw = 0.0
        ts.yawspeed = 0.0
        self.pub_traj.publish(ts)

    def sleep_dt(self):
        time.sleep(self.dt)

    # Teleporting support has been removed per user request

    def wait_for_local(self, timeout=2.0) -> bool:
        t0 = time.time()
        while self.last_local is None and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
        return self.last_local is not None

    def wait_until(self, pred, timeout=3.0):
        t0 = time.time()
        while (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            if pred():
                return True
        return False


class PX4AccelEnv(gym.Env):
    """
    RL controls LATERAL (Y) ACCELERATION only.
    Pre-RL staging: enter OFFBOARD + ARM, hold altitude, re-zero local origin, then start RL.
    """
    metadata = {"render_modes": []}

    @staticmethod
    def _pose_to_str(ned: np.ndarray) -> str:
        return f"x={ned[0]:.2f} y={ned[1]:.2f} z={ned[2]:.2f}"

    def __init__(self,
                 rate_hz=50.0,
                 a_max=3.0,
                 ep_time_s=6.0,
                 target_up_m=2.0,
                 target_lateral_m=2.0,
                 start_up_m=2.5,            # kept but not used for climb
                 takeoff_kp=2.5,            # PD gains for staging ascent (accel command)
                 takeoff_kd=1.2,
                 takeoff_tol=0.05,          # within 5 cm
                 settle_steps=20,           # ~0.25 s at 80 Hz
                 takeoff_max_time=8.0,
                 safety_verbose=True):
        super().__init__()
        self.rate_hz = rate_hz
        self.node = None
        self.a_max = float(a_max)
        self.dt = 1.0 / rate_hz
        self.max_steps = int(ep_time_s * rate_hz)
        self.target_up = float(target_up_m)
        self.target_lateral = float(target_lateral_m)
        self.start_up_m = float(start_up_m)
        self.takeoff_kp = float(takeoff_kp)
        self.takeoff_kd = float(takeoff_kd)
        self.takeoff_tol = float(takeoff_tol)
        self.settle_steps = int(settle_steps)
        self.takeoff_max_time = float(takeoff_max_time)

        self.action_space = spaces.Box(low=np.array([-self.a_max], dtype=np.float32),
                                       high=np.array([ self.a_max], dtype=np.float32),
                                       dtype=np.float32)
        # Observation: [x_pos, y_pos, z_pos, vx, vy, vz]
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0, -5.0, -10.0, -10.0, -10.0], dtype=np.float32),
                                            high=np.array([ 10.0,  10.0,  5.0,  10.0,  10.0,  10.0], dtype=np.float32),
                                            dtype=np.float32)

        self._episode_origin_ned = None
        self._current_global_ned = None
        self._current_local_ned = None
        self._hover_z_ned = None
        self._reset_reward_tracking()
        self.hold_kp = 2.0
        self.hold_kd = 1.0
        self.spawn_xyz = np.array([0.0, 0.0, 0.7], dtype=np.float32)
        self._step = 0
        self._episode_running = False
        self._ready_for_rl = False

        # Frequency monitor variables
        self._freq_monitor_start_time = None
        self._freq_monitor_step_count = 0

        # Reward weights (importance ranking)
        # Highest importance: FINAL distance to target (applied as terminal bonus)
        # Next: step-wise improvement toward target
        # Then: constraint penalties (height, X-drift, overshoot)
        # Least: action cost + new stability penalties
        self.w_terminal   = 100.0  # -w_terminal * final_distance (at episode end)
        self.w_improve    = 10.0   #  w_improve  * (prev_dist - curr_dist) per step
        self.w_height     = 5.0    # -w_height  * |z|
        self.w_xdrift     = 3.0    # -w_xdrift  * |x|
        self.w_overshoot  = 4.0    # -w_overshoot * overshoot_y
        self.w_action     = 0.01   # -w_action * a^2
        self.w_vel_limit  = 2.0    # - penalty when |vel| exceeds soft cap
        self.w_stagnation = 8.0    # - penalty for no progress
        self.w_oscillation= 6.0    # - penalty for rapid oscillations
        self.w_state_fault= 50.0   # - heavy penalty for invalid state (NaN/Inf/stale)
        self.w_backward   = 2.0    # - penalty for moving away from goal
        self.w_action_slew= 0.02   # - penalty for requesting too fast action changes
        self.w_hold_bonus = 0.5    # + reward per step when within tight target band

        # Safety and sanity thresholds (tunable)
        self.max_abs_vxy       = 6.0   # m/s horizontal speed hard cap (termination)
        self.soft_abs_vxy      = 4.5   # soft velocity cap (penalty region)
        self.max_abs_vz        = 3.0   # m/s vertical speed cap
        self.max_abs_x         = 2.0   # m off-course in X before terminate (also penalized)
        self.max_abs_z         = 1.5   # m height deviation before terminate (also penalized)
        self.max_abs_y         = 5.0   # m lateral bound before terminate
        self.stale_timeout_s   = 0.5   # no new local position for this long -> terminate
        self.no_progress_horizon = 25  # steps with negligible improvement triggers penalty
        self.min_improvement_eps = 0.01 # m deemed as progress per step
        self.oscillation_window = 20   # steps to check for sign-flip oscillation in vy
        self.oscillation_flips  = 6    # number of sign flips in window considered oscillation
        self.safety_verbose = bool(safety_verbose)
        # Action rate limiting (slew) settings
        self.max_action_slew_rate = 20.0  # (m/s^3 equivalent) per second
        self._prev_action = 0.0

    # --- Lifecycle helpers ---
    def _reset_reward_tracking(self):
        self._episode_reward_sum = 0.0
        self._episode_improvement_reward = 0.0
        self._episode_speed_reward = 0.0
        self._episode_distance_penalty = 0.0
        self._episode_backward_penalty = 0.0
        self._episode_overshoot_penalty = 0.0
        self._episode_action_penalty = 0.0
        self._episode_height_penalty = 0.0
        self._episode_x_drift_penalty = 0.0
        self._episode_velocity_penalty = 0.0
        self._episode_stagnation_penalty = 0.0
        self._episode_oscillation_penalty = 0.0
        self._episode_stale_penalty = 0.0
        self._episode_state_fault_penalty = 0.0
        self._episode_soft_vel_penalty = 0.0
        self._episode_action_slew_penalty = 0.0
        self._episode_hold_bonus = 0.0
        self._last_distance_to_target = None
        self._last_lateral_position = None
        self._last_vy_local = None
        self._no_progress_steps = 0
        self._vy_sign_history = []

    def _ensure_node(self):
        if self.node is None:
            rclpy.init(args=None)
            self.node = _PX4Node(self.rate_hz)

    def _disarm_and_stop(self):
        """Send a disarm command and do not publish further setpoints/heartbeats."""
        n = self.node
        n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        n.wait_until(lambda: not n.armed, timeout=3.0)
    
    def _read_observation(self):
        """
        Read current position (X, Y, Z) and velocities (vx, vy, vz) from PX4 local position.
        Returns np.array([x_m, y_m, z_m, vx, vy, vz]).
        """
        local = self.node.last_local
        if local is None:
            # No data yet → return neutral observation
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        global_ned = np.array([float(local.x), float(local.y), float(local.z)], dtype=np.float32)
        self._current_global_ned = global_ned

        if self._episode_origin_ned is None:
            self._episode_origin_ned = global_ned.copy()

        local_ned = global_ned - self._episode_origin_ned
        self._current_local_ned = local_ned

        # Velocities in NED frame
        vx_ned = float(local.vx)
        vy_ned = float(local.vy)
        vz_ned = float(local.vz)

        x_m = local_ned[0]
        y_m = local_ned[1]
        z_m = local_ned[2]

        return np.array([x_m, y_m, z_m, vx_ned, vy_ned, vz_ned], dtype=np.float32)


    def _arm_offboard(self):
        """
        Robust OFFBOARD takeoff: enter OFFBOARD, arm, climb smoothly to
        start_up_m, hold hover briefly. If hover fails, land, disarm, wait, and retry.
        """
        n = self.node
        max_takeoff_retries = 3
        
        for takeoff_attempt in range(max_takeoff_retries):
            if takeoff_attempt > 0:
                print(f"[STATE] Takeoff retry {takeoff_attempt + 1}/{max_takeoff_retries}...")
            
            # Warmup Offboard heartbeats and neutral position setpoints
            for _ in range(30):
                n.publish_offboard_heartbeat(position=True)
                ts = TrajectorySetpoint()
                ts.timestamp = int(n.get_clock().now().nanoseconds / 1000)
                ts.position = [0.0, 0.0, 0.0]
                n.pub_traj.publish(ts)
                rclpy.spin_once(n, timeout_sec=0.0)
                n.sleep_dt()

            # Switch to OFFBOARD and arm
            print("[STATE] Entering OFFBOARD mode...")
            n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
            n.wait_until(lambda: n.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD, timeout=3.0)
            print("[STATE] Arming vehicle...")
            arm_attempts = 0
            max_arm_attempts = 3
            armed = False
            while arm_attempts < max_arm_attempts and not armed:
                n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                n.wait_until(lambda: n.armed, timeout=3.0)
                if n.armed:
                    print("[STATE] Armed successfully.")
                    armed = True
                    break
                else:
                    print(f"[WARN] Arm attempt {arm_attempts+1} failed. Disarming, waiting 2s, and retrying...")
                    n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                    n.wait_until(lambda: not n.armed, timeout=2.0)
                    time.sleep(2.0)
                    arm_attempts += 1
            if not armed:
                print("[ERROR] Failed to arm after multiple attempts.")
                if takeoff_attempt < max_takeoff_retries - 1:
                    continue  # Try next takeoff attempt
                else:
                    return False

            # Smooth climb to target altitude via interpolated position setpoints
            n.wait_for_local(timeout=10.0)
            local0 = n.last_local
            if local0 is None:
                print("[ERROR] No VehicleLocalPosition messages for takeoff.")
                if takeoff_attempt < max_takeoff_retries - 1:
                    # Land and disarm before retry
                    print("[STATE] Landing and disarming before retry...")
                    n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    time.sleep(2.0)
                    n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                    n.wait_until(lambda: not n.armed, timeout=3.0)
                    time.sleep(2.0)
                    continue
                else:
                    return False
            
            pre_z0 = float(local0.z)
            target_z_ned = pre_z0 - self.start_up_m  # up in NED is negative
            steps = int(3.0 / self.dt)
            print(f"[STATE] Performing smooth takeoff to {self.start_up_m:.2f} m...")
            for i in range(steps):
                alpha = (i + 1) / steps
                interp_z = pre_z0 + alpha * (target_z_ned - pre_z0)
                ts = TrajectorySetpoint()
                ts.timestamp = int(n.get_clock().now().nanoseconds / 1000)
                ts.position[0] = 0.0
                ts.position[1] = 0.0
                ts.position[2] = interp_z
                ts.yaw = 0.0
                n.pub_traj.publish(ts)
                n.publish_offboard_heartbeat(position=True)
                rclpy.spin_once(n, timeout_sec=0.0)
                n.sleep_dt()

            # Hold hover for stabilization in position mode
            print("[STATE] Hovering to stabilize...")
            hover_t0 = time.time()
            stable_start = None
            hover_success = False
            while True:
                ts = TrajectorySetpoint()
                ts.timestamp = int(n.get_clock().now().nanoseconds / 1000)
                ts.position = [0.0, 0.0, target_z_ned]
                ts.yaw = 0.0
                n.pub_traj.publish(ts)
                n.publish_offboard_heartbeat(position=True)
                rclpy.spin_once(n, timeout_sec=0.0)
                n.sleep_dt()
                now = time.time()
                local_now = n.last_local
                if local_now is not None:
                    z_err = abs(float(local_now.z) - target_z_ned)
                    if z_err <= 0.5:  # Allow ±0.5m for arming/hover
                        stable_start = stable_start or now
                        if (now - stable_start) >= 1.0:
                            print("[STATE] Hover stabilized within 0.5 m for 1.0 s.")
                            hover_success = True
                            break
                    else:
                        stable_start = None
                if (now - hover_t0) >= 10.0:
                    print("[WARN] Hover stabilization timeout (10 s).")
                    break

            if hover_success:
                # Success! Re-zero local origin and mark ready
                n.wait_for_local(timeout=1.0)
                if n.last_local is not None:
                    self._episode_origin_ned = np.array(
                        [float(n.last_local.x), float(n.last_local.y), float(n.last_local.z)],
                        dtype=np.float32,
                    )
                    self._current_global_ned = self._episode_origin_ned.copy()
                    self._current_local_ned = np.zeros(3, dtype=np.float32)
                self._hover_z_ned = target_z_ned
                print("[STATE] Hover complete — switching to lateral RL control.")
                self._ready_for_rl = True
                return True
            else:
                # Hover failed - land, disarm, wait, and retry
                if takeoff_attempt < max_takeoff_retries - 1:
                    print("[STATE] Landing and disarming before retry...")
                    n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    time.sleep(2.0)
                    n.send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                    n.wait_until(lambda: not n.armed, timeout=3.0)
                    time.sleep(2.0)
                else:
                    print("[ERROR] Failed to stabilize hover after all retry attempts.")
                    return False
        
        return False
    

    # --- Gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_node()
        n = self.node
        self._current_global_ned = None
        self._current_local_ned = None
        self._episode_origin_ned = None
        self._reset_reward_tracking()

        # Ensure we have a local position
        print("[STATE] Waiting for PX4 local position...")
        ok = n.wait_for_local(timeout=10.0)
        if not ok or n.last_local is None:
            raise RuntimeError("No VehicleLocalPosition messages received. Check bridge/topic.")

        # Stop any prior episode traffic and disarm to make episodes independent
        self._disarm_and_stop()
        print("Disarmed prior to new episode.")

        # Grab a starting origin (will be re-zeroed after staging)
        n.wait_for_local(timeout=2.0)
        local = n.last_local
        if local is None:
            raise RuntimeError("No VehicleLocalPosition messages received.")
        self._episode_origin_ned = np.array(
            [float(local.x), float(local.y), float(local.z)],
            dtype=np.float32,
        )
        self._hover_z_ned = None

        # Proceed with normal arm + smooth takeoff + hover
        self._ready_for_rl = False
        ok = self._arm_offboard()
        if not ok or not self._ready_for_rl:
            raise RuntimeError("Failed to arm/hover before RL.")

        self._step = 0
        self._episode_running = True

        obs = self._read_observation()  # should be near [0, 0] after re-zero
        if self._current_global_ned is not None and self._current_local_ned is not None:
            print("[STATE] Start pose:")
            print(f"         Global NED: {self._pose_to_str(self._current_global_ned)}")
            print(f"         Local NED:  {self._pose_to_str(self._current_local_ned)}")
        info = {}
        print("[STATE] RL control begins.")
        # Reset frequency monitor
        self._freq_monitor_start_time = time.perf_counter()
        self._freq_monitor_step_count = 0
        return obs, info


    def step(self, action):
        if not self._episode_running:
            raise RuntimeError("Episode not running. Call reset().")
        if not self._ready_for_rl:
            raise RuntimeError("RL step() called before PX4 was ready for RL control.")

        # Defensive action handling
        if action is None:
            if self.safety_verbose:
                print("[SAFETY] Received None action; substituting zero.")
            action = np.zeros(1, dtype=np.float32)
        if isinstance(action, (float, int)):
            action = np.array([float(action)], dtype=np.float32)
        if not isinstance(action, (np.ndarray, list, tuple)):
            raise TypeError(f"Action must be array-like; got {type(action)}")
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            if self.safety_verbose:
                print("[SAFETY] NaN/Inf detected in action; replacing with zeros and applying heavy penalty.")
            action = np.zeros_like(action)
            fault_action = True
        else:
            fault_action = False

        # --- Parse action ---
        a_lat_req = float(np.clip(action[0], -self.a_max, self.a_max))
        # Apply slew rate limit around previous applied action
        da_allowed = float(self.max_action_slew_rate * self.dt)
        a_lat = float(np.clip(a_lat_req, self._prev_action - da_allowed, self._prev_action + da_allowed))
        slew_penalty = self.w_action_slew * abs(a_lat - a_lat_req)
        if abs(action[0]) > self.a_max and self.safety_verbose:
            print(f"[SAFETY] Action saturated from {action[0]:.3f} to {a_lat_req:.3f}")

        # Publish Offboard acceleration setpoint (lateral + tiny Z-hold PD correction)
        self.node.publish_offboard_heartbeat(acceleration=True)
        lp = self.node.last_local
        if lp is not None and self._hover_z_ned is not None:
            z_ned = float(lp.z)   # +down
            vz_ned = float(lp.vz) # +down
            height_up = -(z_ned - self._hover_z_ned)
            vz_up = -vz_ned
            a_up_corr = self.hold_kp * (0.0 - height_up) + self.hold_kd * (0.0 - vz_up)
            a_ned_z = -9.81 - a_up_corr
            # Clamp Z accel to safe band and note saturation
            a_z_sat = False
            if a_ned_z < -15.0:
                a_ned_z = -15.0
                a_z_sat = True
            elif a_ned_z > -5.0:
                a_ned_z = -5.0
                a_z_sat = True
        else:
            a_ned_z = -9.81
            a_z_sat = False
        self.node.publish_accel_setpoint_lat(a_lat, a_ned_z)
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.node.sleep_dt()

        # --- Read current observation ---
        self._step += 1
        obs = self._read_observation()  # -> [x_m, y_m, z_m, vx, vy, vz]
        x_m, y_m, z_m, vx, vy, vz = obs

        # Validate observation sanity
        obs_fault = False
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs_fault = True
            if self.safety_verbose:
                print("[SAFETY] Observation contains NaN/Inf values.")
        if self.node.last_local_time is not None:
            stale_age = time.time() - self.node.last_local_time
        else:
            stale_age = float('inf')
        stale = stale_age > self.stale_timeout_s
        if stale and self.safety_verbose:
            print(f"[SAFETY] Local position stale for {stale_age:.2f}s (> {self.stale_timeout_s:.2f}s)")

        # --- Frequency monitor ---
        self._freq_monitor_step_count += 1
        if self._freq_monitor_step_count % 100 == 0:
            now = time.perf_counter()
            elapsed = now - self._freq_monitor_start_time
            if elapsed > 0:
                freq = self._freq_monitor_step_count / elapsed
                print(f"[FREQ] Control loop: {freq:.2f} Hz over last {self._freq_monitor_step_count} steps.")
            self._freq_monitor_start_time = now
            self._freq_monitor_step_count = 0

        # --- Compute reward components ---
        target_local_lateral = float(self.target_lateral)
        error_lateral = target_local_lateral - y_m
        distance_to_target = abs(error_lateral)
        overshoot = max(0.0, y_m - target_local_lateral)

        # Height deviation (Z in NED: negative is up, so we want z_m ≈ 0)
        height_error = abs(z_m)  # Should stay near 0 after re-zero

        # X-axis drift (should stay near 0, no forward/backward motion)
        x_drift = abs(x_m)

        # Step-wise improvement toward target (potential-based shaping)
        if self._last_distance_to_target is None:
            improvement = 0.0
        else:
            improvement = float(self._last_distance_to_target - distance_to_target)

        # Reward terms (no speed reward; we emphasize progress and constraints)
        improvement_term = self.w_improve * improvement
        overshoot_term   = self.w_overshoot * overshoot
        action_term      = self.w_action * (a_lat ** 2)
        height_term      = self.w_height * height_error
        x_drift_term     = self.w_xdrift * x_drift
        # Backward (moving away) penalty based on vy sign vs error
        if distance_to_target > 0.02:  # ignore noise near target
            dir_ok = np.sign(error_lateral) * vy >= -0.02
            backward_penalty = 0.0 if dir_ok else self.w_backward * abs(vy)
        else:
            backward_penalty = 0.0
        # Tiny hold bonus when within tight band around target
        hold_bonus = self.w_hold_bonus if distance_to_target < 0.05 else 0.0

        # Velocity soft limit penalties (non-terminating)
        soft_vel_penalty = 0.0
        horiz_speed = float(np.sqrt(vx**2 + vy**2))
        if horiz_speed > self.soft_abs_vxy:
            soft_vel_penalty = self.w_vel_limit * (horiz_speed - self.soft_abs_vxy)

        # Stagnation detection (no progress toward target)
        if improvement < self.min_improvement_eps:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        stagnation_penalty = 0.0
        if self._no_progress_steps > self.no_progress_horizon:
            stagnation_penalty = self.w_stagnation * (self._no_progress_steps - self.no_progress_horizon)

        # Oscillation detection in vy sign flips
        self._vy_sign_history.append(np.sign(vy))
        if len(self._vy_sign_history) > self.oscillation_window:
            self._vy_sign_history.pop(0)
        oscillation_penalty = 0.0
        if len(self._vy_sign_history) == self.oscillation_window:
            flips = np.sum(np.diff(self._vy_sign_history) != 0)
            if flips >= self.oscillation_flips:
                oscillation_penalty = self.w_oscillation * (flips - self.oscillation_flips + 1)
                if self.safety_verbose:
                    print(f"[SAFETY] Oscillation detected: {flips} sign flips in last {self.oscillation_window} steps.")

        # Fault penalties (NaN/Inf or stale)
        state_fault_penalty = 0.0
        if obs_fault or fault_action:
            state_fault_penalty += self.w_state_fault
        if stale:
            state_fault_penalty += self.w_state_fault * 0.5

        reward = (improvement_term
                  - overshoot_term
                  - action_term
                  - height_term
                  - x_drift_term
                  - backward_penalty
                  - slew_penalty
                  - soft_vel_penalty
                  - stagnation_penalty
                  - oscillation_penalty
                  - state_fault_penalty
                  + hold_bonus)

        # Track components
        self._episode_reward_sum += reward
        self._episode_improvement_reward += improvement_term
        self._episode_overshoot_penalty += overshoot_term
        self._episode_action_penalty += action_term
        self._episode_height_penalty += height_term
        self._episode_x_drift_penalty += x_drift_term
        self._episode_backward_penalty += backward_penalty
        self._episode_action_slew_penalty += slew_penalty
        self._episode_soft_vel_penalty += soft_vel_penalty
        self._episode_stagnation_penalty += stagnation_penalty
        self._episode_oscillation_penalty += oscillation_penalty
        self._episode_state_fault_penalty += state_fault_penalty
        self._episode_hold_bonus += hold_bonus
        self._last_distance_to_target = distance_to_target
        self._last_lateral_position = y_m
        self._last_vy_local = vy
        self._prev_action = a_lat

        # --- Termination conditions ---
        # Safety limits: terminate if drone goes way off course
        terminated = False
        if height_error > self.max_abs_z:
            print(f"[SAFETY] Episode terminated: excessive height deviation ({height_error:.2f}m)")
            terminated = True
        elif x_drift > self.max_abs_x:
            print(f"[SAFETY] Episode terminated: excessive X-axis drift ({x_drift:.2f}m)")
            terminated = True
        elif abs(y_m) > self.max_abs_y:
            print(f"[SAFETY] Episode terminated: excessive lateral displacement ({y_m:.2f}m)")
            terminated = True
        elif horiz_speed > self.max_abs_vxy:
            print(f"[SAFETY] Episode terminated: excessive horizontal speed ({horiz_speed:.2f}m/s)")
            terminated = True
        elif stale:
            print(f"[SAFETY] Episode terminated: stale local position ({stale_age:.2f}s)")
            terminated = True
        elif obs_fault:
            print("[SAFETY] Episode terminated: invalid observation values (NaN/Inf).")
            terminated = True
        elif (not self.node.armed) or (self.node.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            print("[SAFETY] Episode terminated: vehicle left OFFBOARD or disarmed.")
            terminated = True
        
        truncated = bool(self._step >= self.max_steps)

        # Apply terminal bonus to strongly emphasize final distance to target
        if terminated or truncated:
            terminal_bonus = -self.w_terminal * distance_to_target
            reward += terminal_bonus
            self._episode_reward_sum += terminal_bonus

        if terminated or truncated:
            if self._current_global_ned is not None and self._current_local_ned is not None:
                print("[STATE] End pose:")
                print(f"         Global NED: {self._pose_to_str(self._current_global_ned)}")
                print(f"         Local NED:  {self._pose_to_str(self._current_local_ned)}")
            final_distance = getattr(self, "_last_distance_to_target", None)
            final_lateral = getattr(self, "_last_lateral_position", None)
            if final_distance is not None and final_lateral is not None:
                print("[REWARD] Episode distance to target (local Y): "
                      f"{final_distance:.3f} m (target={self.target_lateral:.2f} m, final={final_lateral:.3f} m)")
                print("[REWARD] Episode return breakdown:")
                print(f"         + Improvement reward:  {self._episode_improvement_reward:.3f}")
                print(f"         - Overshoot penalty:   {self._episode_overshoot_penalty:.3f}")
                print(f"         - Action penalty:      {self._episode_action_penalty:.3f}")
                print(f"         - Height penalty:      {self._episode_height_penalty:.3f}")
                print(f"         - X-drift penalty:     {self._episode_x_drift_penalty:.3f}")
                print(f"         - Backward penalty:    {self._episode_backward_penalty:.3f}")
                print(f"         - Action slew penalty: {self._episode_action_slew_penalty:.3f}")
                print(f"         - Soft vel penalty:    {self._episode_soft_vel_penalty:.3f}")
                print(f"         - Stagnation penalty:  {self._episode_stagnation_penalty:.3f}")
                print(f"         - Oscillation penalty: {self._episode_oscillation_penalty:.3f}")
                print(f"         - State fault penalty: {self._episode_state_fault_penalty:.3f}")
                print(f"         + Hold bonus:          {self._episode_hold_bonus:.3f}")
                print(f"         = Total return:        {self._episode_reward_sum:.3f}")
            self._disarm_and_stop()
            self._episode_running = False
            self._ready_for_rl = False

        # --- Optional debug printout every 20 steps ---
        #if self._step % 100 == 0:
            #print(f"[RL] step={self._step:03d}  y={y_m:.2f}m  target={target_local_lateral:.2f}m  vy={vy:.2f}m/s  a={a_lat:.2f}  R={reward:.3f}")

        info = {
            "reward": reward,
            "vy": float(vy),
            "target_local_y": float(target_local_lateral),
            "lateral_position": float(y_m),
            "x_position": float(x_m),
            "z_position": float(z_m),
            "distance_to_target": float(distance_to_target),
            "height_error": float(height_error),
            "x_drift": float(x_drift),
            "reward_components": {
                "improvement": float(improvement_term),
                "overshoot_penalty": float(overshoot_term),
                "action_penalty": float(action_term),
                "height_penalty": float(height_term),
                "x_drift_penalty": float(x_drift_term),
                "backward_penalty": float(backward_penalty),
                "action_slew_penalty": float(slew_penalty),
                "soft_vel_penalty": float(soft_vel_penalty),
                "stagnation_penalty": float(stagnation_penalty),
                "oscillation_penalty": float(oscillation_penalty),
                "state_fault_penalty": float(state_fault_penalty),
                "hold_bonus": float(hold_bonus),
            },
            "global_pose_ned": self._current_global_ned.copy() if self._current_global_ned is not None else None,
            "local_pose_ned": self._current_local_ned.copy() if self._current_local_ned is not None else None,
            "horiz_speed": float(horiz_speed),
            "stale_age": float(stale_age),
            "stagnation_steps": int(self._no_progress_steps),
            "a_lat": float(a_lat),
            "a_lat_requested": float(a_lat_req),
            "a_ned_z": float(a_ned_z),
            "a_z_saturated": bool(a_z_sat),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.node is not None:
            try:
                self._disarm_and_stop()
            except Exception:
                pass
            self.node.destroy_node()
            rclpy.shutdown()
            self.node = None
