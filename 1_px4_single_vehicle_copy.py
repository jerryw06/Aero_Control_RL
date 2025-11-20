#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
# Auxiliary scipy and numpy modules
import os.path
from scipy.spatial.transform import Rotation

# Try to expose ROS 2 (rclpy) to Isaac Sim's Python by adding common site-packages
# paths based on ROS_DISTRO and AMENT_PREFIX_PATH. This allows embedded ROS 2
# services to run without enabling the Isaac ROS 2 Bridge.
import sys, os
def _inject_ros2_site_packages():
    try:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        candidates = []
        ros_distro = os.environ.get("ROS_DISTRO")
        
        # IMPORTANT: Only inject paths that match Isaac Sim's Python version
        # Isaac Sim uses Python 3.11, system ROS2 might be Python 3.12
        # Mixing them will cause "module has no attribute 'init'" errors
        if ros_distro:
            ros_path = f"/opt/ros/{ros_distro}/lib/python{py_ver}/site-packages"
            if not os.path.isdir(ros_path):
                # Don't inject incompatible Python version paths
                ros_path = None
            else:
                candidates.append(ros_path)
        
        # Also include any colcon/overlay prefixes from AMENT_PREFIX_PATH
        ament_prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(":")
        for prefix in filter(None, ament_prefixes):
            candidates.append(os.path.join(prefix, f"lib/python{py_ver}/site-packages"))
        # De-duplicate and append if exists
        added = []
        for p in candidates:
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
                added.append(p)
    except Exception as e:
        pass  # Silently handle ROS2 path injection failures

_inject_ros2_site_packages()

# Optional: ROS 2 services to request a reset or PX4 restart from external processes.
# (These do NOT require enabling the Isaac ROS2 Bridge extension; they run rclpy inside
#  the simulator process. If you need generic attribute services like /set_prim_attribute,
#  then you must enable the Isaac ROS2 Bridge extension separately in Isaac Sim.)
try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Empty
    # We optionally expose an additional service for a "precise" reset and PX4 restart.
    ROS2_AVAILABLE = True
except Exception as e:
    ROS2_AVAILABLE = False


class IsaacControlServers:
    """Aggregates ROS2 services for resetting the drone precisely and restarting PX4.

    Services exposed (std_srvs/Empty):
      /isaac_sim/reset_drone            -> world.reset() (legacy behavior)
      /isaac_sim/reset_drone_precise    -> force drone back to ORIGINAL spawn pose & zero velocities
      /isaac_sim/restart_px4            -> attempt to restart PX4 backend (best-effort)

    Notes:
      * These custom services run inside the Isaac Sim Python process using rclpy.
      * They do NOT require enabling the Isaac ROS2 Bridge extension.
      * If you want generic prim attribute manipulation over ROS2 (e.g. /set_prim_attribute),
        enable the Isaac ROS2 Bridge extension in the GUI or via command line args.
    """
    def __init__(self, app_ref):
        self._app = app_ref
        self._node = None
        self._thread = None
        if not ROS2_AVAILABLE:
            return
        import threading
        from std_msgs.msg import Bool
        rclpy.init(args=None)
        self._node = Node("isaac_control_servers")
        
        # Standard reset (world.reset)
        self._node.create_service(Empty, "/isaac_sim/reset_drone", self._on_standard_reset)
        # Precise reset (teleport drone to stored original spawn pose even if user teleported it during sim)
        self._node.create_service(Empty, "/isaac_sim/reset_drone_precise", self._on_precise_reset)
        # PX4 restart
        self._node.create_service(Empty, "/isaac_sim/restart_px4", self._on_restart_px4)
        # Combined precise reset + PX4 restart convenience
        self._node.create_service(Empty, "/isaac_sim/reset_and_restart_px4", self._on_reset_and_restart)
        
        # Topic-based reset (more reliable for Python<->C++ communication)
        self._reset_sub = self._node.create_subscription(
            Bool, 
            "/isaac_sim/reset_request", 
            self._on_reset_request_topic, 
            10
        )
        self._reset_done_pub = self._node.create_publisher(Bool, "/isaac_sim/reset_done", 10)
        
        print("[Isaac Sim] ROS2 reset topics ready")

        def _spin():
            try:
                while rclpy.ok():
                    rclpy.spin_once(self._node, timeout_sec=0.1)
            except KeyboardInterrupt:
                pass
        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    # --- Service callbacks ---
    def _on_standard_reset(self, request, response):
        self._app.request_reset(world_only=True)
        return response

    def _on_precise_reset(self, request, response):
        self._app.request_reset(world_only=False)  # will trigger precise drone pose restore
        return response

    def _on_restart_px4(self, request, response):
        self._app.request_px4_restart()
        return response

    def _on_reset_and_restart(self, request, response):
        # Precise reset then restart px4 on next frame
        self._app.request_reset(world_only=False)
        self._app.request_px4_restart()
        return response
    
    def _on_reset_request_topic(self, msg):
        """Topic-based reset callback - more reliable for Python<->C++ communication"""
        self._app.request_reset(world_only=False)  # Precise reset
        # Publish confirmation after a brief delay to ensure reset is queued
        import time
        time.sleep(0.1)
        from std_msgs.msg import Bool
        done_msg = Bool()
        done_msg.data = True
        self._reset_done_pub.publish(done_msg)

    def shutdown(self):
        if not ROS2_AVAILABLE:
            return
        try:
            if self._node is not None:
                self._node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        # Keep a reference to the drone instance in case we want direct control later
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Capture the original spawn transform for the drone (for precise resets later)
        try:
            from pxr import UsdGeom, Gf
            import omni.usd as omni_usd
            stage = omni_usd.get_context().get_stage()
            drone_path = getattr(self.drone, "prim_path", "/World/quadrotor")
            prim = stage.GetPrimAtPath(drone_path)
            self._drone_spawn_matrix = omni_usd.get_world_transform_matrix(prim) if prim.IsValid() else None
        except Exception as e:
            pass  # Silently handle initial transform capture failure
            self._drone_spawn_matrix = None

        # Reset control flags
        self._pending_reset = False
        self._pending_precise = False
        self._pending_px4_restart = False

        # Start optional ROS 2 control services
        self._control_servers = IsaacControlServers(self) if ROS2_AVAILABLE else None

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def request_reset(self, world_only: bool = True):
        """Mark a reset.

        Args:
            world_only: If True, perform world.reset() (baseline). If False, also force drone back to original
                        captured spawn pose even if it was teleported, and zero velocities manually.
        """
        self._pending_reset = True
        self._pending_precise = not world_only

    def request_px4_restart(self):
        """Mark a PX4 restart attempt on next frame (safe point)."""
        self._pending_px4_restart = True

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Execute a deferred reset if requested via ROS2 service
            if self._pending_reset:
                # First perform the normal world reset (brings physics & articulations back to initial config)
                self.world.reset()

                # If precise reset requested, explicitly reapply stored drone transform
                if self._pending_precise and self._drone_spawn_matrix is not None:
                    try:
                        from pxr import Gf
                        import omni.usd as omni_usd
                        import omni.usd.utils as usd_utils
                        stage = omni_usd.get_context().get_stage()
                        prim = stage.GetPrimAtPath(self.drone.prim_path)
                        if prim.IsValid():
                            usd_utils.set_prim_transform(prim, self._drone_spawn_matrix)
                            # Optionally clear linear/angular velocity if PhysX API available
                            try:
                                import omni.physx
                                physx_iface = omni.physx.get_physx_interface()
                                # Best-effort velocity zeroing (depends on simulation component implementation)
                                rb_api = omni.physx.get_physx_rigid_body_api()
                                if rb_api:
                                    rb_api.set_linear_velocity(prim, (0.0, 0.0, 0.0))
                                    rb_api.set_angular_velocity(prim, (0.0, 0.0, 0.0))
                            except Exception:
                                pass
                    except Exception as e:
                        pass  # Silently handle precise reset failures

                self._pending_reset = False
                self._pending_precise = False

            # Handle PX4 restart request (after physics step to avoid mid-step issues)
            if self._pending_px4_restart:
                try:
                    backend = None
                    if hasattr(self.drone, 'backends') and self.drone.backends:
                        backend = self.drone.backends[0]
                    if backend is not None:
                        # Try common method names defensively
                        for method_name in ("restart", "reset", "shutdown"):
                            if hasattr(backend, method_name):
                                try:
                                    getattr(backend, method_name)()
                                except Exception:
                                    pass
                        # Re-initialize if constructor-like method exists
                        for method_name in ("start", "initialize"):
                            if hasattr(backend, method_name):
                                try:
                                    getattr(backend, method_name)()
                                except Exception:
                                    pass
                except Exception:
                    pass  # Silently handle PX4 restart failures
                self._pending_px4_restart = False

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        # Shutdown the reset server if it was created
        try:
            if self._control_servers is not None:
                self._control_servers.shutdown()
        except Exception:
            pass
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()