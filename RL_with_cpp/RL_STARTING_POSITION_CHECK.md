# RL Starting Position Verification

## Overview

This enhancement ensures **consistent starting conditions** for the RL agent at the beginning of every episode. The system now verifies that the drone reaches the target altitude with near-zero velocities before allowing the RL policy to take control.

## Why This Matters

Reinforcement learning training is highly sensitive to initial conditions. Without consistent starting positions:
- The RL model receives varying initial states across episodes
- Training becomes unstable and convergence is slower
- Policy performance becomes unpredictable
- Reward signals become noisy

## Implementation Details

### Location
The verification logic is implemented in the `arm_offboard()` function in `src/px4_accel_env.cpp`, right after the hover stabilization phase and before transitioning to RL control.

### Verification Criteria

The system checks three conditions:

1. **Altitude Accuracy**: `|z_actual - z_target| ≤ 0.2m`
   - The drone must be within 20cm of the target hover altitude
   - Target altitude is set by `start_up_m` parameter (default: 2.5m)

2. **Near-Zero Velocities**: `max(|vx|, |vy|, |vz|) ≤ 0.15 m/s`
   - All velocity components must be below 15 cm/s
   - Ensures the drone is effectively stationary

3. **Stability Duration**: Conditions must hold for **1.0 second**
   - Prevents premature handoff during transient oscillations
   - Ensures true stabilization rather than momentary alignment

### Workflow

```
1. Hover stabilization (existing code)
   ↓
2. [NEW] RL starting condition verification
   ↓
   → Check altitude within tolerance
   → Check all velocities near zero
   → Wait for 1.0s of stable conditions
   ↓
3. [NEW] Additional 1-second hold
   ↓
4. Re-zero origin coordinates
   ↓
5. Hand control to RL policy
```

### Timeout and Retry Logic

- **Verification Timeout**: 10 seconds
  - If conditions aren't consistently met within 10s, the takeoff sequence retries
  
- **Retry Mechanism**: Up to 3 takeoff attempts
  - On failure, the drone lands, disarms, and retries the entire sequence
  - Ensures robust initialization even with simulator delays

### Console Output

The verification process provides detailed feedback:

```
[RL-READY-CHECK] Starting conditions met - z_err=0.05m, max_vel=0.08m/s. Waiting 1.0s for stability...
[RL-READY-CHECK] ✓ Altitude within 0.2m (error: 0.05m)
[RL-READY-CHECK] ✓ All velocities ≤ 0.15m/s (vx=0.03, vy=0.02, vz=0.04)
[RL-READY-CHECK] ✓ Conditions stable for 1.0s
[RL-READY-CHECK] Starting position VERIFIED - RL takeover in 1 second...
[STATE] RL starting position verified and consistent — switching to lateral RL control.
```

## Tunable Parameters

If you need to adjust the verification thresholds, modify these constants in `arm_offboard()`:

```cpp
const float altitude_tolerance = 0.2f;   // meters
const float velocity_threshold = 0.15f;   // m/s
const double stability_duration = 1.0;    // seconds
```

**Recommendations:**
- **Tighter tolerance** (e.g., 0.1m altitude, 0.1 m/s velocity): Better consistency but may increase episode startup time
- **Looser tolerance** (e.g., 0.3m altitude, 0.2 m/s velocity): Faster startup but potentially more initial state variation

## Benefits

1. **Consistent Initial States**: Every episode starts with the drone at approximately the same position and velocity
2. **Improved Training Stability**: Reduced variance in initial conditions leads to more consistent reward signals
3. **Better Policy Convergence**: The policy learns from a more uniform experience distribution
4. **Reduced Simulation Artifacts**: Filters out Isaac Sim/PX4 initialization transients

## Integration with Existing Guards

This check works in conjunction with the existing **vertical hold guard** mechanism:

- **RL Starting Position Check** (new): Ensures consistent *episode initialization*
- **Vertical Hold Guard** (existing): Maintains altitude stability during *early RL steps*

Both mechanisms work together to provide robust altitude control during the critical takeoff-to-RL-control transition.

## Testing

After building, run training and observe the console output:
```bash
cd RL_with_cpp
./build.sh
./run_training.sh
```

Look for the `[RL-READY-CHECK]` log messages to confirm the verification is working.

## Troubleshooting

### Verification keeps timing out
- Check if PX4 altitude controller is properly tuned
- Verify Isaac Sim physics are stable
- Try loosening thresholds temporarily to diagnose

### Episodes start too slowly
- Reduce `stability_duration` from 1.0s to 0.5s
- Slightly loosen `altitude_tolerance` or `velocity_threshold`

### Large velocity spikes at episode start
- Check that the additional 1-second hold is executing
- Verify position setpoint commands are being published continuously
