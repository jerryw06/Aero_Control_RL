# Stable Episode Starting Conditions

## Overview
This document explains the improvements made to ensure each RL training episode starts with **identical, stable conditions** for consistent and reproducible training.

## Problem
Previously, episodes could start while the drone was still settling after reset/teleport, leading to:
- Variable starting velocities
- Position drift from spawn point
- Inconsistent initial observations
- Reduced training stability

## Solution
Enhanced the `wait_for_position_settle()` function in `px4_accel_env.cpp` to guarantee:

### 1. **Position Stability**
- Checks position drift between consecutive measurements
- Requires drift < 0.05 m (5 cm) to proceed

### 2. **Velocity Stability** (NEW)
- Monitors velocity magnitude from VehicleLocalPosition messages
- Requires velocity < 0.1 m/s to proceed
- Ensures drone is truly hovering, not just slowly drifting

### 3. **Consecutive Stability Checks** (NEW)
- Requires **3 consecutive checks** (0.3 seconds) meeting both criteria
- Prevents false positives from transient measurements
- Guarantees sustained stability before episode start

### 4. **Increased Timeout**
- Increased from 2.0s to 3.0s to allow adequate stabilization time
- Prevents premature episode starts

## Technical Details

### Stability Criteria
```cpp
// Both must be true for 3 consecutive checks (0.3s):
drift < 0.05f        // Position drift less than 5 cm
vel_mag < 0.1f       // Velocity magnitude less than 0.1 m/s
```

### Reset Sequence (7 Steps)
1. **Land**: Ensure drone is on ground
2. **Disarm**: Disable motors
3. **Teleport**: Move to spawn position (via Isaac Sim)
4. **Physics Reset**: Reset Isaac Sim physics state
5. **Settle**: **Wait for position AND velocity stability** ← Enhanced
6. **Origin Tracking**: Reset PX4 local origin
7. **Takeoff**: Arm and takeoff to hover altitude

### Output (Reduced Verbosity)
```
[RESET] Step 5: Waiting for position and velocity to stabilize...
[RESET] Stable (drift=0.02m, vel=0.05 m/s)
[RESET] Position offset from spawn: 0.03 m
```

## Benefits

### For Training
- **Reproducible episodes**: Same starting state every time
- **Fair comparisons**: Policy evaluated from identical conditions
- **Faster convergence**: Reduced variance in initial observations

### For Debugging
- Easier to identify issues (consistent baseline)
- Reduced noise in reward signals
- Clearer learning curves

## Verification
To verify stable starts:
1. Run training: `./run_training.sh --isaac-reset`
2. Check logs for `[RESET] Stable` messages
3. Observe consistent drift/velocity values across episodes
4. Monitor episode returns - reduced variance indicates success

## Related Files
- `RL_with_cpp/src/px4_accel_env.cpp`: Implementation
- `RL_with_cpp/run_training.sh`: Training launcher (filters warnings)
- `RESET_MECHANISM_EXPLAINED.md`: Complete reset mechanism documentation

## Additional Improvements
- Suppressed CycloneDDS type hash warnings via `run_training.sh`
- Reduced debug output to essential messages only
- Clean training logs focused on episode data

---
**Last Updated**: 2025-01-07
**Implementation**: px4_accel_env.cpp, lines 1315-1376
