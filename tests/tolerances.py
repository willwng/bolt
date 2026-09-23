""" Validation tolerances for Bolt vs the OpenSim oracle

Bolt runs in float32, so these are looser than OpenSim-vs-OpenSim comparisons.
"""

from __future__ import annotations

POSITION_M = 1e-5            # body-origin positions (forward kinematics)
ROTATION = 1e-5              # body rotation matrix entries
VELOCITY = 1e-4              # body spatial velocities (rad/s, m/s)

# Accelerations can be large, so use a relative tolerance too
ACCELERATION_ATOL = 1e-3   # body spatial accelerations, qacc
ACCELERATION_RTOL = 5e-4   # measured float32 error is up to ~1.5e-4 of the largest acceleration

MUSCLE_LENGTH_M = 1e-5       # muscle-tendon length
MUSCLE_SPEED_MS = 1e-4       # muscle lengthening speed
MUSCLE_MOMENT_ARM_M = 1e-5   # moment arm (-d length / d q)
