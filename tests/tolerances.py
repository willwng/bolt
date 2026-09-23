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

BODY_FORCE_ATOL = 1e-5          # standalone body forces (e.g. drag), N and N*m
BODY_FORCE_RTOL = 1e-4

METABOLIC_POWER_ATOL = 1e-4     # W
METABOLIC_POWER_RTOL = 1e-4

# Integrator tests
INTEGRATOR_ACTIVATION = {"EULER_FIXED": 5e-2, "RK4_FIXED": 5e-5, "EULER_ADAPTIVE": 1e-2, "RK_MERSON_ADAPTIVE": 5e-6}
INTEGRATOR_QPOS = {"EULER_FIXED": 1e-1, "RK4_FIXED": 1e-2, "EULER_ADAPTIVE": 1e-2}  # vs RK-Merson, m or rad
INTEGRATOR_QVEL = {"EULER_FIXED": 10.0, "RK4_FIXED": 5.0, "EULER_ADAPTIVE": 0.5}  # vs RK-Merson, m/s or rad/s
