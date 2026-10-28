__all__ = [
    "MAX_POLY_NUM_DOFS",
    "MAX_POLY_ORDER",
    "BOLT_MINVAL",
    "BOLT_MAXVAL",
    "BOLT_SIG_REAL",
    "INITIAL_STEP_SIZE",
    "COND_PATH_POINT_RANGE_TOL",
    "IDX_SCRATCH_ROT_F",
    "IDX_SCRATCH_ROT_DF",
    "IDX_SCRATCH_ROT_D2F",
    "IDX_SCRATCH_TRANS_F",
    "IDX_SCRATCH_TRANS_DF",
    "IDX_SCRATCH_TRANS_D2F",
    "MIN_NORM_FIBER_LENGTH",
    "MAX_NORM_FIBER_LENGTH",
    "M_MIN_NORM_TENDON_FORCE",
    "M_MAX_NORM_TENDON_FORCE",
    "M_MIN_PENNATION_ANGLE",
    "M_MAX_PENNATION_ANGLE",
    "MILLARD_MIN_NORM_ACTIVE_FIBER_LENGTH",
    "WRAP_SMOOTHNESS_TOL",
    "WRAP_MAX_ITERATIONS",
    "WRAP_GEODESIC_ACCURACY",
    "WRAP_MAX_STEP_ANGLE",
    "WRAP_PROJECTION_TOL",
    "WRAP_NEAREST_POINT_TOL",
    "WRAP_NEAREST_POINT_MAX_ITERATIONS",
    "WRAP_MAX_GEODESIC_STEPS",
    "WRAP_INITIAL_STEP_SIZE",
]


# Function-based fitting for muscle paths
MAX_POLY_NUM_DOFS = 12  # Maximum number of DOFs a function can support
MAX_POLY_ORDER = 10  # Maximum polynomial order

# Numerical constants
BOLT_MINVAL = 1e-15
BOLT_MAXVAL = 10000000000.0
BOLT_SIG_REAL = 1e-6

# Adaptive step size a world starts with, after allocation or a reset
INITIAL_STEP_SIZE = 1.0 / 100.0

# OpenSim padding for a ConditionalPathPoint's coordinate range
COND_PATH_POINT_RANGE_TOL = 1e-5

# Index helpers for custom joints
IDX_SCRATCH_ROT_F = 0
IDX_SCRATCH_ROT_DF = 1
IDX_SCRATCH_ROT_D2F = 2
IDX_SCRATCH_TRANS_F = 3
IDX_SCRATCH_TRANS_DF = 4
IDX_SCRATCH_TRANS_D2F = 5

# These are overridden for each muscle depending on max pennation angles
MIN_NORM_FIBER_LENGTH = 0.2
MAX_NORM_FIBER_LENGTH = 1.8

# Muscle properties (todo: move to metadata)
M_MIN_NORM_TENDON_FORCE = 0.0
M_MAX_NORM_TENDON_FORCE = 5.0
M_MIN_PENNATION_ANGLE = 0.0
M_MAX_PENNATION_ANGLE = 1.47062891

# MILLARD only: minimum active fiber length
MILLARD_MIN_NORM_ACTIVE_FIBER_LENGTH = 0.4441

# Muscle wrapping (Scholz2015GeometryPath), defaults for the Option fields
WRAP_SMOOTHNESS_TOL = 1e-5  # rad
WRAP_MAX_ITERATIONS = 50
WRAP_GEODESIC_ACCURACY = 1e-6
# Muscle wrapping solver internals (Simbody's CableSpan, relaxed for float32)
WRAP_MAX_STEP_ANGLE = 10.0 / 180.0 * 3.141592653589793  # max angular correction per solver iteration
WRAP_PROJECTION_TOL = 1e-5  # implicit surface value tolerance when projecting the geodesic onto the surface
WRAP_NEAREST_POINT_TOL = 1e-6  # touchdown: convergence of the point on the line nearest the surface
WRAP_NEAREST_POINT_MAX_ITERATIONS = 50
WRAP_MAX_GEODESIC_STEPS = 200  # integration steps per geodesic over an implicit surface
WRAP_INITIAL_STEP_SIZE = 1e-4  # first integration step size for a new geodesic
