# Function-based fitting for muscle paths
MAX_POLY_NUM_DOFS = 12  # Maximum number of DOFs a function can support
MAX_POLY_ORDER = 10  # Maximum polynomial order

# Numerical constants
BOLT_MINVAL = 1e-15
BOLT_MAXVAL = 10000000000.0
BOLT_SIG_REAL = 1e-6

# Index helpers for custom joints
IDX_SCRATCH_ROT_F = 0
IDX_SCRATCH_ROT_DF = 1
IDX_SCRATCH_ROT_D2F = 2
IDX_SCRATCH_TRANS_F = 3
IDX_SCRATCH_TRANS_DF = 4
IDX_SCRATCH_TRANS_D2F = 5

# Air drag
A_Cd = 0.9
A_Af = 0.50641133
A_rho = 1.20474061
A_AFK = 0.5 * A_rho * A_Af * A_Cd

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
