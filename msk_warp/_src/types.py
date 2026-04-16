import enum
from dataclasses import dataclass
from . import consts
import warp as wp


@dataclass
class TileBlockDim:
    """Block dimension 'block_dim' settings for wp.launch_tiled. """

    # Variable-step integration
    adjust_scales: int = 16
    error_step: int = 16
    restore_state: int = 16

    # function-based muscle paths
    muscle_path: int = 32


class MobilizerType(enum.IntEnum):
    """Type of mobilizer

    Attributes:
      WORLD: dummy joint for ground                                   (0,)
      WELD: no dofs                                                   (0,)
      FREE:  global position and orientation (quat)                   (7,)
      PIN: rotation angle (rad) around joint z-axis                   (1,)
      SLIDER: sliding distance along body-fixed axis                  (1,)
      UNIVERSAL: two rotation angles (rad) around joint x- and y-axes (2,)
      GIMBAL: three euler angles (XYZ order)                          (3,)
      BEAM: Cantilever Free Beam bending model                        (3,)
      ELLIPSOID: Ellipsoid joint                                      (3,)
      BALL:  orientation (quat) relative to parent                    (4,)
      CUSTOM: custom joint with up to 6 dofs                          (<=6,)
    """

    WORLD = 0
    WELD = 1
    FREE = 2
    PIN = 3
    SLIDER = 4
    UNIVERSAL = 5
    GIMBAL = 6
    BEAM = 7
    ELLIPSOID = 8
    BALL = 9
    CUSTOM = 10


class GeomType(enum.IntEnum):
    """Type of geometry.

    Attributes:
      PLANE: plane
      HFIELD: heightfield
      SPHERE: sphere
      CAPSULE: capsule
      ELLIPSOID: ellipsoid
      CYLINDER: cylinder
      BOX: box
      MESH: mesh
    """

    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7


class vec5(wp.types.vector(length=5, dtype=float)):
    pass


class vec6(wp.types.vector(length=6, dtype=float)):
    pass


class mat34(wp.types.matrix(shape=(3, 4), dtype=float)):
    pass


class mat36(wp.types.matrix(shape=(3, 6), dtype=float)):
    pass


class mat43(wp.types.matrix(shape=(4, 3), dtype=float)):
    pass


class mat411(wp.types.matrix(shape=(4, 11), dtype=float)):
    pass


# Used for function-based fitting for muscle paths
class PolyVec(wp.types.vector(length=consts.MAX_POLY_NUM_DOFS, dtype=float)):
    pass


class PolyInts(wp.types.vector(length=consts.MAX_POLY_NUM_DOFS, dtype=int)):
    pass


class PolyEval(wp.types.vector(length=consts.MAX_POLY_NUM_DOFS + 1, dtype=float)):
    pass


class PolyPowCache(wp.types.matrix(shape=(consts.MAX_POLY_NUM_DOFS, consts.MAX_POLY_ORDER + 1), dtype=float)):
    pass


@wp.struct
class SpatialInertia:
    m: float  # mass
    p: wp.vec3  # mass center
    G: wp.mat33  # unit inertia measured in the body frame (not about COM frame)


@wp.struct
class ArticulatedInertia:
    M: wp.mat33  # mass
    J: wp.mat33  # inertia
    F: wp.mat33  # mass moment


@wp.struct
class CoordinateLimitForce:
    qpos_range: wp.vec2
    qpos_adr: int
    dof_adr: int
    stiffness: wp.vec2
    damping: float
    transition: float


@wp.struct
class SwingTwistLimit:
    qpos_adr: int
    dof_adr: int

    twist_range: wp.vec2
    swing1_range: wp.vec2
    swing2_range: wp.vec2
    stiffness: float
    damping: float


@wp.struct
class ExponentialContact:
    contact_plane_transform: wp.transform
    shape_parameters: wp.vec3
    normal_viscosity: float
    max_normal_force: float
    friction_elasticity: float
    friction_viscosity: float
    settle_velocity: float
    initial_mu_static: float
    initial_mu_kinetic: float

    siteid: int
    bodyid: int
    station_B: wp.vec3


def array(*args) -> wp.array:
    """A wrapper around wp.array that adds extra metadata to ease type introspection.

    Format is array(dim_1, dim_2, ..., dtype).  dim may be a constant int, or reference a size from
    Model or Data (e.g. "nq" or "nworld").  dim may also be "*", which means any nonzero size.
    """
    shape, dtype = args[:-1], args[-1]

    arr = wp.array(ndim=len(shape), dtype=dtype)
    arr.shape = shape

    return arr


class ActivationType(enum.IntEnum):
    """ Muscle activation dynamics
    Attributes:
        DGF: DeGroote-Fregly muscle activation dynamics
        MILLARD: Millard muscle activation dynamics
    """
    DGF = 1
    MILLARD = 2


class ContractionType(enum.IntEnum):
    """ Muscle contraction dynamics
    Attributes:
        DGF: DeGroote-Fregly muscle contraction dynamics
        MILLARD: Millard muscle contraction dynamics (splines)
    """
    DGF = 1
    MILLARD = 2


class IntegratorType(enum.IntEnum):
    """ Integrator type.
    Attributes:
        EULER_FIXED: Fixed-step Euler (semi-implicit)
        EULER_MIDPOINT_FIXED: Fixed-step Midpoint Euler method
        RK4_FIXED: Fixed-step 4th-order Runge-Kutta
        EULER_ADAPTIVE: Adaptive-step Euler
        EULER_MIDPOINT_ADAPTIVE: Adaptive-step Midpoint Euler method
        RK_MERSON_ADAPTIVE: Adaptive-step 4th-order Runge-Kutta-Merson
    """
    EULER_FIXED = 1
    EULER_MIDPOINT_FIXED = 2
    RK4_FIXED = 3
    EULER_ADAPTIVE = 4
    EULER_MIDPOINT_ADAPTIVE = 5
    RK_MERSON_ADAPTIVE = 6


@dataclass
class MetabolicOptions:
    """
    Settings for muscle metabolic energy calculations
    """
    activation_maintenance_rate_on: bool
    shortening_rate_on: bool
    mechanical_work_rate_on: bool
    enforce_minimum_heat_rate: bool

    aerobic_factor: float
    muscle_effort_scaling_factor: float
    use_bhargava_recruitment: bool
    include_negative_mechanical_work: bool
    forbid_negative_total_power: bool


@dataclass
class Option:
    """

    Attributes:
      gravity: gravitational acceleration
      explicit_gravity: flag to compute gravity as an explicit force (or as fictitious acceleration)
      implicit_damping: flag to add h*damping to mass matrix for implicit damping
      enable_drag: flag to enable drag forces
      visuals: whether to handle visual geometry
      nbeam_visuals: number of beam visuals (for rendering beams joints)

      activation_type: muscle activation dynamics type (ActivationType)
      contraction_type: muscle contraction dynamics type (ContractionType)
      integrator: integrator type (IntegratorType)

      metabolic_options: options for muscle metabolic energy calculations (MetabolicOptions)

      safety: (variable-step integration) safety factor
      min_shrink: (variable-step integration) minimum step shrink factor
      max_grow: (variable-step integration) maximum step grow factor
      min_step_size: (variable-step integration) minimum step size
      max_step_size: (variable-step integration) maximum step size
      hysteresis_low: (variable-step integration) error hysteresis lower bound
      hysteresis_high: (variable-step integration) error hysteresis upper bound
      accuracy: (variable-step integration) target accuracy
      use_inf_norm: (variable-step integration) whether to use infinity norm for error calculation
      qvel_weights: (variable-step integration) weights for qvel error calculation
      z_weights: (variable-step integration) weights for additional state error calculation
    """

    gravity: float
    explicit_gravity: bool
    implicit_damping: bool
    enable_drag: bool
    visuals: bool
    nbeam_visuals: int

    activation_type: ActivationType
    contraction_type: ContractionType
    integrator: IntegratorType

    metabolic_options: MetabolicOptions

    # Variable-step integration options
    safety: float
    min_shrink: float
    max_grow: float
    min_step_size: float
    max_step_size: float
    hysteresis_low: float
    hysteresis_high: float
    accuracy: float
    use_inf_norm: bool
    qvel_weights: array("nv", float)
    z_weights: array("nz", float)


@wp.struct
class ResidualResult:
    """
    Residual result from solving for muscle equilibrium.
    """
    norm_tendon_force: float
    residual: float

    pennation_angle: float
    fiber_length: float
    norm_fiber_length: float
    tendon_length: float
    norm_tendon_length: float
    norm_tendon_velocity: float
    active_fiber_force: float
    fiber_velocity: float
    fiber_force_along_tendon: float


@wp.struct
class MuscleMetadata:
    """Muscle metadata. """
    ignore_tendon_compliance: bool

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    optimal_pennation_angle: float
    fiber_damping: float
    v_max: float

    # Passive force-length curve
    strain_at_zero_force: float
    strain_at_one_norm_force: float
    stiffness_at_low_force: float
    stiffness_at_one_norm_force: float
    curviness: float

    activation_time_const: float
    deactivation_time_const: float
    activation_dynamics_smoothing: float

    min_norm_fiber_length: float
    max_norm_fiber_length: float
    min_activation: float
    max_activation: float

    # Additional parameters for metabolic calculations
    specific_tension: float
    density: float
    slow_twitch_ratio: float


@wp.struct
class ActuatorMetadata:
    """Muscle metadata. """
    optimal_force: float
    activation_time_constant: float
    coordinate: int
    default_activation: float

    min_activation: float
    max_activation: float


@wp.struct
class MuscleLengthInfo:
    """ Dynamic length info for muscle length calculation """
    fiber_length: float
    fiber_length_along_tendon: float
    norm_fiber_length: float
    tendon_length: float
    norm_tendon_length: float
    tendon_strain: float
    pennation_angle: float
    cos_pennation_angle: float
    sin_pennation_angle: float

    fiber_passive_force_length_multiplier: float
    fiber_active_force_length_multiplier: float
    tendon_force_multiplier: float
    fiber_state_clamped: bool


@wp.struct
class FiberVelocityInfo:
    """ Dynamic velocity info for muscle velocity calculation """
    fiber_velocity: float
    fiber_velocity_along_tendon: float
    norm_fiber_velocity: float
    pennation_angular_velocity: float
    tendon_velocity: float
    norm_tendon_velocity: float
    fiber_force_velocity_multiplier: float
    fiber_damping_force_multiplier: float


@wp.struct
class MuscleDynamicsInfo:
    """ Dynamic info for muscle force calculation """
    fiber_force: float
    fiber_force_along_tendon: float
    norm_fiber_force: float
    active_fiber_force: float
    passive_fiber_force: float
    tendon_force: float
    norm_tendon_force: float


@dataclass
class MeshLoadResult:
    """ Result of loading a mesh from file """
    file: str
    scale: tuple[float, float, float]


@dataclass
class Model:
    """Model definition and parameters.

    Attributes:
      nbody: number of bodies
      nq: number of generalized coordinates
      nv: number of generalized speeds
      nmuscle: number of muscles
      nactuator: number of "ideal" actuators
      nz: number of additional state variables
      njnts_cst: number of custom joints
      nbeams: number of beam joints
      ngeom: number of collision geometry
      nexpcontact: number of exponential contact forces
      nvis: number of visual geometry
      nsite: number of sites
      nlimitforce: number of (CoordinateLimitForce) limits
      nswingtwist: number of swing-twist limits

      nfunctions: number of custom functions
      nlinearfn: number of linear functions
      nconstfn: number of constant functions
      npolyfn: number of polynomial functions

      opt: physics options
      muscle_metadata: muscle metadata                         (nmuscle,)
      muscle_data: same as above, but intended for future modification

      actuator_metadata: actuator metadata                     (nactuator,)
      actuator_data: same as above, but intended for future modification

      body_mass: mass                                          (nbody,)
      body_unit_inertia_OB_B: inertia about B body frame       (nbody, mat33)
      body_mass_center: local transform of center of mass      (nbody, transform)

      body_parentid: id of body's parent                       (nbody,)
      body_tree: list of body ids by tree level
      body_children: list of body ids of each body's children
      body_children_adr: start adr in 'body_children'          (nbody,)
      body_children_num: number of children for each body      (nbody,)

      mob_type: type of joint's mobilizer (MobilizerType)      (nbody,)
      mob_qposadr: start adr in qpos for joint's data          (nbody,)
      mob_dofnum: number of dofs for each joint                (nbody,)
      mob_dofadr: start adr in qvel for joint's data           (nbody,)
      mob_X_PF: parent -> parent joint frame                   (nbody, transform)
      mob_X_MB: mobilizer -> child                             (nbody, transform)
      mob_extra_info: extra info for each mobilizer            (nbody, vec3)

     * custom functions *
      linear_fn_mb: slope, intercept                           (nlinearfn, vec2)
      const_fn_c: constant value                               (nconstfn,)
      poly_fn_coeff: polynomial coefficients                   (total number of coefficients)
      poly_fn_coeff_adr: starting adr in coeffs                (npolyfn,)
      poly_fn_coeff_num: num coefficients for each poly fn     (npolyfn,)
      linear_fn_adr: "global" fn address for each linear fn    (nlinearfn,)
      const_fn_adr: "global" fn address for each const fn      (nconstfn,)
      poly_fn_adr: "global" fn address for each polynomial fn  (npolyfn,)
      linear_fn_qpos_adr: qpos address for linear fn input     (nlinearfn,)
      poly_fn_qpos_adr: qpos address for polynomial fn input   (npolyfn,)
     
     * custom joints *
      mob_to_cst_id: map mobilizer idx -> custom joint idx     (nbody,)
      cst_to_mob_id: map custom joint idx -> mobilizer idx     (njnts_cst,)
      cst_txfm_axes: custom transform axes (3 rot, 3 trans)    (njnts_cst, 6, vec3)
      cst_txfm_dof: dof idx offset (FROM JOINT) for each txfm  (njnts_cst, 6)

     * beam joints *
      beam_to_mob_id: map beam idx -> mobilizer idx             (nbeam,)

     * stiffness/damping *
      dof_damping: damping coefficient                         (nv)
      dof_armature: armature added to joint-space inertia diag (nv,)
      dof_stiffness: stiffness coefficient                     (nv)
      qpos_spring_rest: rest position for dof spring           (nq,)

     * limits *
      coordinate_limit_force: coordinate limit forces          (nlimitforce, CoordinateLimitForce)
      swing_twist_limit: swing-twist limits for quaternions    (nswingtwist, SwingTwistLimit)

     * collision geometry *
      geom_type: geometric type (GeomType)                     (ngeom,)
      geom_bodyid: id of geom's body                           (ngeom,)
      geom_X_loc: local transform of geom rel. to body         (ngeom, transform)
      geom_size: geom-specific size parameters                 (ngeom, 3)
      geom_friction: friction for (slide, spin, roll)          (ngeom, 3)
      geom_stiffness: contact stiffness (Hunt-Crossley)        (ngeom,)
      geom_dissipation: contact dissipation (Hunt-Crossley)    (ngeom,)
      geom_transition_velocity: friction transition velocity   (ngeom,)
      geom_priority: collision priority (Hunt-Crossley)        (ngeom,)
      geom_aabb: axis-aligned bounding box (center, size)      (ngeom, 2, 3)
      geom_rbound: bounding sphere radius                      (ngeom,)

      exp_contact: exponential contact force parameters        (nexpcontact, ExponentialContact)

     * colliders *
      geom_pair_type_count: count of max number of each potential collision
      nxn_geom_pair_filtered: valid collision pair geom ids    (<=ngeom*(ngeom-1)/2,)
      nxn_pairid_filtered: active subset of nxn_pairid         (<=ngeom*(ngeom-1)/2, 2)

     * visuals *
      vis_bodyid: id of visual geometry's body                 (nvis,)
      vis_X_loc: local transform of visual rel. to body        (nvis, transform)

     * sites *
      site_bodyid: id of site's body                           (nsite,)
      site_offset: local position offset rel. to body          (nsite, 3)

     * muscles *
      muscle_pts_adr: address of first point in muscle's path  (nmuscle,)
      muscle_pts_num: number of points in muscle's path        (nmuscle,)

     * muscle paths *
      muscle_pt_group: muscle ids for point paths
      muscle_fn_groups: muscle ids for function paths, grouped by same (dim, order)

     * muscle function-based paths *
      fn_path_qpos_adr: qpos adr for each muscle fn path term  (nmuscle, PolyInt)
      fn_path_dimension: number of dependent variables         (nmuscle,)
      fn_path_order: order of polynomial function              (nmuscle,)
      fn_path_term_start: starting adr of each muscle's terms  (nmuscle,)
      fn_path_term_coeffs: coefficients for each fn path term  (nmuscle, num_fn_terms)
    """

    nbody: int
    nq: int
    nv: int
    nmuscle: int
    nactuator: int
    nz: int
    njnts_cst: int
    nbeams: int
    ngeom: int
    nexpcontact: int
    nvis: int
    nsite: int
    nlimitforce: int
    nswingtwist: int

    nfunctions: int
    nlinearfn: int
    nconstfn: int
    npolyfn: int

    opt: Option
    muscle_metadata: array("nmuscle", MuscleMetadata)
    muscle_data: list[MuscleMetadata]

    actuator_metadata: array("nactuator", ActuatorMetadata)
    actuator_data: list[ActuatorMetadata]

    body_mass: array("nbody", float)
    body_unit_inertia_OB_B: array("nbody", wp.mat33)
    body_mass_center: array("nbody", wp.vec3)

    body_parentid: array("nbody", int)
    body_tree: tuple[wp.array(dtype=int), ...]
    body_children: wp.array(dtype=int)
    body_children_adr: wp.array(dtype=int)
    body_children_num: wp.array(dtype=int)

    mob_type: array("nbody", int)
    mob_qposadr: array("nbody", int)
    mob_dofadr: array("nbody", int)
    mob_dofnum: array("nbody", int)
    mob_X_PF: array("nbody", wp.transform)
    mob_X_MB: array("nbody", wp.transform)
    mob_extra_info: array("nbody", wp.vec3)

    mob_to_cst_id: array("nbody", int)
    cst_to_mob_id: array("njnts_cst", int)
    cst_txfm_axes: array("njnts_cst", 6, wp.vec3)
    cst_txfm_dof: array("njnts_cst", 6, int)

    beam_to_mob_id: array("nbeam", int)

    linear_fn_mb: array("nlinearfn", wp.vec2)
    const_fn_c: array("nconstfn", float)
    poly_fn_coeff: array("total_coeff", float)
    poly_fn_coeff_adr: array("npolyfn", int)
    poly_fn_coeff_num: array("npolyfn", int)
    linear_fn_adr: array("nlinearfn", int)
    const_fn_adr: array("nconstfn", int)
    poly_fn_adr: array("npolyfn", int)
    linear_fn_qpos_adr: array("nlinearfn", int)
    poly_fn_qpos_adr: array("npolyfn", int)

    dof_damping: array("nv", float)
    dof_armature: array("nv", float)
    dof_stiffness: array("nv", float)
    qpos_spring_rest: array("nq", float)

    coordinate_limit_force: array("nlimitforce", CoordinateLimitForce)
    swing_twist_limit: array("nswingtwist", SwingTwistLimit)

    # Collision geometry
    geom_type: array("ngeom", int)
    geom_bodyid: array("ngeom", int)
    geom_X_loc: array("ngeom", wp.transform)
    geom_size: array("ngeom", wp.vec3)
    geom_friction: array("ngeom", wp.vec3)
    geom_stiffness: array("ngeom", float)
    geom_dissipation: array("ngeom", float)
    geom_transition_velocity: array("ngeom", float)
    geom_priority: array("ngeom", int)
    geom_aabb: array("ngeom", 2, wp.vec3)
    geom_rbound: array("ngeom", float)

    exp_contact: array("nexpcontact", ExponentialContact)

    geom_pair_type_count: tuple[int, ...]
    nxn_geom_pair_filtered: array("<=ngeom*(ngeom-1)/2", wp.vec2i)
    nxn_pairid_filtered: array("<=ngeom*(ngeom-1)/2", wp.vec2i)

    # Visual geometry
    vis_bodyid: array("nvis", int)
    vis_X_loc: array("nvis", wp.transform)

    # Attachment sites (muscle path)
    site_bodyid: array("nsite", int)
    site_offset: array("nsite", wp.vec3)

    # Muscles
    muscle_pts_adr: wp.array(dtype=int)
    muscle_pts_num: wp.array(dtype=int)

    # Muscle paths
    muscle_pt_group: wp.array(dtype=int)
    muscle_fn_groups: tuple[wp.array(dtype=int), ...]

    # Polynomial/function paths
    fn_path_qpos_adr: array("nmuscle", PolyInts)
    fn_path_dimension: array("nmuscle", int)
    fn_path_order: array("nmuscle", int)
    fn_path_term_start: array("nmuscle", int)
    fn_path_term_coeffs: array("nmuscle", "num_fn_terms", float)

    block_dim: TileBlockDim


@dataclass
class Contact:
    """Contact data.

    Attributes:
      dist: distance between nearest points; neg: penetration          (naconmax,)
      pos: position of contact point: midpoint between geoms           (naconmax, 3)
      frame: normal is in [0-2], points from geom[0] to geom[1]        (naconmax, 3, 3)
      friction: tangent1, 2, spin, roll1, 2                            (naconmax, 5)
      dim: contact space dimensionality: 1, 3, 4 or 6                  (naconmax,)
      curvature: effective radius of curvature                         (naconmax,)
      stiffness: contact stiffness                                     (naconmax,)
      dissipation: contact dissipation                                 (naconmax,)
      transition_velocity: contact transition velocity                 (naconmax,)
      geom: geom ids; -1 for flex                                      (naconmax, 2)
      worldid: world id                                                (naconmax,)
    """

    dist: wp.array(dtype=float)
    pos: wp.array(dtype=wp.vec3)
    frame: wp.array(dtype=wp.mat33)
    friction: wp.array(dtype=vec5)
    dim: wp.array(dtype=int)
    curvature: wp.array(dtype=float)
    stiffness: wp.array(dtype=float)
    dissipation: wp.array(dtype=float)
    transition_velocity: wp.array(dtype=float)
    geom: wp.array(dtype=wp.vec2i)
    worldid: wp.array(dtype=int)


@dataclass
class IntegratorStateScratch:
    time: wp.array(dtype=float)
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    m_state: wp.array2d(dtype=float)
    m_act: wp.array2d(dtype=float)
    a_act: wp.array2d(dtype=float)
    exp_contact_state: wp.array2d(dtype=wp.vec4)


@dataclass
class IntegratorDotScratch:
    qvel: wp.array2d(dtype=float)
    qacc: wp.array2d(dtype=float)
    m_state_dot: wp.array2d(dtype=float)
    m_act_dot: wp.array2d(dtype=float)
    a_act_dot: wp.array2d(dtype=float)
    exp_contact_state_dot: wp.array2d(dtype=wp.vec4)


@dataclass
class IntegratorMidpointScratch:
    qvel: wp.array2d(dtype=float)
    qacc: wp.array2d(dtype=float)


@dataclass
class Data:
    """Dynamic state that updates each step.

    Attributes:
      nworld: number of parallel worlds being simulated
      naconmax: maximum number of contacts total
      rng_state: random number generator state (if needed)

      world_reset: whether the world needs to be reset            (nworld,)
      next_time: final target time for integrator (tMax)          (nworld,)

      * Current state *
      time: simulation time                                       (nworld,)
      qpos: generalized position                                  (nworld, nq)
      qvel: generalized speeds                                    (nworld, nv)
      m_state: muscle state variable                              (nworld, nmuscles)
      m_act: muscle activation                                    (nworld, nmuscles)
      a_act: actuator activation                                  (nworld, nactuator)
      exp_contact_state: state variable for exponential contact   (nworld, nexpcontact, 4)

     * current controls *
      m_excitations: muscle excitations                           (nworld, nmuscles)
      a_excitations: actuator excitations                         (nworld, nactuator)

     * State derivatives *
      qdot: derived from qvel, i.e. qdot = N(q) @ qvel            (nworld, nq)
      qacc: acceleration                                          (nworld, nv)
      m_state_dot: time-derivative of muscle state variable       (nworld, nmuscles)
      m_act_dot: time-derivative of actuator activation           (nworld, na)
      a_act_dot: time-derivative of actuator activation           (nworld, nactuator)
      exp_contact_state_dot: derivative info of exp contact state (nworld, nexpcontact, 4)

     *
        simulator forces.
        body_F_ are Cartesian forces applied to bodies.
        ufrc_ are generalized forces (in generalized speed space)
        qfrc_ are generalized forces (in generalized position space, need to be converted)
     *
      body_F_gravity: gravity Cartesian force/torque on body      (nworld, nbody, 6)
      body_F_contact: contact Cartesian force/torque on body      (nworld, nbody, 6)
      body_F_muscle: muscle Cartesian force/torque on body        (nworld, nbody, 6)
      body_F_drag: drag Cartesian force/torque on body            (nworld, nbody, 6)
      body_F: net Cartesian force/torque on body                  (nworld, nbody, 6)
      qfrc_muscle: muscle generalized force in qpos space         (nworld, nq)
      qfrc_muscle_passive: only passive component of muscle force (nworld, nq)
      ufrc_spring: passive spring force                           (nworld, nv)
      ufrc_damper: passive damper force                           (nworld, nv)
      ufrc_muscle: muscle generalized force                       (nworld, nv)
      ufrc_muscle_passive: only passive component of muscle force (nworld, nv)
      ufrc_actuator: actuator generalized force                   (nworld, nv)
      ufrc_limit: dof limit generalized force                     (nworld, nv)
      ufrc_total: net generalized force                           (nworld, nv)

     * user-facing forces *
      ufrc_applied: user-facing applied generalized force         (nworld, nv)
      body_F_applied: applied Cartesian force/torque              (nworld, nbody, 6)

     * post-dynamics analytics *
      grf: ground reaction force                                  (nworld, 3)
      geom_cforce: contact force on geoms                         (nworld, ngeom)
      geom_self_cforce: normal force on geoms, excl. ground ctc   (nworld, ngeom)
      body_self_cforce: normal force on bodies, excl. ground      (nworld, nbody)
      joint_moments: net moment                                   (nworld, nv)

      contact: contact data

     * custom joints/functions *
      cst_fn_output: f, f', f'' output of custom joint functions (nworld, nfunction, vec3)

     * mobilizers *
      mob_X_GB: Cartesian position of body frame                  (nworld, nbody, 3)
      mob_X_FM: Mobilizer transformation                          (nworld, nbody, transform)
      mob_X_PB: Transform from parent to body frame               (nworld, nbody, transform)
      mob_scratch: scratch space for mobilizer calculations       (nworld, nbody, vec3)
      mob_phi: parent-to-child shift (Bo-Po) in the ground frame  (nworld, nbody, 3)
      mob_H_FM: cross joint jacobian of each dof (rot:lin)        (nworld, nv, 6)
      mob_H: cross joint jacobian of each dof (in ground)         (nworld, nv, 6)
      mob_HDot_FM: time-derivative of cross joint jacobian        (nworld, nv, 6)
      mob_HDot: time-derivative of cross joint jacobian in ground (nworld, nv, 6)
      mob_DI: DI = inverse(~H @ P @ H)                            (nworld, nv, 6) # ndof x ndof, won't use all 6
      mob_G: G = PH * DI                                          (nworld, nv, nv)
      mob_coriolis_acc: Coriolis/centrifugal acceleration         (nworld, nbody, 6)

      body_COM_G: Position of body com relative to ground         (nworld, nbody, 3)
      body_Mk_G: Spatial inertia in ground frame                  (nworld, nbody, SpatialInertia)
      body_P: Articulated inertia mass matrix in ground frame     (nworld, nbody, ArticulatedInertia)
      body_PPlus: Articulated inertia (including children)        (nworld, nbody, ArticulatedInertia)
      body_V_FM: spatial velocity of mobilizer                    (nworld, nbody, 6)
      body_V_PB_G: spatial velocity of parent to body in ground   (nworld, nbody, 6)
      body_V_GB: spatial velocity of body in ground frame         (nworld, nbody, 6)
      body_VD_PB_G: spatial acc of parent to body in ground       (nworld, nbody, 6)
      body_A_GB: spatial acceleration of body in ground frame     (nworld, nbody, 6)
      body_gyro_force: gyroscopic force on body                   (nworld, nbody, 6)
      body_total_coriolis_acc: total coriolis acc                 (nworld, nbody, 6)
      body_total_centrifugal_force: total centrifugal force       (nworld, nbody, 6)
      body_articulated_centrifugal_force:                         (nworld, nbody, 6)
      body_zPlus: z = Pa + b - F, zPlus includes children         (nworld, nbody, 6)
      body_zTmp: similar to zPlus, used for J^T calculations      (nworld, nbody, 6)
      body_eps: f - ~H * z                                        (nworld, nbody, 6)

      geom_X: Cartesian geom transform                            (nworld, ngeom, transform)
      vis_X: Cartesian visual transform                           (nworld, nvis, transform)
      vis_beam_pos: position of beam visuals                      (nworld, nbeams, nbeam_visuals, 3)

     * contacts *
      collision_pair: pair of geoms in contact                    (nacon, 2)
      collision_pairid: pair of geom ids in contact               (nacon, 2)
      collision_worldid: world id of contact                      (nacon,)
      ncollision: number of detected collisions across all worlds (1,)
      nacon: numbet of collisions per world                       (nworld,)
      contact: contact data

     * muscle paths
      muscle_length: muscle lengths                               (nworld, nmuscle)
      muscle_velocity: muscle velocities                          (nworld, nmuscle)

     * point-path based muscle paths
      site_rel_pos_B: site position relative to body              (nworld, nsite, 3)
      site_pos_G: site position measured in ground                (nworld, nsite, 3)
      site_vel_G: site velocity measured in ground                (nworld, nsite, 3)

     * function-based muscle paths
      muscle_moment_arm: moment arm of muscle r = dL/dq           (nworld, nmuscle, nq)

     * muscle dynamics
      muscle_length_info: info for muscle length calculation      (nworld, nmuscle)
      muscle_velocity_info: info for muscle velocity calculation  (nworld, nmuscle)
      muscle_dynamics_info: info for muscle force calculation     (nworld, nmuscle)
      muscle_norm_fiber_length: norm fiber lengths (obs only)     (nworld, nmuscle)
      muscle_actuation: muscle actuation forces                   (nworld, nmuscle)
      muscle_actuation_passive: passive component of actuation    (nworld, nmuscle)
      muscle_metabolic: muscle metabolic energy rate              (nworld, nmuscle)


    warp only fields:
      nworld: number of worlds
      naconmax: maximum number of contacts (shared across all worlds)
      njmax: maximum number of constraints per world
      nacon: number of detected contacts (across all worlds)
      nsolving: number of unconverged worlds                      (1,)
      subtree_bodyvel: subtree body velocity (ang, vel)           (nworld, nbody, 6)
    """
    nworld: int
    naconmax: int
    rng_state: array("1", int)

    world_reset: array("nworld", bool)
    next_time: array("nworld", float)

    time: array("nworld", float)
    qpos: array("nworld", "nq", float)
    qvel: array("nworld", "nv", float)
    m_state: array("nworld", "nmuscle", float)
    m_act: array("nworld", "nmuscle", float)
    a_act: array("nworld", "nactuator", float)
    exp_contact_state: array("nworld", "nexpcontact", wp.vec4)

    m_excitations: array("nworld", "nmuscle", float)
    a_excitations: array("nworld", "nactuator", float)

    qdot: array("nworld", "nv", float)
    qacc: array("nworld", "nv", float)
    m_state_dot: array("nworld", "nmuscle", float)
    m_act_dot: array("nworld", "nmuscle", float)
    a_act_dot: array("nworld", "nactuator", float)
    exp_contact_state_dot: array("nworld", "nexpcontact", wp.vec4)

    body_F_gravity: array("nworld", "nbody", wp.spatial_vector)
    body_F_contact: array("nworld", "nbody", wp.spatial_vector)
    body_F_muscle: array("nworld", "nbody", wp.spatial_vector)
    body_F_drag: array("nworld", "nbody", wp.spatial_vector)
    body_F: array("nworld", "nbody", wp.spatial_vector)

    qfrc_muscle: wp.array2d(dtype=float)
    qfrc_muscle_passive: wp.array2d(dtype=float)

    ufrc_spring: wp.array2d(dtype=float)
    ufrc_damper: wp.array2d(dtype=float)
    ufrc_muscle: wp.array2d(dtype=float)
    ufrc_muscle_passive: wp.array2d(dtype=float)
    ufrc_actuator: wp.array2d(dtype=float)
    ufrc_limit: wp.array2d(dtype=float)
    ufrc_total: wp.array2d(dtype=float)

    ufrc_applied: array("nworld", "nv", float)
    body_F_applied: array("nworld", "nbody", wp.spatial_vector)

    grf: array("nworld", wp.vec3)
    geom_cforce: array("nworld", "ngeom", float)
    geom_self_cforce: array("nworld", "ngeom", float)
    body_self_cforce: array("nworld", "nbody", float)
    joint_moments: array("nworld", "nv", float)

    cst_fn_output: array("nworld", "nfunction", wp.vec3)

    mob_X_GB: wp.array2d(dtype=wp.transform)
    mob_X_FM: wp.array2d(dtype=wp.transform)
    mob_X_PB: wp.array2d(dtype=wp.transform)
    mob_scratch: wp.array3d(dtype=wp.vec3)  # used for storing precomputed values
    mob_phi: wp.array2d(dtype=wp.vec3)
    mob_H_FM: wp.array2d(dtype=wp.spatial_vector)
    mob_H: wp.array2d(dtype=wp.spatial_vector)
    mob_HDot_FM: wp.array2d(dtype=wp.spatial_vector)
    mob_HDot: wp.array2d(dtype=wp.spatial_vector)
    mob_DI: wp.array2d(dtype=wp.spatial_vector)  # ndof x ndof, so we won't use all 6 of spatial vector
    mob_G: wp.array2d(dtype=wp.spatial_vector)
    mob_coriolis_acc: wp.array2d(dtype=wp.spatial_vector)

    body_COM_G: wp.array2d(dtype=wp.vec3)
    body_Mk_G: wp.array2d(dtype=SpatialInertia)
    body_P: wp.array2d(dtype=ArticulatedInertia)
    body_PPlus: wp.array2d(dtype=ArticulatedInertia)
    body_V_FM: wp.array2d(dtype=wp.spatial_vector)
    body_V_PB_G: wp.array2d(dtype=wp.spatial_vector)
    body_V_GB: wp.array2d(dtype=wp.spatial_vector)
    body_VD_PB_G: wp.array2d(dtype=wp.spatial_vector)
    body_A_GB: wp.array2d(dtype=wp.spatial_vector)
    body_gyro_force: wp.array2d(dtype=wp.spatial_vector)
    body_total_coriolis_acc: wp.array2d(dtype=wp.spatial_vector)
    body_total_centrifugal_force: wp.array2d(dtype=wp.spatial_vector)
    body_articulated_centrifugal_force: wp.array2d(dtype=wp.spatial_vector)
    body_zPlus: wp.array2d(dtype=wp.spatial_vector)
    body_zTmp: wp.array2d(dtype=wp.spatial_vector)
    body_eps: wp.array2d(dtype=wp.spatial_vector)

    subtree_com: wp.array2d(dtype=wp.vec3)
    subtree_mass: wp.array2d(dtype=float)

    geom_X: wp.array2d(dtype=wp.transform)
    vis_X: wp.array2d(dtype=wp.transform)
    vis_beam_pos: array("nworld", "nbeams", "nbeam_visuals", wp.vec3)

    collision_pair: wp.array(dtype=wp.vec2i)
    collision_pairid: wp.array(dtype=wp.vec2i)
    collision_worldid: wp.array(dtype=int)
    ncollision: wp.array(dtype=int)
    nacon: wp.array(dtype=int)
    contact: Contact

    muscle_length: wp.array2d(dtype=float)
    muscle_velocity: wp.array2d(dtype=float)

    site_rel_pos_B: wp.array2d(dtype=wp.vec3)
    site_pos_G: wp.array2d(dtype=wp.vec3)
    site_vel_G: wp.array2d(dtype=wp.vec3)

    muscle_moment_arm: wp.array3d(dtype=float)

    muscle_length_info: wp.array2d(dtype=MuscleLengthInfo)
    muscle_velocity_info: wp.array2d(dtype=FiberVelocityInfo)
    muscle_dynamics_info: wp.array2d(dtype=MuscleDynamicsInfo)
    muscle_norm_fiber_length: wp.array2d(dtype=float)
    muscle_actuation: wp.array2d(dtype=float)
    muscle_actuation_passive: wp.array2d(dtype=float)
    muscle_metabolic: wp.array2d(dtype=float)

    # Adaptive integrator fields
    time1: wp.array(dtype=float)
    step_size: wp.array(dtype=float)
    actual_step_size: wp.array(dtype=float)
    artificially_limited: wp.array(dtype=bool)
    step_accepted: wp.array(dtype=bool)
    integration_done: wp.array(dtype=bool)
    nintegrating: wp.array(dtype=int)
    # error estimate for adaptive stepping
    qvel_scales: wp.array2d(dtype=float)
    z_scales: wp.array2d(dtype=float)
    qpos_diff: wp.array2d(dtype=float)
    ninv_dq_tmp: wp.array2d(dtype=float)
    qpos_diff_scaled: wp.array2d(dtype=float)
    qvel_diff: wp.array2d(dtype=float)
    z_diff: wp.array2d(dtype=float)
    qpos_err: wp.array(dtype=float)
    qvel_err: wp.array(dtype=float)
    z_err: wp.array(dtype=float)
    error: wp.array(dtype=float)
    steps_attempted: wp.array(dtype=int)

    # Midpoint integrator scratch
    integrator_midpoint_scratch: IntegratorMidpointScratch

    # Stored state for adaptive time-stepper
    integrator_scratch: list[IntegratorStateScratch]
    integrator_dot_scratch: list[IntegratorDotScratch]  # for higher order integrators
    qvel_buffer: wp.array2d(dtype=float)
