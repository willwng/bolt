import dataclasses
import enum

import warp as wp

TILE_SIZE_JTDAJ_DENSE = 16
TILE_SIZE_SITE = 256


# TODO(team): add check that all wp.launch_tiled 'block_dim' settings are configurable
@dataclasses.dataclass
class BlockDim:
    """Block dimension 'block_dim' settings for wp.launch_tiled.

    TODO(team): experimental and may be removed
    """

    # collision_driver
    segmented_sort: int = 128
    # forward
    euler_dense: int = 32
    actuator_velocity: int = 32
    site_diffs: int = 128

    adjust_scales: int = 64
    error_step: int = 64

    # ray
    ray: int = 64
    # sensor
    contact_sort: int = 64
    energy_vel_kinetic: int = 32
    # smooth
    cholesky_factorize: int = 32
    cholesky_solve: int = 32
    cholesky_factorize_solve: int = 32
    # solver
    update_gradient_cholesky: int = 64
    update_gradient_cholesky_blocked: int = 32
    update_gradient_JTDAJ_sparse: int = 64
    update_gradient_JTDAJ_dense: int = 96
    # support
    mul_m_dense: int = 32


class BroadphaseType(enum.IntEnum):
    """Type of broadphase algorithm.

    Attributes:
       NXN: Broad phase checking all pairs
       SAP_TILE: Sweep and prune broad phase using tile sort
       SAP_SEGMENTED: Sweep and prune broad phase using segment sort
    """

    NXN = 0
    SAP_TILE = 1
    SAP_SEGMENTED = 2


class BroadphaseFilter(enum.IntFlag):
    """Bitmask specifying which collision functions to run during broadphase.

    Attributes:
      PLANE: collision between bounding sphere and plane
      SPHERE: collision between bounding spheres
      AABB: collision between axis-aligned bounding boxes
      OBB: collision between oriented bounding boxes
    """

    PLANE = 1
    SPHERE = 2
    AABB = 4
    OBB = 8


class JointType(enum.IntEnum):
    """Type of degree of freedom.

    Attributes:
      FREE:  global position and orientation (quat)       (7,)
      BALL:  orientation (quat) relative to parent        (4,)
      SLIDE: sliding distance along body-fixed axis       (1,)
      PIN: rotation angle (rad) around joint z-axis       (1,)
      UNIVERSAL: two rotation angles (rad) around joint x- and y-axes (2,)
      CUSTOM: custom joint with up to 6 dofs              (<=6,)
      WELD: no dofs
      DUMMY: for ground (represents world body)
    """

    FREE = 0
    BALL = 1
    SLIDE = 2
    PIN = 3
    UNIVERSAL = 4
    CUSTOM = 5
    WELD = 6
    DUMMY = 7  # for ground


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


class ConstraintState(enum.IntEnum):
    """State of constraint.

    Attributes:
      SATISFIED: constraint satisfied, zero cost (limit, contact)
      QUADRATIC: quadratic cost (equality, friction, limit, contact)
      LINEARNEG: linear cost, negative side (friction)
      LINEARPOS: linear cost, positive side (friction)
      CONE: square distance to cone cost (elliptic contact)
    """

    SATISFIED = 0
    QUADRATIC = 1
    LINEARNEG = 2
    LINEARPOS = 3
    CONE = 4


class ConstraintType(enum.IntEnum):
    """Type of constraint.

    Attributes:
      LIMIT_JOINT: joint limit
      CONTACT_FRICTIONLESS: frictionless contact
      CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
    """

    LIMIT_JOINT = 0
    CONTACT_FRICTIONLESS = 1
    CONTACT_ELLIPTIC = 2


class EqType(enum.IntEnum):
    """Type of equality constraint.

    Attributes:
      CONNECT: connect two bodies at a point (ball joint)
      JOINT: couple the values of two scalar joints with cubic
      WELD: fix relative position and orientation of two bodies
      TENDON: couple the lengths of two tendons with cubic
    """

    CONNECT = 0
    WELD = 1
    JOINT = 2
    TENDON = 3


class CustomFnType(enum.IntEnum):
    """
    Custom function
    CONSTANT: constant function
    LINEAR: linear function
    """

    CONSTANT = 0
    LINEAR = 1


class vec5f(wp.types.vector(length=5, dtype=float)):
    pass


class vec6f(wp.types.vector(length=6, dtype=float)):
    pass


class vec8f(wp.types.vector(length=8, dtype=float)):
    pass


class vec8i(wp.types.vector(length=8, dtype=int)):
    pass


class vec10f(wp.types.vector(length=10, dtype=float)):
    pass


class mat34f(wp.types.matrix(shape=(3, 4), dtype=float)):
    pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=float)):
    pass


class mat411f(wp.types.matrix(shape=(4, 11), dtype=float)):
    pass


vec5 = vec5f
vec6 = vec6f
vec10 = vec10f
mat34 = mat34f
mat43 = mat43f
mat411 = mat411f


def array(*args) -> wp.array:
    """A wrapper around wp.array that adds extra metadata to ease type introspection.

    Format is array(dim_1, dim_2, ..., dtype).  dim may be a constant int, or reference a size from
    Model or Data (e.g. "nq" or "nworld").  dim may also be "*", which means any nonzero size.
    """
    shape, dtype = args[:-1], args[-1]

    arr = wp.array(ndim=len(shape), dtype=dtype)
    arr.shape = shape

    return arr


class SolverType(enum.IntEnum):
    """Constraint solver algorithm.

    Attributes:
      CG: Conjugate gradient (primal)
      NEWTON: Newton (primal)
    """

    CG = 1
    NEWTON = 2


class ContactType(enum.IntEnum):
    """ Contact model type.
    Attributes:
        MUJOCO: MuJoCo contact model (constraint-based)
        HUNT_CROSSLEY: Hunt-Crossley contact model (force-based)
    """
    MUJOCO = 1
    HUNT_CROSSLEY = 2
    HUNT_CROSSLEY_SMOOTH = 3


class LimitType(enum.IntEnum):
    """ Contact model type.
    Attributes:
        MUJOCO: MuJoCo limit model (constraint-based)
        EXPONENTIAL: Exponential Spring Function
        HUNT_CROSSLEY: Hunt-Crossley-like limit model
    """
    MUJOCO = 1
    EXPONENTIAL = 2
    HUNT_CROSSLEY = 3


class IntegratorType(enum.IntEnum):
    """ Integrator type.
    Attributes:
        EULER_FIXED: Fixed-step Euler (semi-implicit with implicit damping)
        RK4_FIXED: Fixed-step 4th-order Runge-Kutta
    """
    EULER_FIXED = 1
    RK4_FIXED = 2


@dataclasses.dataclass
class MetabolicOptions:
    activation_maintenance_rate_on: bool
    shortening_rate_on: bool
    mechanical_work_rate_on: bool
    enforce_minimum_heat_rate: bool

    aerobic_factor: float
    muscle_effort_scaling_factor: float
    use_bhargava_recruitment: bool
    include_negative_mechanical_work: bool
    forbid_negative_total_power: bool


@dataclasses.dataclass
class Option:
    """Physics options.

    Attributes:
      impratio: ratio of friction-to-normal contact impedance
      tolerance: main solver tolerance
      ls_tolerance: CG/Newton linesearch tolerance
      ccd_tolerance: convex collision detection tolerance
      gravity: gravitational acceleration
      solver: solver algorithm (SolverType)
      iterations: number of main solver iterations
      ls_iterations: maximum number of CG/Newton linesearch iterations
      ccd_iterations: number of iterations in convex collision detection
      warm_start: flag to enable warm starting of solver

      muscle_dyn_substeps: number of substeps to take for muscle dynamics. 0 = no substeps but still integrate
      use_fn_path: flag to use muscle path functions

    variable-step size integrator:
      safety:
      min_shrink:
      max_grow:
      hysteresis_low:
      hysteresis_high:
      accuracy:

    warp only fields:
      ls_parallel: evaluate engine solver step sizes in parallel
      ls_parallel_min_step: minimum step size for solver linesearch
      graph_conditional: flag to use cuda graph conditional
      run_collision_detection: if False, skips collision detection and allows user-populated
        contacts during the physics step (as opposed to DisableBit.CONTACT which explicitly
        zeros out the contacts at each step)
    """

    impratio: float
    tolerance: float
    ls_tolerance: float
    ccd_tolerance: float
    gravity: float
    solver: SolverType
    contact_type: ContactType
    limit_type: LimitType
    integrator: IntegratorType
    iterations: int
    ls_iterations: int
    ccd_iterations: int
    warm_start: bool

    enable_drag: bool

    muscle_dyn_substeps: int
    use_fn_path: bool
    metabolic_options: MetabolicOptions

    safety: float
    min_shrink: float
    max_grow: float
    hysteresis_low: float
    hysteresis_high: float
    accuracy: float
    use_inf_norm: bool

    qvel_weights: wp.array(dtype=float)

    solref: wp.vec2
    solimp: vec5

    ls_parallel: bool
    ls_parallel_min_step: float
    graph_conditional: bool

    visuals: bool


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
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    optimal_pennation_angle: float
    fiber_damping: float
    v_max: float

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
    fiber_velocity: float
    fiber_velocity_along_tendon: float
    norm_fiber_velocity: float
    pennation_angular_velocity: float
    tendon_velocity: float
    norm_tendon_velocity: float
    fiber_force_velocity_multiplier: float


@wp.struct
class MuscleDynamicsInfo:
    fiber_force: float
    fiber_force_along_tendon: float
    norm_fiber_force: float
    active_fiber_force: float
    passive_fiber_force: float
    tendon_force: float
    norm_tendon_force: float


@dataclasses.dataclass
class MeshLoadResult:
    file: str
    scale: list[float]


@dataclasses.dataclass
class Constraint:
    """Constraint data.

    Attributes:
      type: constraint type (ConstraintType)            (nworld, njmax)
      id: id of object of specific type                 (nworld, njmax)
      J: constraint Jacobian                            (nworld, njmax, nv)
      pos: constraint position (equality, contact)      (nworld, njmax)
      margin: inclusion margin (contact)                (nworld, njmax)
      D: constraint mass                                (nworld, njmax)
      vel: velocity in constraint space: J*qvel         (nworld, njmax)
      aref: reference pseudo-acceleration               (nworld, njmax)
      frictionloss: frictionloss (friction)             (nworld, njmax)
      force: constraint force in constraint space       (nworld, njmax)
      Jaref: Jac*qacc - aref                            (nworld, njmax)
      Ma: M*qacc                                        (nworld, nv)
      grad: gradient of master cost                     (nworld, nv)
      grad_dot: dot(grad, grad)                         (nworld,)
      Mgrad: M / grad                                   (nworld, nv)
      search: linesearch vector                         (nworld, nv)
      search_dot: dot(search, search)                   (nworld,)
      gauss: Gauss Cost                                 (nworld,)
      cost: constraint + Gauss cost                     (nworld,)
      prev_cost: cost from previous iter                (nworld,)
      state: constraint state                           (nworld, njmax)
      mv: qM @ search                                   (nworld, nv)
      jv: efc_J @ search                                (nworld, njmax)
      quad: quadratic cost coefficients                 (nworld, njmax, 3)
      quad_gauss: quadratic cost Gauss coefficients     (nworld, 3)
      h: Hessian                                        (nworld, nv, nv)
      alpha: line search step size                      (nworld,)
      prev_grad: previous grad                          (nworld, nv)
      prev_Mgrad: previous Mgrad                        (nworld, nv)
      beta: Polak-Ribiere beta                          (nworld,)
      done: solver done                                 (nworld,)
    """

    type: array("nworld", "njmax", int)
    id: array("nworld", "njmax", int)
    J: array("nworld", "njmax_pad", "nv", float)
    pos: array("nworld", "njmax", float)
    margin: array("nworld", "njmax", float)
    D: array("nworld", "njmax_pad", float)
    vel: array("nworld", "njmax", float)
    aref: array("nworld", "njmax", float)
    frictionloss: array("nworld", "njmax", float)
    force: array("nworld", "njmax", float)
    Jaref: array("nworld", "njmax", float)
    Ma: array("nworld", "nv", float)
    grad: array("nworld", "nv", float)
    cholesky_L_tmp: array("nworld", "nv", "nv", float)
    cholesky_y_tmp: array("nworld", "nv", "nv", float)
    grad_dot: array("nworld", float)
    Mgrad: array("nworld", "nv", float)
    search: array("nworld", "nv", float)
    search_dot: array("nworld", float)
    gauss: array("nworld", float)
    cost: array("nworld", float)
    prev_cost: array("nworld", float)
    state: array("nworld", "njmax_pad", int)
    mv: array("nworld", "nv", float)
    jv: array("nworld", "njmax", float)
    quad: array("nworld", "njmax", wp.vec3)
    quad_gauss: array("nworld", wp.vec3)
    h: array("nworld", "nv", "nv", float)
    alpha: array("nworld", float)
    prev_grad: array("nworld", "nv", float)
    prev_Mgrad: array("nworld", "nv", float)
    beta: array("nworld", float)
    done: array("nworld", bool)


@dataclasses.dataclass
class TileSet:
    """Tiling configuration for decomposable block diagonal matrix.

    For non-square, non-block-diagonal tiles, use two tilesets.

    Attributes:
      adr: address of each tile in the set
      size: size of all the tiles in this set
    """

    adr: wp.array(dtype=int)
    size: int


@dataclasses.dataclass
class Model:
    """Model definition and parameters.

    Attributes:
      nbody: number of bodies
      nq: number of generalized coordinates
      nv: number of degrees of freedom
      nmuscle: number of muscles
      nactuator: number of "ideal" actuators
      ndoflimit: number of dofs with limits

      njnts_conv: number of conventional joints
      njnts_cst: number of custom joints

      ngeom: number of geoms
      nsite: number of sites
      nsite_cond: number of conditional sites

      opt: physics options
      muscle_metadata: muscle metadata                         (nmuscle,)
      muscle_data: same as above, but for modification

      actuator_metadata: actuator metadata                     (nactuator,)

      qpos0: qpos values at default pose                       (nq,)
      qpos_spring: reference pose for springs                  (nq,)

      body_mass: mass                                          (nbody,)
      body_subtreemass: mass of subtree starting at this body  (nbody,)
      body_inertia: diagonal inertia in ipos/iquat frame       (nbody, 3)
      body_ipos: local position of center of mass              (nbody, 3)
      body_iquat: local orientation of inertia ellipsoid       (nbody, 4)

      body_rootid: id of root above body                       (nbody,)
      body_parentid: id of body's parent                       (nbody,)
      body_tree: list of body ids by tree level

      jnt_type: type of joint (JointType)                      (nbody,)
      jnt_stiffness: joint stiffness                           (nbody,)
      jnt_qposadr: start addr in 'qpos' for joint's data       (nbody,)
      jnt_dofadr: start addr in 'qvel' for joint's data        (nbody,)
      jnt_rel_parent: offset from parent frame                 (nbody, 3)
      jnt_rel_child: offset from child frame                   (nbody, 3)
      jnt_rel_parent_rot: rotation from parent frame           (nbody, 4)
      jnt_rel_child_rot: rotation from child frame             (nbody, 4)

      * for custom joints *
      jnt_cst_adr: address of custom joint, -1 if conventional (nbody,)
      const_fns:     (c) of constant functions                 (<=6*njnts_cst,)
      linear_fns:    (m, b) of linear functions                (<=6*njnts_cst, 2)
      cst_txfm_axis: axis for each spatial transform           (njnts_cst, 6, vec3)
      cst_txfm_fn: function type for each spatial transform    (njnts_cst, 6)
      cst_txfm_fn_adr: address of spatial transform function   (njnts_cst, 6)
      cst_txfm_qadr: qpos address for each spatial transform   (njnts_cst, 6)
      cst_txfm_dofadr: dof address for each spatial transform  (njnts_cst, 6)

      dof_bodyid: id of dof's body                             (nv,)
      dof_parentid: id of dof's parent; -1: none               (nv,)
      dof_armature: dof armature inertia/mass                  (nv)
      dof_damping: damping coefficient                         (nv)

      * dof limits
      limit_dof_range: joint limits (min, max)                 (ndoflimit, 2)
      limit_dof_adr: dof address of dof-limit                  (ndoflimit,)
      limit_dof_qadr: qpos address of dof-limit                (ndoflimit,)

      geom_type: geometric type (GeomType)                     (ngeom,)
      geom_bodyid: id of geom's body                           (ngeom,)
      geom_size: geom-specific size parameters                 (ngeom, 3)
      geom_pos: local position offset rel. to body             (ngeom, 3)
      geom_quat: local orientation offset rel. to body         (ngeom, 4)
      geom_friction: friction for (slide, spin, roll)          (ngeom, 3)
      geom_stiffness: contact stiffness (Hunt-Crossley)        (ngeom,)
      geom_dissipation: contact dissipation (Hunt-Crossley)    (ngeom,)
      geom_transition_velocity: friction transition velocity   (ngeom,)
      geom_priority: collision priority (Hunt-Crossley)        (ngeom,)
      geom_aabb: axis-aligned bounding box (center, size)      (ngeom, 2, 3)
      geom_rbound: bounding sphere radius                      (ngeom,)

      geom_pair_type_count: count of max number of each potential collision
      nxn_geom_pair_filtered: valid collision pair geom ids    (<=ngeom*(ngeom-1)/2,)
      nxn_pairid_filtered: active subset of nxn_pairid         (<=ngeom*(ngeom-1)/2, 2)

      site_bodyid: id of site's body                           (nsite,)
      site_pos: local position offset rel. to body             (nsite, 3)
      site_cond_id: conditional site id                        (nsite_cond,)
      site_cond_qadr: conditional site qpos address            (nsite_cond,)
      site_cond_range: conditional site range (min, max)       (nsite_cond, 2)

      muscle_pts_adr: address of first point in muscle's path  (nmuscle,)
      muscle_pts_num: number of points in muscle's path        (nmuscle,)

      mean_inertia: mean diagonal inertia                      ()
      body_invweight0: mean inv inert in qpos0 (trn, rot)      (nbody, 2)
      dof_invweight0: diag. inverse inertia in qpos0           (nv)
    """

    nbody: int
    nq: int
    nv: int
    nmuscle: int
    nactuator: int
    ndoflimit: int

    njnts_conv: int
    njnts_cst: int

    ngeom: int
    nvis: int
    nsite: int
    nsite_cond: int

    opt: Option
    muscle_metadata: array("nmuscle", MuscleMetadata)
    muscle_data: list[MuscleMetadata]

    actuator_metadata: array("nactuator", ActuatorMetadata)

    qpos0: array("nq", float)
    qpos_spring: array("nq", float)

    body_mass: array("nbody", float)
    body_subtreemass: array("nbody", float)
    body_inertia: array("nbody", wp.vec3)
    body_ipos: array("nbody", wp.vec3)
    body_iquat: array("nbody", wp.quat)

    body_rootid: array("nbody", int)
    body_parentid: array("nbody", int)
    body_tree: tuple[wp.array(dtype=int), ...]

    jnt_type: array("nbody", int)
    jnt_stiffness: array("nbody", float)
    jnt_qposadr: array("nbody", int)
    jnt_dofnum: array("nbody", int)
    jnt_dofadr: array("nbody", int)
    jnt_rel_parent: array("nbody", wp.vec3)
    jnt_rel_child: array("nbody", wp.vec3)
    jnt_rel_parent_rot: array("nbody", wp.quat)
    jnt_rel_child_rot: array("nbody", wp.quat)

    # Custom joint data
    jnt_cst_adr: array("nbody", int)
    const_fns: wp.array(dtype=float)
    linear_fns: wp.array(dtype=wp.vec2)
    cst_txfm_axis: wp.array2d(dtype=wp.vec3)
    cst_txfm_fn: wp.array2d(dtype=int)
    cst_txfm_fn_adr: wp.array2d(dtype=int)
    cst_txfm_qadr: wp.array2d(dtype=int)
    cst_txfm_dofadr: wp.array2d(dtype=int)

    # Dof data
    dof_armature: wp.array(dtype=float)
    dof_damping: wp.array(dtype=float)
    dof_bodyid: wp.array(dtype=int)
    dof_parentid: wp.array(dtype=int)

    # Dof limits
    limit_dof_range: wp.array2d(dtype=wp.vec2)
    limit_dof_adr: wp.array(dtype=int)
    limit_dof_qadr: wp.array(dtype=int)
    # For exponential-force limits
    limit_dof_forces: wp.array2d(dtype=wp.vec2)
    limit_dof_shapes: wp.array2d(dtype=wp.vec2)

    # Collision geometry
    geom_type: wp.array(dtype=int)
    geom_bodyid: wp.array(dtype=int)
    geom_size: wp.array(dtype=wp.vec3)
    geom_pos: wp.array(dtype=wp.vec3)
    geom_quat: wp.array(dtype=wp.quat)
    geom_friction: wp.array(dtype=wp.vec3)
    geom_stiffness: wp.array(dtype=float)
    geom_dissipation: wp.array(dtype=float)
    geom_transition_velocity: wp.array(dtype=float)
    geom_priority: wp.array(dtype=int)
    geom_aabb: wp.array2d(dtype=wp.vec3)
    geom_rbound: wp.array(dtype=float)

    geom_pair_type_count: tuple[int, ...]
    nxn_geom_pair_filtered: wp.array(dtype=wp.vec2i)
    nxn_pairid_filtered: wp.array(dtype=wp.vec2i)

    # Visual geometry
    vis_pos: wp.array(dtype=wp.vec3)
    vis_quat: wp.array(dtype=wp.quat)
    vis_bodyid: wp.array(dtype=int)

    # Attachment sites (muscle path)
    site_bodyid: wp.array(dtype=int)
    site_pos: wp.array(dtype=wp.vec3)
    site_cond_id: wp.array(dtype=int)
    site_cond_qadr: wp.array(dtype=int)
    site_cond_range: wp.array2d(dtype=wp.vec2)

    # Muscles
    muscle_pts_adr: wp.array(dtype=int)
    muscle_pts_num: wp.array(dtype=int)
    # Polynomial/function paths
    muscle_poly_coeffs: array("npoly_coeffs", float)
    muscle_poly_adr: array("nmuscle", int)
    muscle_poly_order: array("nmuscle", int)
    muscle_poly_qpos_adr: array("total_order", int)
    muscle_poly_dof_adr: array("total_order", int)
    muscle_dep_dof_num: array("nmuscle", int)
    muscle_dep_dof_adr: array("nmuscle", int)

    # To be computed at model creation
    mean_inertia: float
    body_invweight0: wp.array(dtype=wp.vec2)
    dof_invweight0: wp.array(dtype=float)

    qM_tiles: tuple[TileSet, ...]
    block_dim: BlockDim
    dof_tri_row: wp.array(dtype=int)
    dof_tri_col: wp.array(dtype=int)


@dataclasses.dataclass
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
      efc_address: address in efc; -1: not included                    (naconmax, ncondim)
      worldid: world id                                                (naconmax,)
      geomcollisionid: i-th contact generated for geom                 (naconmax,)
                       helps uniquely identity contact when multiple
                       contacts are generated for geom pair
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
    efc_address: wp.array2d(dtype=int)
    worldid: wp.array(dtype=int)
    geomcollisionid: wp.array(dtype=int)



@dataclasses.dataclass
class Data:
    """Dynamic state that updates each step.

    Attributes:
      world_reset: whether to reset the world                     (nworld,)

      solver_niter: number of solver iterations                   (nworld,)
      nl: number of limit constraints                             (nworld,)
      nefc: number of constraints                                 (nworld,)

      time: simulation time                                       (nworld,)
      time1: current target time for integrator (t1)              (nworld,)
      next_time: final target time for integrator (tMax)          (nworld,)

      qpos: position                                              (nworld, nq)
      qvel: velocity                                              (nworld, nv)
      m_act: muscle activation                                    (nworld, nmuscles)
      a_act: actuator activation                                  (nworld, nactuator)
      m_state: muscle state variable                               (nworld, nmuscles)

      qacc: acceleration                                          (nworld, nv)
      m_act_dot: time-derivative of actuator activation           (nworld, na)
      a_act_dot: time-derivative of actuator activation           (nworld, nactuator)
      m_excitations: muscle excitations                            (nworld, nmuscles)
      m_state_dot: time-derivative of muscle state variable        (nworld, nmuscles)

      qacc_warmstart: acceleration used for warmstart             (nworld, nv)
      qfrc_applied: applied generalized force                     (nworld, nv)
      xfrc_applied: applied Cartesian force/torque                (nworld, nbody, 6)
      xfrc_user: user Cartesian force/torque applied to body      (nworld, nbody, 6)
      grf: ground reaction force                                  (nworld, 6)
      joint_moments: joint moments                                (nworld, nv)

      xpos: Cartesian position of body frame                      (nworld, nbody, 3)
      xquat: Cartesian orientation of body frame                  (nworld, nbody, 4)
      xmat: Cartesian orientation of body frame                   (nworld, nbody, 3, 3)
      xipos: Cartesian position of body com                       (nworld, nbody, 3)
      ximat: Cartesian orientation of body inertia                (nworld, nbody, 3, 3)
      xanchor: Cartesian position of joint anchor                 (nworld, njnt, 3)
      xaxis: Cartesian joint axis (including temporaries)         (nworld, njnt, 6, 3)

      geom_xpos: Cartesian geom position                          (nworld, ngeom, 3)
      geom_xquat: Cartesian geom orientation                      (nworld, ngeom, 4)
      geom_xmat: Cartesian geom orientation                       (nworld, ngeom, 3, 3)

      site_rpos: local position of site rel. to body              (nworld, nsite, 3)
      site_xpos: Cartesian site position                          (nworld, nsite, 3)
      site_xvel: Cartesian site velocity                          (nworld, nsite, 3)
      site_active: whether site is active                         (nworld, nsite)

      muscle_active_sites: "compacted" active sites               (nworld, nsite)
       [ for muscle i, active sites indices are consecutive ]
      muscle_num_active: number of active sites per muscle        (nworld, nmuscle)

      site_diff_vec: unit vector b/w consecutive active sites     (nworld, nsite-1, 3)
      site_diff_len: length b/w consecutive active sites          (nworld, nsite-1)
      site_diff_vel: projected velocity b/w active sites          (nworld, nsite-1)

      subtree_com: center of mass of each subtree                 (nworld, nbody, 3)
      cdof: com-based motion axis of each dof (rot:lin)           (nworld, nv, 6)
      cdof_tmp: cdof temporaries for custom joints                (nworld, njnts_cst, 6, 6)
      cinert: com-based body inertia and mass                     (nworld, nbody, 10)

      crb: com-based composite inertia and mass                   (nworld, nbody, 10)
      qM: total inertia (sparse) (nworld, 1, nM) or               (nworld, nv, nv) if dense
      qLD: L'*D*L factorization of M (sparse) (nworld, 1, nM) or  (nworld, nv, nv) if dense
      qLDiagInv: 1/diag(D)                                        (nworld, nv)

      muscle_length: muscle lengths                               (nworld, nmuscle)
      muscle_velocity: muscle velocities                          (nworld, nmuscle)
      muscle_actuation: muscle actuation forces                   (nworld, nmuscle)
      muscle_metabolic: muscle metabolic energy rate              (nworld, nmuscle)

      * for substepping muscle dynamics *
      muscle_length_prev: previous muscle lengths                 (nworld, nmuscle)
      muscle_velocity_prev: previous muscle velocities            (nworld, nmuscle)

      cvel: com-based velocity (rot:lin)                          (nworld, nbody, 6)
      xvel: Cartesian body velocity in body frame (ang, vel)      (nworld, nbody, 6)
      xivel: Cartesian body velocity in body-com frame            (nworld, nbody, 6)
      cdof_dot: time-derivative of cdof (rot:lin)                 (nworld, nv, 6)

      qfrc_bias: C(qpos,qvel)                                     (nworld, nv)
      qfrc_spring: passive spring force                           (nworld, nv)
      qfrc_damper: passive damper force                           (nworld, nv)

      qfrc_muscle: muscle generalized force                       (nworld, nv)

      subtree_linvel: linear velocity of subtree com              (nworld, nbody, 3)
      subtree_angmom: angular momentum about subtree com          (nworld, nbody, 3)

      qfrc_smooth: net unconstrained force                        (nworld, nv)
      qacc_smooth: unconstrained acceleration                     (nworld, nv)
      qfrc_constraint: constraint force                           (nworld, nv)
      qfrc_inverse: net external force; should equal:             (nworld, nv)
                    qfrc_applied + J.T @ xfrc_applied
      cacc: com-based acceleration                                (nworld, nbody, 6)
      cfrc_int: com-based interaction force with parent           (nworld, nbody, 6)
      cfrc_ext: com-based external force on body                  (nworld, nbody, 6)
      contact: contact data
      efc: constraint data

      dof_lim_efc_address: dof limit efc adr. -1: not limited     (nworld, ndoflimit)
      dof_lim_torque: computed dof limit torque                   (nworld, ndoflimit)

    warp only fields:
      nworld: number of worlds
      naconmax: maximum number of contacts (shared across all worlds)
      njmax: maximum number of constraints per world
      nacon: number of detected contacts (across all worlds)
      nsolving: number of unconverged worlds                      (1,)
      subtree_bodyvel: subtree body velocity (ang, vel)           (nworld, nbody, 6)
    """
    world_reset: array("nworld", bool)

    solver_niter: wp.array(dtype=int)
    nl: wp.array(dtype=int)
    nefc: wp.array(dtype=int)
    needs_solve: wp.array(dtype=bool)

    time: wp.array(dtype=float)
    time1: wp.array(dtype=float)
    next_time: wp.array(dtype=float)

    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    m_act: wp.array2d(dtype=float)
    a_act: wp.array2d(dtype=float)
    m_state: wp.array2d(dtype=float)

    qacc: wp.array2d(dtype=float)
    m_act_dot: wp.array2d(dtype=float)
    a_act_dot: wp.array2d(dtype=float)
    m_excitations: wp.array2d(dtype=float)
    a_excitations: wp.array2d(dtype=float)
    m_state_dot: wp.array2d(dtype=float)

    qacc_warmstart: wp.array2d(dtype=float)
    qfrc_applied: wp.array2d(dtype=float)
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)
    xfrc_contact: wp.array2d(dtype=wp.spatial_vector)
    xfrc_user: wp.array2d(dtype=wp.spatial_vector)
    grf: wp.array2d(dtype=wp.vec3)
    joint_moments: wp.array2d(dtype=float)

    xpos: wp.array2d(dtype=wp.vec3)
    xquat: wp.array2d(dtype=wp.quat)
    xmat: wp.array2d(dtype=wp.mat33)
    xipos: wp.array2d(dtype=wp.vec3)
    ximat: wp.array2d(dtype=wp.mat33)
    xanchor: wp.array2d(dtype=wp.vec3)
    xaxis: wp.array3d(dtype=wp.quat)

    geom_xpos: wp.array2d(dtype=wp.vec3)
    geom_xquat: wp.array2d(dtype=wp.quat)
    geom_xmat: wp.array2d(dtype=wp.mat33)

    vis_xpos: wp.array2d(dtype=wp.vec3)
    vis_xquat: wp.array2d(dtype=wp.quat)

    site_rpos: wp.array2d(dtype=wp.vec3)
    site_xpos: wp.array2d(dtype=wp.vec3)
    site_xvel: wp.array2d(dtype=wp.vec3)
    site_active: wp.array2d(dtype=bool)

    muscle_active_sites: wp.array2d(dtype=int)
    muscle_num_active: wp.array2d(dtype=int)
    muscle_moment_arm: wp.array3d(dtype=float)

    site_diff_vec: wp.array2d(dtype=wp.vec3)
    site_diff_len: wp.array2d(dtype=float)
    site_diff_vel: wp.array2d(dtype=float)

    subtree_com: wp.array2d(dtype=wp.vec3)
    cdof: wp.array2d(dtype=wp.spatial_vector)
    cdof_tmp: wp.array3d(dtype=wp.spatial_vector)
    cinert: wp.array2d(dtype=vec10)

    crb: wp.array2d(dtype=vec10)
    qM: wp.array3d(dtype=float)
    qLD: wp.array3d(dtype=float)
    qLDiagInv: wp.array2d(dtype=float)

    muscle_length: wp.array2d(dtype=float)
    muscle_velocity: wp.array2d(dtype=float)
    muscle_actuation: wp.array2d(dtype=float)
    muscle_metabolic: wp.array2d(dtype=float)

    muscle_length_prev: wp.array2d(dtype=float)
    muscle_velocity_prev: wp.array2d(dtype=float)

    muscle_length_info: wp.array2d(dtype=MuscleLengthInfo)
    muscle_velocity_info: wp.array2d(dtype=FiberVelocityInfo)
    muscle_dynamics_info: wp.array2d(dtype=MuscleDynamicsInfo)

    cvel: wp.array2d(dtype=wp.spatial_vector)
    xvel: wp.array2d(dtype=wp.spatial_vector)
    xivel: wp.array2d(dtype=wp.spatial_vector)
    cdof_dot: wp.array2d(dtype=wp.spatial_vector)

    qfrc_bias: wp.array2d(dtype=float)
    qfrc_spring: wp.array2d(dtype=float)
    qfrc_damper: wp.array2d(dtype=float)

    qfrc_muscle: wp.array2d(dtype=float)
    qfrc_actuator: wp.array2d(dtype=float)
    qfrc_contact: wp.array2d(dtype=float)
    qfrc_limit: wp.array2d(dtype=float)

    subtree_linvel: wp.array2d(dtype=wp.vec3)
    subtree_angmom: wp.array2d(dtype=wp.vec3)

    qfrc_smooth: wp.array2d(dtype=float)
    qacc_smooth: wp.array2d(dtype=float)
    qfrc_constraint: wp.array2d(dtype=float)
    qfrc_inverse: wp.array2d(dtype=float)
    cacc: wp.array2d(dtype=wp.spatial_vector)
    cfrc_int: wp.array2d(dtype=wp.spatial_vector)
    cfrc_ext: wp.array2d(dtype=wp.spatial_vector)
    contact: Contact
    efc: Constraint

    dof_lim_efc_address: wp.array2d(dtype=int)

    #
    nworld: int
    naconmax: int
    njmax: int
    nacon: wp.array(dtype=int)
    nsolving: wp.array(dtype=int)
    subtree_bodyvel: wp.array2d(dtype=wp.spatial_vector)

    actual_step_size: wp.array(dtype=float)

    # collision driver
    collision_pair: wp.array(dtype=wp.vec2i)
    collision_pairid: wp.array(dtype=wp.vec2i)
    collision_worldid: wp.array(dtype=int)
    ncollision: wp.array(dtype=int)
