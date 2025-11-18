import dataclasses
import enum

import warp as wp

MJ_MAXVAL = 10000000000.0
MJ_MINIMP = 0.0001
MJ_MAXIMP = 0.9999
MJ_MINMU = 1e-05
MJ_MINVAL = 1e-15
MJ_MAXCONPAIR = 50
# maximum size (by number of edges) of an horizon in EPA algorithm
MJ_MAX_EPAHORIZON = 12
# maximum average number of trianglarfaces EPA can insert at each iteration
MJ_MAX_EPAFACES = 5

TILE_SIZE_JTDAJ_SPARSE = 16
TILE_SIZE_JTDAJ_DENSE = 16


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
    tendon_velocity: int = 32
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
      HINGE: rotation angle (rad) around body-fixed axis  (1,)
    """

    FREE = 0
    BALL = 1
    SLIDE = 2
    HINGE = 3
    UNIVERSAL = 4
    CUSTOM = 5
    DUMMY = 6 # for ground


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
      EQUALITY: equality constraint
      FRICTION_DOF: dof friction
      FRICTION_TENDON: tendon friction
      LIMIT_JOINT: joint limit
      LIMIT_TENDON: tendon limit
      CONTACT_FRICTIONLESS: frictionless contact
      CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
      CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
    """

    EQUALITY = 0
    FRICTION_DOF = 1
    FRICTION_TENDON = 2
    LIMIT_JOINT = 3
    LIMIT_TENDON = 4
    CONTACT_FRICTIONLESS = 5
    CONTACT_PYRAMIDAL = 6
    CONTACT_ELLIPTIC = 7


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


class WrapType(enum.IntEnum):
    """Type of tendon wrapping object.

    Attributes:
      JOINT: constant moment arm
      PULLEY: pulley used to split tendon
      SITE: pass through site
      SPHERE: wrap around sphere
      CYLINDER: wrap around (infinite) cylinder
    """

    JOINT = 0
    PULLEY = 1
    SITE = 2
    SPHERE = 3
    CYLINDER = 4


class State(enum.IntEnum):
    """State component elements as integer bitflags.

    Includes several convenient combinations of these flags.

    Attributes:
      TIME: time
      QPOS: position
      QVEL: velocity
      ACT: actuator activation
      WARMSTART: acceleration used for warmstart
      CTRL: control
      QFRC_APPLIED: applied generalized force
      XFRC_APPLIED: applied Cartesian force/torque
      EQ_ACTIVE: enable/disable constraints
      NSTATE: number of state elements
      PHYSICS: QPOS | QVEL | ACT
      FULLPHYSICS: TIME | PHYSICS | PLUGIN
      INTEGRATION: FULLPHYSICS | USER | WARMSTART
    """

    TIME = 0
    QPOS = 1
    QVEL = 2
    ACT = 3
    WARMSTART = 4
    CTRL = 5
    QFRC_APPLIED = 6
    XFRC_APPLIED = 7
    EQ_ACTIVE = 8
    NSTATE = 9
    PHYSICS = 10
    FULLPHYSICS = 11
    INTEGRATION = 12

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


class vec11f(wp.types.vector(length=11, dtype=float)):
    pass


vec5 = vec5f
vec6 = vec6f
vec10 = vec10f
vec11 = vec11f


class SolverType(enum.IntEnum):
    """Constraint solver algorithm.

    Attributes:
      CG: Conjugate gradient (primal)
      NEWTON: Newton (primal)
    """

    CG = 1
    NEWTON = 2


@dataclasses.dataclass
class Option:
    """Physics options.

    Attributes:
      timestep: simulation timestep
      impratio: ratio of friction-to-normal contact impedance
      tolerance: main solver tolerance
      ls_tolerance: CG/Newton linesearch tolerance
      ccd_tolerance: convex collision detection tolerance
      gravity: gravitational acceleration
      solver: solver algorithm (SolverType)
      iterations: number of main solver iterations
      ls_iterations: maximum number of CG/Newton linesearch iterations
      ccd_iterations: number of iterations in convex collision detection

    warp only fields:
      is_sparse: whether to use sparse representations
      ls_parallel: evaluate engine solver step sizes in parallel
      ls_parallel_min_step: minimum step size for solver linesearch
      graph_conditional: flag to use cuda graph conditional
      run_collision_detection: if False, skips collision detection and allows user-populated
        contacts during the physics step (as opposed to DisableBit.CONTACT which explicitly
        zeros out the contacts at each step)
    """

    timestep: float
    impratio: float
    tolerance: float
    ls_tolerance: float
    ccd_tolerance: float
    gravity: float
    solver: SolverType
    iterations: int
    ls_iterations: int
    ccd_iterations: int
    warm_start: bool
    ls_parallel: bool
    ls_parallel_min_step: float
    graph_conditional: bool


@dataclasses.dataclass
class Statistic:
    """Model statistics (in qpos0).

    Attributes:
      meaninertia: mean diagonal inertia
    """

    meaninertia: float


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

    type: wp.array2d(dtype=int)
    id: wp.array2d(dtype=int)
    J: wp.array3d(dtype=float)
    pos: wp.array2d(dtype=float)
    margin: wp.array2d(dtype=float)
    D: wp.array2d(dtype=float)
    vel: wp.array2d(dtype=float)
    aref: wp.array2d(dtype=float)
    frictionloss: wp.array2d(dtype=float)
    force: wp.array2d(dtype=float)
    Jaref: wp.array2d(dtype=float)
    Ma: wp.array2d(dtype=float)
    grad: wp.array2d(dtype=float)
    cholesky_L_tmp: wp.array3d(dtype=float)
    cholesky_y_tmp: wp.array2d(dtype=float)
    grad_dot: wp.array(dtype=float)
    Mgrad: wp.array2d(dtype=float)
    search: wp.array2d(dtype=float)
    search_dot: wp.array(dtype=float)
    gauss: wp.array(dtype=float)
    cost: wp.array(dtype=float)
    prev_cost: wp.array(dtype=float)
    state: wp.array2d(dtype=int)
    mv: wp.array2d(dtype=float)
    jv: wp.array2d(dtype=float)
    quad: wp.array2d(dtype=wp.vec3)
    quad_gauss: wp.array(dtype=wp.vec3)
    h: wp.array3d(dtype=float)
    alpha: wp.array(dtype=float)
    prev_grad: wp.array2d(dtype=float)
    prev_Mgrad: wp.array2d(dtype=float)
    beta: wp.array(dtype=float)
    done: wp.array(dtype=bool)


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
      nmuscle: number of tendons
      ndoflimit: number of dofs with limits

      njnts_conv: number of conventional joints
      njnts_cst: number of custom joints

      ngeom: number of geoms
      nsite: number of sites

      opt: physics options

      qpos0: qpos values at default pose                       (nq,)
      qpos_spring: reference pose for springs                  (nq,)

      body_mass: mass                                          (nbody,)
      body_subtreemass: mass of subtree starting at this body  (nbody,)
      body_inertia: diagonal inertia in ipos/iquat frame       (nbody, 3)
      body_ipos: local position of center of mass              (nbody, 3)
      body_iquat: local orientation of inertia ellipsoid       (nbody, 4)

      body_geomnum: number of geoms                            (nbody,)
      body_geomadr: start addr of geoms; -1: no geoms          (nbody,)

      body_rootid: id of root above body                       (nbody,)
      body_parentid: id of body's parent                       (nbody,)
      jnt_type: type of joint (JointType)                      (nbody,)
      jnt_stiffness: joint stiffness                           (nbody,)
      jnt_qposadr: start addr in 'qpos' for joint's data       (nbody,)
      jnt_dofadr: start addr in 'qvel' for joint's data        (nbody,)
      jnt_rel_parent: offset from parent frame                 (nbody, 3)
      jnt_rel_child: offset from child frame                   (nbody, 3)
      jnt_rel_parent_rot: rotation from parent frame           (nbody, 4)
      jnt_rel_child_rot: rotation from child frame             (nbody, 4)

      dof_bodyid: id of dof's body                             (nv,)
      dof_parentid: id of dof's parent; -1: none               (nv,)

      * for custom joints *
      jnt_cst_adr: address of custom joint, -1 if conventional (nbody,)
      const_fns:     (c) of constant functions                 (<=6*njnts_cst,)
      linear_fns:    (m, b) of linear functions                (<=6*njnts_cst, 2)
      cst_txfm_axis: axis for each spatial transform           (njnts_cst, 6, vec3)
      cst_txfm_fn: function type for each spatial transform    (njnts_cst, 6)
      cst_txfm_fn_adr: address of spatial transform function   (njnts_cst, 6)
      cst_txfm_qadr: qpos address for each spatial transform   (njnts_cst, 6)
      cst_txfm_dofadr: dof address for each spatial transform  (njnts_cst, 6)

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

      nxn_geom_pair_filtered: valid collision pair geom ids    (<=ngeom*(ngeom-1)/2,)

      site_bodyid: id of site's body                           (nsite,)
      site_pos: local position offset rel. to body             (nsite, 3)

      muscle_pts_adr: address of first point in muscle's path  (nmuscle,)
      muscle_pts_num: number of points in muscle's path        (nmuscle,)

      dof_armature: dof armature inertia/mass                  (nv)
      dof_damping: damping coefficient                         (nv)

      mean_inertia: mean diagonal inertia                      ()
      body_invweight0: mean inv inert in qpos0 (trn, rot)      (nbody, 2)
      dof_invweight0: diag. inverse inertia in qpos0           (nv)
    """

    nbody: int
    nq: int
    nv: int
    nmuscle: int
    ndoflimit: int

    njnts_conv: int
    njnts_cst: int

    ngeom: int
    nsite: int

    opt: Option

    qpos0: wp.array(dtype=float)
    qpos_spring: wp.array(dtype=float)

    body_mass: wp.array(dtype=float)
    body_subtreemass: wp.array(dtype=float)
    body_inertia: wp.array(dtype=wp.vec3)
    body_ipos: wp.array(dtype=wp.vec3)
    body_iquat: wp.array(dtype=wp.quat)

    body_geomnum: wp.array(dtype=int)
    body_geomadr: wp.array(dtype=int)

    body_rootid: wp.array(dtype=int)
    body_parentid: wp.array(dtype=int)
    body_tree: tuple[wp.array(dtype=int), ...]

    jnt_type: wp.array(dtype=int)
    jnt_stiffness: wp.array(dtype=float)
    jnt_qposadr: wp.array(dtype=int)
    jnt_dofnum: wp.array(dtype=int)
    jnt_dofadr: wp.array(dtype=int)
    jnt_rel_parent: wp.array(dtype=wp.vec3)
    jnt_rel_child: wp.array(dtype=wp.vec3)
    jnt_rel_parent_rot: wp.array(dtype=wp.quat)
    jnt_rel_child_rot: wp.array(dtype=wp.quat)

    # Custom joint data
    jnt_cst_adr: wp.array(dtype=int)
    const_fns: wp.array(dtype=float)
    linear_fns: wp.array(dtype=wp.vec2)
    cst_txfm_axis: wp.array2d(dtype=wp.vec3)
    cst_txfm_fn: wp.array2d(dtype=int)
    cst_txfm_fn_adr: wp.array2d(dtype=int)
    cst_txfm_qadr: wp.array2d(dtype=int)
    cst_txfm_dofadr: wp.array2d(dtype=int)

    # Dof limits
    limit_dof_range: wp.array2d(dtype=wp.vec2)
    limit_dof_adr: wp.array(dtype=int)
    limit_dof_qadr: wp.array(dtype=int)

    # Collision geometry
    geom_type: wp.array(dtype=int)
    geom_bodyid: wp.array(dtype=int)
    geom_size: wp.array(dtype=wp.vec3)
    geom_pos: wp.array(dtype=wp.vec3)
    geom_quat: wp.array(dtype=wp.quat)
    geom_friction: wp.array(dtype=wp.vec3)
    geom_aabb: wp.array2d(dtype=wp.vec3)
    geom_rbound: wp.array(dtype=float)
    geom_margin: wp.array(dtype=float)

    geom_pair_type_count: tuple[int, ...]
    nxn_geom_pair_filtered: wp.array(dtype=wp.vec2i)
    nxn_pairid_filtered: wp.array(dtype=wp.vec2i)

    # Attachment sites (muscle path)
    site_bodyid: wp.array(dtype=int)
    site_pos: wp.array(dtype=wp.vec3)

    # Muscles
    muscle_pts_adr: wp.array(dtype=int)
    muscle_pts_num: wp.array(dtype=int)

    dof_armature: wp.array(dtype=float)
    dof_damping: wp.array(dtype=float)

    dof_bodyid: wp.array(dtype=int)
    dof_parentid: wp.array(dtype=int)

    # To be computed at model creation
    body_invweight0: wp.array(dtype=wp.vec2)
    mean_inertia: float
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
    geom: wp.array(dtype=wp.vec2i)
    efc_address: wp.array2d(dtype=int)
    worldid: wp.array(dtype=int)
    geomcollisionid: wp.array(dtype=int)


@dataclasses.dataclass
class Data:
    """Dynamic state that updates each step.

    Attributes:
      solver_niter: number of solver iterations                   (nworld,)
      nl: number of limit constraints                             (nworld,)
      nefc: number of constraints                                 (nworld,)
      time: simulation time                                       (nworld,)
      qpos: position                                              (nworld, nq)
      qvel: velocity                                              (nworld, nv)
      act: actuator activation                                    (nworld, nmuscles)
      qacc_warmstart: acceleration used for warmstart             (nworld, nv)
      qfrc_applied: applied generalized force                     (nworld, nv)
      xfrc_applied: applied Cartesian force/torque                (nworld, nbody, 6)

      qacc: acceleration                                          (nworld, nv)
      act_dot: time-derivative of actuator activation             (nworld, na)

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
      site_diff_vec: Cartesian unit vector b/w consecutive sites  (nworld, nsite-1, 3)
      site_diff_len: length b/w consecutive sites                 (nworld, nsite-1)

      subtree_com: center of mass of each subtree                 (nworld, nbody, 3)
      cdof: com-based motion axis of each dof (rot:lin)           (nworld, nv, 6)
      cinert: com-based body inertia and mass                     (nworld, nbody, 10)

      crb: com-based composite inertia and mass                   (nworld, nbody, 10)
      qM: total inertia (sparse) (nworld, 1, nM) or               (nworld, nv, nv) if dense
      qLD: L'*D*L factorization of M (sparse) (nworld, 1, nM) or  (nworld, nv, nv) if dense
      qLDiagInv: 1/diag(D)                                        (nworld, nv)

      muscle_length: muscle lengths                               (nworld, nmuscle)
      muscle_velocity: muscle velocities                          (nworld, nmuscle)

      cvel: com-based velocity (rot:lin)                          (nworld, nbody, 6)
      cdof_dot: time-derivative of cdof (rot:lin)                 (nworld, nv, 6)

      qfrc_bias: C(qpos,qvel)                                     (nworld, nv)
      qfrc_spring: passive spring force                           (nworld, nv)
      qfrc_damper: passive damper force                           (nworld, nv)
      qfrc_passive: total passive force                           (nworld, nv)

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

    warp only fields:
      nworld: number of worlds
      naconmax: maximum number of contacts (shared across all worlds)
      njmax: maximum number of constraints per world
      nacon: number of detected contacts (across all worlds)
      nsolving: number of unconverged worlds                      (1,)
      subtree_bodyvel: subtree body velocity (ang, vel)           (nworld, nbody, 6)
    """

    solver_niter: wp.array(dtype=int)
    nl: wp.array(dtype=int)
    nefc: wp.array(dtype=int)

    time: wp.array(dtype=float)
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    act: wp.array2d(dtype=float)
    qacc_warmstart: wp.array2d(dtype=float)
    qfrc_applied: wp.array2d(dtype=float)
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)

    qacc: wp.array2d(dtype=float)
    act_dot: wp.array2d(dtype=float)

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

    site_rpos: wp.array2d(dtype=wp.vec3)
    site_xpos: wp.array2d(dtype=wp.vec3)
    site_xvel: wp.array2d(dtype=wp.vec3)
    site_diff_vec: wp.array2d(dtype=wp.vec3)
    site_diff_len: wp.array2d(dtype=float)
    site_diff_vel: wp.array2d(dtype=float)

    subtree_com: wp.array2d(dtype=wp.vec3)
    cdof: wp.array2d(dtype=wp.spatial_vector)
    cinert: wp.array2d(dtype=vec10)

    crb: wp.array2d(dtype=vec10)
    qM: wp.array3d(dtype=float)
    qLD: wp.array3d(dtype=float)
    qLDiagInv: wp.array2d(dtype=float)

    muscle_length: wp.array2d(dtype=float)
    muscle_velocity: wp.array2d(dtype=float)

    cvel: wp.array2d(dtype=wp.spatial_vector)
    cdof_dot: wp.array2d(dtype=wp.spatial_vector)

    qfrc_bias: wp.array2d(dtype=float)
    qfrc_spring: wp.array2d(dtype=float)
    qfrc_damper: wp.array2d(dtype=float)
    qfrc_passive: wp.array2d(dtype=float)

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

    #
    nworld: int
    naconmax: int
    njmax: int
    nacon: wp.array(dtype=int)
    nsolving: wp.array(dtype=int)
    subtree_bodyvel: wp.array2d(dtype=wp.spatial_vector)

    # collision driver
    collision_pair: wp.array(dtype=wp.vec2i)
    collision_pairid: wp.array(dtype=wp.vec2i)
    collision_worldid: wp.array(dtype=int)
    ncollision: wp.array(dtype=int)
