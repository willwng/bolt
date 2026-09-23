""" Allocates Data using its fields' annotations """
import warp as wp

from bolt.loader.array_util import allocate_from_annotations
from bolt.loader.array_util import dataclass_sizes
from bolt.loader.array_util import make_zero
from bolt.types import Contact
from bolt.types import Data
from bolt.types import IntegratorDotScratch
from bolt.types import IntegratorStateScratch
from bolt.types import IntegratorType
from bolt.types import Model


def get_num_scratch_states(integrator: IntegratorType) -> tuple[int, int]:
    """ Returns number of additional copies of state and state_dot required for integration """
    if integrator == IntegratorType.EULER_ADAPTIVE:
        return 2, 1
    elif integrator == IntegratorType.RK_MERSON_ADAPTIVE:
        return 2, 5
    elif integrator == IntegratorType.RK4_FIXED:
        return 1, 1  # y_0 and the weighted sum of the stage derivatives
    return 0, 0


def data_sizes(m: Model, n_worlds: int, naconmax: int) -> dict[str, int]:
    """ Names usable as array(...) dims when allocating Data """
    return dataclass_sizes(m.opt, m) | {"nworld": n_worlds, "naconmax": naconmax}


def allocate_data(m: Model, n_worlds: int, integrator: IntegratorType) -> Data:
    naconmax = max(512, n_worlds * 64)  # we're capping it at 64 contacts per world. TODO(check if this is reasonable)

    # Arrays are allocated as zeros from their types.array(...) annotations; only non-array fields and arrays whose
    # size isn't a Model/Data field are passed explicitly
    sizes = data_sizes(m, n_worlds, naconmax)
    # Scratch space for adaptive integrators
    n_int_states, n_int_dot_states = get_num_scratch_states(integrator)
    integrator_scratch = [allocate_from_annotations(IntegratorStateScratch, sizes) for _ in range(n_int_states)]
    integrator_dot_scratch = [allocate_from_annotations(IntegratorDotScratch, sizes) for _ in range(n_int_dot_states)]
    # Custom joints may need up to 6 additional vectors: [f(q), f'(q), f''(q)] for each 6 functions
    num_mob_scratch = 3 if m.njnts_cst == 0 else 6
    d = allocate_from_annotations(
        Data, sizes,
        nworld=n_worlds,
        naconmax=naconmax,
        integrator_scratch=integrator_scratch,
        integrator_dot_scratch=integrator_dot_scratch,
        contact=allocate_from_annotations(Contact, sizes),
        mob_scratch=make_zero((n_worlds, m.nbody, num_mob_scratch), dtype=wp.vec3),
    )

    # Non-zero initial values
    dt = 1.0 / 100.0  # This will be modified later by the user
    d.world_reset.fill_(True)
    d.rng_state.fill_(wp.rand_init(0))
    d.step_size.fill_(dt)
    d.actual_step_size.fill_(dt)
    d.qvel_scales.fill_(1.0)
    d.z_scales.fill_(1.0)
    d.a_act.fill_(0.5)
    d.m_excitations.fill_(0.5)
    d.a_excitations.fill_(0.5)
    return d
