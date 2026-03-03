# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import warp as wp

from . import support
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _joint_moments_kernel(
        # Data in:
        joint_moments_in: wp.array2d(dtype=float),
        qfrc_bias_in: wp.array2d(dtype=float),
        qfrc_contact_in: wp.array2d(dtype=float),
        qfrc_drag_in: wp.array2d(dtype=float),
        # Data out:
        joint_moments_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    joint_moments_out[worldid, dofid] = (
            joint_moments_in[worldid, dofid]
            + qfrc_bias_in[worldid, dofid]
            - qfrc_contact_in[worldid, dofid]
            - qfrc_drag_in[worldid, dofid]
    )


@event_scope
def compute_joint_moments(m: Model, d: Data):
    d.joint_moments.zero_()
    support.mul_m(m, d, d.joint_moments, d.qacc)
    wp.launch(
        _joint_moments_kernel,
        dim=(d.nworld, m.nv),
        inputs=[
            d.joint_moments,
            d.qfrc_bias,
            d.qfrc_contact,
            d.qfrc_drag,
        ],
        outputs=[d.joint_moments],
    )
    return
