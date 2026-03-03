# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
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

from .math import upper_trid_index
from .types import GeomType

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})

MULTI_CONTACT_COUNT = 8
mat3c = wp.types.matrix(shape=(MULTI_CONTACT_COUNT, 3), dtype=float)
mat63 = wp.types.matrix(shape=(6, 3), dtype=float)

_CONVEX_COLLISION_PAIRS = [
  (GeomType.HFIELD, GeomType.SPHERE),
  (GeomType.HFIELD, GeomType.CAPSULE),
  (GeomType.HFIELD, GeomType.ELLIPSOID),
  (GeomType.HFIELD, GeomType.CYLINDER),
  (GeomType.HFIELD, GeomType.BOX),
  (GeomType.HFIELD, GeomType.MESH),
  (GeomType.SPHERE, GeomType.ELLIPSOID),
  (GeomType.SPHERE, GeomType.MESH),
  (GeomType.CAPSULE, GeomType.ELLIPSOID),
  (GeomType.CAPSULE, GeomType.CYLINDER),
  (GeomType.CAPSULE, GeomType.MESH),
  (GeomType.ELLIPSOID, GeomType.ELLIPSOID),
  (GeomType.ELLIPSOID, GeomType.CYLINDER),
  (GeomType.ELLIPSOID, GeomType.BOX),
  (GeomType.ELLIPSOID, GeomType.MESH),
  (GeomType.CYLINDER, GeomType.CYLINDER),
  (GeomType.CYLINDER, GeomType.BOX),
  (GeomType.CYLINDER, GeomType.MESH),
  (GeomType.BOX, GeomType.MESH),
  (GeomType.MESH, GeomType.MESH),
]


def _check_convex_collision_pairs():
  prev_idx = -1
  for pair in _CONVEX_COLLISION_PAIRS:
    idx = upper_trid_index(len(GeomType), pair[0].value, pair[1].value)
    if pair[1] < pair[0] or idx <= prev_idx:
      return False
    prev_idx = idx
  return True


assert _check_convex_collision_pairs(), "_CONVEX_COLLISION_PAIRS is in invalid order."