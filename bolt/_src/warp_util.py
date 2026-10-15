# Copyright 2025 The Newton Developers
# Modified for Bolt by Will Wang
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

import functools
import inspect
import math

import warp as wp

_STACK = None

# Smallest number of blocks per SM that launch() aims for before shrinking blocks
_MIN_BLOCKS_PER_SM = 2
_MIN_BLOCK_DIM, _MAX_BLOCK_DIM = 32, 256

# When trees have greater bodies than this amount at a single level, parallelize the tree pass per level
_TREE_PASS_PER_WORLD_MAX_WIDTH = 2


def block_dim_for(nthreads: int, device: wp.Device) -> int:
    """
    Largest power-of-two block size in [32, 256] that still gives every SM at least _MIN_BLOCKS_PER_SM blocks.
    Small grids may otherwise run on only a few SMs with the default of 256
    """
    per_block = nthreads // (_MIN_BLOCKS_PER_SM * device.sm_count)
    if per_block < _MIN_BLOCK_DIM:
        return _MIN_BLOCK_DIM
    return min(_MAX_BLOCK_DIM, 1 << int(math.log2(per_block)))


def tree_pass_per_world(body_tree: tuple[wp.array, ...]) -> bool:
    """
    Whether heavy tree passes (articulated inertia, udot) run as one thread per world walking the whole tree,
        instead of one launch per tree level
    """
    return max(level.size for level in body_tree) <= _TREE_PASS_PER_WORLD_MAX_WIDTH


def launch(kernel, dim, *args, **kwargs):
    """ wp.launch that picks block_dim from the grid size on CUDA devices, unless one is given """
    device = wp.get_device(kwargs.get("device"))
    if device.is_cuda and kwargs.get("block_dim") is None:
        nthreads = math.prod(dim) if isinstance(dim, (tuple, list)) else int(dim)
        kwargs["block_dim"] = block_dim_for(nthreads, device)
    return wp.launch(kernel, dim, *args, **kwargs)


class EventTracer:
    """Calculates elapsed times of functions annotated with `event_scope`.

    Use as a context manager like so:

      @event_trace
      def my_warp_function(...):
        ...

      with EventTracer() as tracer:
        my_warp_function(...)
        print(tracer.trace())
    """

    def __init__(self, enabled: bool = True):
        global _STACK
        if _STACK is not None:
            raise ValueError("only one EventTracer can run at a time")
        if enabled:
            _STACK = {}

    def __enter__(self):
        return self

    def trace(self) -> dict:
        """Calculates elapsed times for every node of the trace."""
        global _STACK

        if _STACK is None:
            return {}

        ret = {}

        for k, v in _STACK.items():
            events, sub_stack = v
            # push into next level of stack
            saved_stack, _STACK = _STACK, sub_stack
            sub_trace = self.trace()
            # pop!
            _STACK = saved_stack
            events = tuple(
                wp.get_event_elapsed_time(beg, end) for beg, end in events)
            ret[k] = (events, sub_trace)

        return ret

    def __exit__(self, type, value, traceback):
        global _STACK
        _STACK = None


def _merge(a: dict, b: dict) -> dict:
    """Merges two event trace stacks."""
    ret = {}
    if not a or not b:
        return dict(**a, **b)
    if set(a) != set(b):
        raise ValueError("incompatible stacks")
    for key in a:
        a1_events, a1_substack = a[key]
        a2_events, a2_substack = b[key]
        ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
    return ret


def event_scope(fn, name: str = ""):
    """Wraps a function and records an event before and after the function invocation."""
    name = name or getattr(fn, "__name__")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _STACK
        if _STACK is None:
            return fn(*args, **kwargs)

        for frame_info in inspect.stack():
            if frame_info.function in ("capture_while", "capture_if"):
                return fn(*args, **kwargs)

        # push into next level of stack
        saved_stack, _STACK = _STACK, {}
        beg = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        wp.record_event(beg)
        res = fn(*args, **kwargs)
        wp.record_event(end)
        # pop back up to current level
        sub_stack, _STACK = _STACK, saved_stack
        # append events and substack
        prev_events, prev_substack = _STACK.get(name, ((), {}))
        events = prev_events + ((beg, end),)
        sub_stack = _merge(prev_substack, sub_stack)
        _STACK[name] = (events, sub_stack)
        return res

    return wrapper
