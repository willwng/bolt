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

import time
from typing import Callable, Optional, Tuple
from tqdm import tqdm

import numpy as np
import warp as wp

from bolt._src import warp_util
from bolt._src.types import Data
from bolt._src.types import Model


def _sum(stack1, stack2):
    ret = {}
    for k in stack1:
        times1, sub_stack1 = stack1[k]
        times2, sub_stack2 = stack2[k]
        times = [t1 + t2 for t1, t2 in zip(times1, times2)]
        ret[k] = (times, _sum(sub_stack1, sub_stack2))
    return ret


def benchmark(
        fn: Callable[[Model, Data, float, float], None],
        m: Model,
        d: Data,
        dt: float,
        nstep: int,
        event_trace: bool = False,
        measure_alloc: bool = False,
) -> Tuple[float, float, dict, list, int]:
    """Benchmark a function of Model and Data.

    Args:
      fn: Function to benchmark.
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output information (device).
      dt: Timestep.
      nstep: Number of timesteps.
      event_trace: If True, time routines decorated with @event_scope.
      measure_alloc: If True, record number of contacts and constraints.

    Returns:
      - Time to JIT fn.
      - Total time to run the benchmark.
      - Trace.
      - Number of contacts.
      - Number of constraints.
      - Number of solver iterations.
      - Number of converged worlds.
    """
    trace = {}
    nacon, nefc = [], []

    with warp_util.EventTracer(enabled=event_trace) as tracer:
        # capture the whole function as a CUDA graph
        jit_beg = time.perf_counter()
        with wp.ScopedCapture() as capture:
            fn(m, d, dt)
        jit_end = time.perf_counter()
        jit_duration = jit_end - jit_beg

        graph = capture.graph

        time_vec = np.zeros(nstep)
        for i in tqdm(range(nstep)):
            with wp.ScopedStream(wp.get_stream()):
                wp.synchronize()

                run_beg = time.perf_counter()
                wp.capture_launch(graph)
                wp.synchronize()
                run_end = time.perf_counter()

            time_vec[i] = run_end - run_beg
            if trace:
                trace = _sum(trace, tracer.trace())
            else:
                trace = tracer.trace()
            if measure_alloc:
                nacon.append(
                    np.max([d.nacon.numpy()[0], d.ncollision.numpy()[0]]))

        nsuccess = np.sum(~np.any(np.isnan(d.qpos.numpy()), axis=1))
        run_duration = np.sum(time_vec)

    return jit_duration, run_duration, trace, nacon, nsuccess
