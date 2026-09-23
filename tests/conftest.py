"""
Pytest configuration and shared fixtures.

Tests compare Bolt against OpenSim over batches of random states, with one state per Bolt world
"""

from __future__ import annotations

import numpy as np
import opensim as osim
import pytest
import warp as wp

import bolt
import opensim_oracle

wp.config.log_level = wp.LOG_WARNING
osim.Logger.setLevelString("error")

N_STATES = 16


class Case:
    """ A Bolt model loaded with N_STATES worlds, plus its OpenSim counterpart """

    def __init__(self, path: str, muscle_fn_path: str | None = None):
        self.path = path
        self.load_result = bolt.load_model(
            model_path=path,
            n_worlds=N_STATES,
            integrator=bolt.IntegratorType.EULER_FIXED,
            requires_visuals=False,
            muscle_fn_path=muscle_fn_path,
        )
        self.m, self.d = self.load_result.model, self.load_result.data
        self.osim_model = osim.Model(path)
        self.osim_state = self.osim_model.initSystem()

    def set_states(self, q: np.ndarray, u: np.ndarray):
        """ Sets each Bolt world i to the OpenSim state (q[i], u[i]) """
        qpos, qvel = zip(*[
            opensim_oracle.bolt_state(self.osim_model, self.osim_state, self.load_result, qi, ui)
            for qi, ui in zip(q, u)
        ])
        self.d.qpos.assign(np.array(qpos, dtype=np.float32))
        self.d.qvel.assign(np.array(qvel, dtype=np.float32))


_CACHE: dict[tuple, Case] = {}


@pytest.fixture(scope="session")
def load_case():
    """ Loads (and caches) a Case, since model loading dominates test time """

    def _load(path: str, muscle_fn_path: str | None = None) -> Case:
        key = (path, muscle_fn_path)
        if key not in _CACHE:
            _CACHE[key] = Case(path, muscle_fn_path)
        return _CACHE[key]

    return _load
