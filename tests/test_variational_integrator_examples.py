"""
Tests of the examples of the variational integrator.
"""

import numpy as np
import os
import pytest
from bioptim import Solver


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_pendulum(use_sx):
    """Test the variational integrator pendulum example"""
    from bioptim.examples.discrete_mechanics_and_optimal_control import example_variational_integrator_pendulum

    bioptim_folder = os.path.dirname(example_variational_integrator_pendulum.__file__)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q"][:, 0].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, 10].squeeze(),
        [-0.325653795765479, 0.514317755981177],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, -1].squeeze(),
        [0.0, 3.14],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 0].squeeze(),
        [9.952650040830257, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 10].squeeze(),
        [1.326124391015805, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, -2].squeeze(),
        [-24.871395482788490, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot_start"].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot_end"].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_pendulum_with_holonomic_constraints(use_sx):
    """Test the variational integrator pendulum with holonomic constraints example"""
    from bioptim.examples.discrete_mechanics_and_optimal_control import (
        example_variational_integrator_with_holonomic_constraints_pendulum,
    )

    bioptim_folder = os.path.dirname(example_variational_integrator_with_holonomic_constraints_pendulum.__file__)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_with_holonomic_constraints_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/models/pendulum_holonomic.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q"][:, 0].squeeze(),
        [0.0, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, 10].squeeze(),
        [-5.31812533e-01, -3.45378660e-14,  7.06503085e-01],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, -1].squeeze(),
        [0.0, 0.0, 3.14],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 0].squeeze(),
        [10.52248442,  0.        ,  0.        ],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 10].squeeze(),
        [12.73929061,  0.        ,  0.        ],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, -2].squeeze(),
        [-19.06595356,   0.        ,   0.        ],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot_start"].squeeze(),
        [0.01000001, 0.01000001, 0.01000001],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot_end"].squeeze(),
        [-0.01000001, -0.01      ,  0.01000001],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        sol.states["lambdas"][:, 0].squeeze(),
        [-16.06577635],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        sol.states["lambdas"][:, -1].squeeze(),
        [7.59354865],
        decimal=6,
    )

