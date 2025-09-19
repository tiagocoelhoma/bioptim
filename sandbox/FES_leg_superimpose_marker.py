"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""

import numpy as np

from bioptim import (
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    DynamicsOptionsList,
    PhaseDynamics,
    DynamicsOptions, ConstraintList, ObjectiveList, VariableScalingList,
)

# import the custom model
from sandbox.custom_package.fes_dynamics import CustomFesDynamics


def prepare_ocp(
    model: CustomFesDynamics,
    final_time: float,
    n_shooting: int,
    with_torque: bool = False,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 8,
    expand_dynamics: bool = True,
    **extra_params,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    model: BioModel
        The object of the custom model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    configure_dynamics: callable
        The function to configure the dynamics
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    n_threads: int
        Number of threads to use
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # DynamicsOptions
    dynamics = DynamicsOptionsList()
    dynamics.add(
        DynamicsOptions(
            ode_solver=ode_solver,
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            **extra_params,
        )
    )
    # Add constraint functions
    constraints = ConstraintList()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=0.1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=1e-1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
        first_marker="target",
        second_marker="R.Shank.Lower",
        # node=Node.END,
        weight=1e6,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    x_bounds.add("cn",
                 min_bound=[cn_min] * model.nb_muscles,
                 max_bound=[cn_max] * model.nb_muscles)
    x_bounds.add("f",
                 min_bound=[f_min] * model.nb_muscles,
                 max_bound=[f_max] * model.nb_muscles)
    for m in range(model.nb_muscles):
        x_bounds.add(f"alpha_{m}",
                     min_bound=[alpha_min] * n_pulse_trunc,
                     max_bound=[alpha_max] * n_pulse_trunc)
        x_bounds[f"alpha_{m}"][:, [0]] = alpha_init
    x_bounds["cn"][:, [0]] = 0
    x_bounds["f"][:, [0]] = 0
    joint_angle = 90
    x_bounds["q"][:, [0]] = - (180 - joint_angle) * np.pi / 180
    x_bounds["qdot"][:, [0]] = 0

    x_init = InitialGuessList()
    for m in range(model.nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc

    # Define control path constraint
    u_bounds = BoundsList()
    if with_torque:
        u_bounds.add("tau",
                     min_bound=[tau_min] * model.nb_tau,
                     max_bound=[tau_max] * model.nb_tau)
        u_bounds["tau"][:, [0]] = 0
    u_bounds.add("pw",
                 min_bound=[fes_min] * model.nb_muscles,
                 max_bound=[fes_max] * model.nb_muscles)
    u_bounds["pw"][:, [0]] = fes_min

    u_init = InitialGuessList()

    u_scaling = VariableScalingList()
    u_scaling["pw"] = [fes_max] * model.nb_muscles

    return OptimalControlProgram(
        model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        use_sx=False,
        n_threads=n_threads,
    )


# PROBLEM PARAMETERS

pd0 = 0.0001684
pdt = 0.0002979

f_min, f_max, f_init = 0, 2000, 0
cn_min, cn_max, cn_init = 0.0, 1.5, 0.0
tau_min, tau_max, tau_init = 0.00, 0.00, 0.0
fes_min, fes_max, fes_init = pd0, 800e-6, pd0  # todo: verificar scala?
alpha_min, alpha_max, alpha_init = -1.0, 2.0, 0.0
pd_offset = fes_min
stim_freq = 35  # stimulation frequency in Hertz
n_pulse_trunc = 5
phase_period = round(1 / stim_freq, 3)
n_shooting = 40

def main():
    """
        Basic FES-driven example to superimpose markers
    """
    # biorbd_model_path = "models/legcyc.bioMod"
    biorbd_model_path = "models/leg4.bioMod"
    # --- Prepare the ocp --- #
    joint_idx = 0
    extra_params = {'t_phase': phase_period,
                    'n_pulse_trunc': n_pulse_trunc,
                    'pd_offset': pd_offset,
                    'idx_crank': joint_idx,
                    'pd0': pd0,
                    'pdt': pdt,
                    # 'validation_flag': False,
                    }
    ocp = prepare_ocp(
        # model=MusclesWithExcitationsBiorbdModel(biorbd_model_path, with_residual_torque=True),
        model=CustomFesDynamics(biorbd_model_path, with_residual_torque=False, **extra_params),
        final_time=phase_period * n_shooting,
        n_shooting=n_shooting,
        **extra_params,
    )

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    # ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver=solver)
    sol.graphs()
    # --- Show results --- #
    # sol.graphs(show_bounds=True)
    sol.print_cost()

    # --- Animation --- #
    # not implemented yet
    sol.animate()


if __name__ == "__main__":
    main()
