"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
    ControlType, Node,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    weight: float,
    n_phases: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    n_threads: int = 8,
) -> OptimalControlProgram:

    bio_model = tuple(BiorbdModel(biorbd_model_path) for _ in range(n_phases))
    n_shooting = tuple(n_shooting for _ in range(n_phases))
    final_time = tuple(final_time for _ in range(n_phases))

    # Add objective functions
    objective_functions = ObjectiveList()
    for phase_index in range(n_phases):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase_index)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", phase=phase_index)
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            key="q",
            index=0,
            node=Node.END,
            target=1,
            weight=1e4,
            phase=n_phases-1,
        )

    # Dynamics
    dynamics = DynamicsList()
    for phase_index in range(n_phases):
        dynamics.add(
            DynamicsFcn.MUSCLE_DRIVEN,
            with_residual_torque=True,
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
        )

    # Path constraint
    x_bounds = BoundsList()
    for phase_index in range(n_phases):
        x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=phase_index)
        x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=phase_index)
    x_bounds[0]["q"][:, 0] = (0.07, 1.4)
    x_bounds[0]["qdot"][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init[0]["q"] = [1.57] * bio_model[0].nb_q

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
    tau_min, tau_max, tau_init = -1.0, 1.0, 0.0
    u_bounds = BoundsList()
    for phase_index in range(n_phases):
        u_bounds.add("tau",
                     min_bound=[tau_min] * bio_model[0].nb_tau,
                     max_bound=[tau_max] * bio_model[0].nb_tau,
                     phase=phase_index)

    u_init = InitialGuessList()
    u_init[0]["muscles"] = [muscle_init] * bio_model[0].nb_muscles
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        control_type=control_type,
        n_threads=n_threads,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    ### The following line is responsible for the plot error in the end if n_shooting = 1. Otherwise, if n_shooting > 1, there is no problem.
    ocp = prepare_ocp(biorbd_model_path="models/arm26.bioMod", final_time=0.2, n_shooting=1, weight=1000, n_phases=10)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    sol.animate(show_meshes=True)


if __name__ == "__main__":
    main()
