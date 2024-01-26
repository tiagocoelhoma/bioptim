import numpy as np
import biorbd as brbd
from bioptim.limits.path_conditions import InitialGuessList

from bioptim import (
    BiorbdModel,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    BoundsList,
    Solver,
    PhaseDynamics,
    ObjectiveList,
    OptimalControlProgram,
    OdeSolverBase,
    OdeSolver,
)

def prepare_ocp(
        model_path,
        n_shooting,
        stim_duration,
        max_joint_torque,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        with_residual_torque=False,
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
):
    model = BiorbdModel(model_path)
    dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN,
                        expand_dynamics=expand_dynamics,
                        with_residual_torque=with_residual_torque,
                        phase_dynamics=phase_dynamics,
                        with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e0)

    x_bound = BoundsList()
    x_bound["q"] = model.bounds_from_ranges("q")
    x_bound["qdot"] = model.bounds_from_ranges("qdot")

    u_bound = BoundsList()
    if not with_residual_torque:
        max_joint_torque = 0
    u_bound["tau"] = ([-max_joint_torque] * model.nb_tau,
                      [max_joint_torque] * model.nb_tau)
    muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
    u_bound["muscles"] = ([muscle_min] * model.nb_muscles,
                          [muscle_max] * model.nb_muscles)
    x_init = InitialGuessList()
    u_init = InitialGuessList()

    ocp = OptimalControlProgram(
        model,
        dynamics,
        n_shooting,
        stim_duration,
        objective_functions=objective_functions,
        x_bounds=x_bound,
        u_bounds=u_bound,
        x_init=x_init,
        u_init=u_init,
        ode_solver=ode_solver,
        n_threads=14,
    )

    # numerical_model = brbd.Model(model_path)
    # ocp.add_plot(
    #     "Torque develop on the gear crank",
    #     lambda t0, phases_dt, node_idx, x, u, p, a: custom_plot_callback(x, u, numerical_model),
    #     plot_type=PlotType.INTEGRATED,
    # )
    return ocp


def main():
    # problem parameters
    model_path = "models/arm26_one_muscle_with_contact.bioMod"
    max_joint_torque = 0.001
    stim_duration = 0.5
    with_residual_torque = True
    n_shooting = int(10 + stim_duration * 20)

    ocp = prepare_ocp(model_path=model_path,
                      n_shooting=n_shooting,
                      stim_duration=stim_duration,
                      max_joint_torque=max_joint_torque,
                      with_residual_torque=with_residual_torque)

    # Solve the program
    sol = ocp.solve(solver=Solver.IPOPT(show_online_optim=False, _max_iter=500))
    sol.graphs()
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
