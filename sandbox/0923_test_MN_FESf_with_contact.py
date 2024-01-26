"""

"""
import numpy as np
from casadi import MX

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    InitialGuessList,
    PhaseDynamics,
)

from custom_package.fes_dynamics import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective
from custom_package.fes_plotting import fes_plot_callback

# PROBLEM PARAMETERS
f = 50  # stimulation frequency in Hertz
t_phase = round(1 / f, 3)  # period a stimulation pulse
n_nodes = 2
f_min, f_max, f_init = 0, 1000, 0
cn_min, cn_max, cn_init = 0.0, 1.4, 0.0
tau_min, tau_max, tau_init = -0.00, 0.00, 0.0
fes_min, fes_max, fes_init = 0.000086, 0.0012, 0.000086
ascale_min, ascale_max, ascale_init = 500, 500000, 1000
t_stim = 25 * t_phase


def prepare_ocp(
        biorbd_model_path: str,
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
        phase_period: float = 0,
        n_phase: int = 0,
        n_shooting: int = 0,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:

    stim = [1] * n_phase
    bio_model = tuple(BiorbdModel(biorbd_model_path) for _ in range(n_phase))
    n_shooting = tuple(n_shooting for _ in range(n_phase))
    final_time = [phase_period] * n_phase

    dynamics = DynamicsList()
    for phase_index in range(n_phase):
        extra_parameters = {'phase_index': phase_index,
                            'stim': stim,
                            't_phase': t_phase,
                            }
        dynamics.add(FesDynamicsFcn.FES_DRIVEN,
                     expand=False,
                     with_contact=True,
                     phase_dynamics=phase_dynamics,
                     **extra_parameters)

    # Constraints
    constraints = ConstraintList()
    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(n_phase):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1, phase=i)
        objective_functions.add(ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
                                first_marker="target",
                                second_marker="R.Shank.Lower",
                                weight=1e4,
                                phase=i)
        objective_functions.add(
            FesObjective.custom_func_track_torque,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.ALL_SHOOTING,
            # all node expect the last one (Node.END) as this is related to the control i think.
            target=np.ones((bio_model[i].nb_tau, n_shooting[i])) * 50.0,  # automatically handled for every cost functions
            quadratic=True,
            weight=1e3,
            phase=i,
        )
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES, target=[1000], weight=1e4)


    # Path constraint
    x_bounds = BoundsList()
    for phase_index in range(n_phase):
        x_bounds.add("q", bounds=bio_model[phase_index].bounds_from_ranges("q"), phase=phase_index)
        x_bounds.add("qdot", bounds=bio_model[phase_index].bounds_from_ranges("qdot"), phase=phase_index)
        x_bounds.add("cn",
                     min_bound=[cn_min] * bio_model[phase_index].nb_muscles,
                     max_bound=[cn_max] * bio_model[phase_index].nb_muscles,
                     phase=phase_index)
        x_bounds.add("f",
                     min_bound=[f_min] * bio_model[phase_index].nb_muscles,
                     max_bound=[f_max] * bio_model[phase_index].nb_muscles,
                     phase=phase_index)
        x_bounds.add("t",
                     min_bound=[phase_period * (phase_index - 1)] * bio_model[phase_index].nb_muscles,
                     max_bound=[phase_period * phase_index] * bio_model[phase_index].nb_muscles,
                     phase=phase_index)
    # set to a steady position in the beginning with no forces (starting node @ phase 0)
    x_bounds[0]["f"][:, [0]] = 0
    x_bounds[0]["cn"][:, [0]] = 0
    x_bounds[0]["q"][:, [0]] = x_bounds[0]["q"].min[0][0]

    x_init = InitialGuessList()

    # Define control path constraint
    u_bounds = BoundsList()
    for i in range(n_phase):
        u_bounds.add("tau",
                     min_bound=[tau_min] * bio_model[i].nb_tau,
                     max_bound=[tau_max] * bio_model[i].nb_tau,
                     phase=i)
        u_bounds.add("pw",
                     min_bound=[fes_min] * bio_model[i].nb_muscles,
                     max_bound=[fes_max] * bio_model[i].nb_muscles,
                     phase=i)
    u_bounds[0]["tau"][:, [0]] = 0
    u_bounds[0]["pw"][:, [0]] = 0

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)
    u_init.add("pw", [fes_init] * bio_model[0].nb_muscles, phase=0)

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=10,
    )

    # numerical_model = biorbd.Model(biorbd_model_path)
    # for i in range(n_phase + n_phase_off):
    #     ocp.add_plot("Muscular joint torque",
    #                  lambda t, x, u, p, s: fes_plot_callback(x, numerical_model),
    #                  plot_type=PlotType.INTEGRATED,
    #                  phase=i)
    return ocp


def main():
    model_path = "models/arm26_one_muscle_with_contact.bioMod"
    n_phases = int(t_stim / t_phase)  # number of stimulation corresponding to phases

    ocp = prepare_ocp(biorbd_model_path=model_path,
                      phase_period=t_phase,
                      n_shooting=n_nodes,
                      n_phase=n_phases)

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500))
    # sol = ocp.solve()
    # sol.print_cost()
    sol.graphs()
    # --- Show results --- #
    sol.animate()
    # sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
