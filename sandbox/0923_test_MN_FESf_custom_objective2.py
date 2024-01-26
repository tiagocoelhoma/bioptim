"""

"""
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    PenaltyController,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    CostType,
    MultinodeObjectiveList, PhaseTransitionList, PhaseTransitionFcn, MultinodeConstraintList, MultinodeConstraintFcn,
    NonLinearProgram, DynamicsEvaluation, DynamicsFunctions, ConfigureProblem, Constraint, Axis, InitialGuessList,
    InterpolationType, ParameterList, ParameterObjectiveList, PhaseDynamics,
)
import numpy as np

from custom_package.fes_dynamics import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective

# PROBLEM PARAMETERS
# t_stim = 1.0  # stimulation period (total)
t_phase = round(1 / 50, 3)  # period a stimulation pulse
n_nodes = 1
f_min, f_max, f_init = 0, 1000, 0
cn_min, cn_max, cn_init = 0.0, 2.0, 0.0
tau_min, tau_max, tau_init = -0.0, 0.0, 0.0
fes_min, fes_max, fes_init = 0.0002, 0.0008, 0.0002  # todo: verificar scala?
t_stim = 10 * t_phase  # stimulation period (total)


def prepare_ocp(
        biorbd_model_path: str,
        ode_solver: OdeSolverBase = OdeSolver.RK8(),
        phase_period: float = 0,
        n_phase: int = 0,
        n_shooting: int = 0,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:

    stim = [1] * n_phase + [0] * n_phase
    bio_model = tuple(BiorbdModel(biorbd_model_path) for _ in range(n_phase))
    n_shoot = tuple(n_shooting for _ in range(n_phase))
    final_time = [phase_period] * n_phase

    # Dynamics
    dynamics = DynamicsList()
    for phase_index in range(n_phase):
        extra_parameters = {'phase_index': phase_index, 'stim': stim, 't_phase': t_phase}
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
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0, phase=i)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e0, phase=i)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES, target=[20], weight=1e2, phase=i)

    # Path constraint
    x_bounds = BoundsList()
    for i in range(n_phase):
        x_bounds.add("q", bounds=bio_model[i].bounds_from_ranges("q"), phase=i)
        x_bounds.add("qdot", bounds=bio_model[i].bounds_from_ranges("qdot"), phase=i)
        x_bounds.add("cn",
                     min_bound=[cn_min] * bio_model[i].nb_muscles,
                     max_bound=[cn_max] * bio_model[i].nb_muscles,
                     phase=i)
        x_bounds.add("f",
                     min_bound=[f_min] * bio_model[i].nb_muscles,
                     max_bound=[f_max] * bio_model[i].nb_muscles,
                     phase=i)

    x_init = InitialGuessList()
    x_init.add("cn", [cn_init] * bio_model[0].nb_muscles)
    x_init.add("f", [f_init] * bio_model[0].nb_muscles)

    # Define control path constraint
    u_bounds = BoundsList()
    for i in range(n_phase):
        u_bounds.add("tau",
                     min_bound=[tau_min] * bio_model[i].nb_tau,
                     max_bound=[tau_max] * bio_model[i].nb_tau,
                     phase=i)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shoot,
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


def main():
    # --- Prepare the optimal control program --- #
    model_path = "models/arm26_one_muscle.bioMod"
    n_phases = int(t_stim / t_phase)  # number of stimulation corresponding to phases

    ocp = prepare_ocp(biorbd_model_path=model_path,
                      phase_period=t_phase,
                      n_shooting=n_nodes,
                      n_phase=n_phases)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.graphs()

    # --- Show results --- #
    # sol.print_cost()
    sol.animate()


if __name__ == "__main__":
    main()
