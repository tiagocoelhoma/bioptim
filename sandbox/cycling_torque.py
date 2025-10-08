"""

"""
import pickle
import platform

import numpy as np

from bioptim import (
    TorqueBiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsOptions,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    PhaseTransitionList,
    PhaseTransitionFcn,
    Solver,
    PhaseDynamics,
    ObjectiveList,
    ContactType,
)


def prepare_ocp(
        biorbd_model_path: str,
        n_shooting: int,
        final_time: float,
        q_init: list,
        max_joint_torque: float,
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    loop_from_constraint: bool
        If the looping cost should be a constraint [True] or an objective [False]
    ode_solver: OdeSolverBase
        The type of ode solver used
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=0,
        node=Node.ALL,
        target=-4,
        weight=1e3,
    )

    # DynamicsOptions
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path constraint
    x_bound = BoundsList()
    x_bound["q"] = bio_model.bounds_from_ranges("q")
    x_bound["q"].min[:, [0]] = q_init
    x_bound["q"].max[:, [0]] = q_init
    x_bound["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bound["qdot"].max[0, :] = [0]  # gear moving backwards only (negative velocity)
    x_bound["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    x_bound["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

    # Define control path constraint
    u_bound = BoundsList()
    # ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    u_bound["tau"] = ([-max_joint_torque, 0, 0, 0, 0, 0, 0], [max_joint_torque, 0, 0, 0, 0, 0, 0])
    u_bound["tau"].max[[3], [0]] = 0  # starting with negative torque right leg (so, removing the positive torque)
    u_bound["tau"].min[[4], [0]] = 0  # starting with positive torque left leg (so, removing the negative torque)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bound,
        u_bounds=u_bound,
        objective_functions=objective_functions,
    )


def main(
        biorbd_model_path: str = "models/legcyc.bioMod",
        dump_path: str = "opt_sol_results/kinematics/cycling",
        q_init: np.array = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]]),
):
    """
    Runs and animate the program
    """
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        n_shooting=140,
        final_time=2.5,
        q_init=q_init,
        max_joint_torque=50,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    # sol.animate()
    print("Initialization failure" if sol.status else "Initialization success")

    states = sol.stepwise_states()
    controls = sol.stepwise_controls()
    solver_time = sol.solver_time_to_optimize
    real_time = sol.real_time_to_optimize
    n_iterations = sol.iterations
    time = sol.stepwise_time()
    final_time = ocp.phase_time

    q_sol = states["q"]
    qdot_sol = states["qdot"]
    tau_joints = controls["tau"]

    data = {
        "q": q_sol,
        "qdot": qdot_sol,
        "tau_joints": tau_joints,
        "time": time,
        "solver_time": solver_time,
        "real_time": real_time,
        "n_iterations": n_iterations,
        "final_time": final_time,
    }

    with open(dump_path, "wb") as file:
        del sol.ocp
        pickle.dump(data, file)
        print(f"Solution dumped to {dump_path}")


if __name__ == "__main__":
    main()
