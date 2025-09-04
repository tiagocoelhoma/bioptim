"""

"""
import os, platform
import pickle

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
    MusclesBiorbdModel, ControlType, InitialGuessList,
)

from cycling_torque import main as main_torque

def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    q_init: np.ndarray,
    dq_init: np.ndarray,
    max_joint_torque: float,
    with_residual_torque: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK2(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    n_threads: int = 8,
) -> OptimalControlProgram:

    bio_model = MusclesBiorbdModel(biorbd_model_path,
                                   with_residual_torque=with_residual_torque,
                                   contact_types = [ContactType.RIGID_EXPLICIT])

    # Add objective functions
    objective_functions = ObjectiveList()
    if with_residual_torque:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=0,
        node=Node.ALL,
        target=-1,
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
    x_bound["qdot"].min[:, [0]] = dq_init  # Initial velocity = 0 so the contact points would be still
    x_bound["qdot"].max[:, [0]] = dq_init  # Initial velocity = 0 so the contact points would be still
    # x_bound["qdot"].max[0, :] = [0]  # gear moving backwards only (negative velocity)
    # x_bound["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    # x_bound["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

    # Define control path constraint
    u_bound = BoundsList()
    if with_residual_torque:
        # ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
        u_bound["tau"] = ([-max_joint_torque, 0, 0, 0, 0, 0, 0], [max_joint_torque, 0, 0, 0, 0, 0, 0])
        u_bound["tau"].max[[3], [0]] = 0  # starting with negative torque right leg (so, removing the positive torque)
        u_bound["tau"].min[[4], [0]] = 0  # starting with positive torque left leg (so, removing the negative torque)
    muscle_min, muscle_max, muscle_init = 0.0, 10.0, 0.0
    u_bound["muscles"] = ([muscle_min] * bio_model.nb_muscles, [muscle_max] * bio_model.nb_muscles)

    x_init = InitialGuessList()
    x_init["q"] = q_init

    u_init = InitialGuessList()

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bound,
        u_bounds=u_bound,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        control_type=control_type,
        n_threads=n_threads,
    )

def unpack_sol(list):
    out = []
    for item in list:
        out.append([it[0] for it in item])
    return out

def main():
    """
    Runs and animate the program
    """
    # q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
    kni_pos_value_at = 2

    duration, nb_shooting, target_vel, target_power = 1.50, 50, 42, 10
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/cycling'
    if not os.path.isfile(dump_path):
        print(f"Intializing kinematics...")
        main_torque()
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T
    dqvec = np.array(unpack_sol(sol_dict["qdot"])).T

    q_init = qvec[:, kni_pos_value_at, np.newaxis]
    dq_init = dqvec[:, kni_pos_value_at, np.newaxis]
    ocp = prepare_ocp("models/legcyc.bioMod",
                      n_shooting=20,
                      final_time=0.5,
                      with_residual_torque=False,
                      q_init=q_init,
                      dq_init=dq_init,
                      max_joint_torque=50,
                      )

    # --- Solve the program --- #
    # sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
