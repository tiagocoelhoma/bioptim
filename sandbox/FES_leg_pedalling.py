"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import os
import pickle

import numpy as np

from bioptim import (
    BiorbdModel,
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
    DynamicsOptions, ConstraintList, ObjectiveList, VariableScalingList, Node, ContactType,
)
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import unpack_sol
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params

# import the custom model
from sandbox.custom_package.fes_dynamics import FESCyclingDynamics, CustomFesDynamics


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    q_init: np.array,
    with_torque: bool = False,
    with_friction: bool = False,
    max_joint_torque: float = 0.0,
    target_power: int = 0,
    target_velocity: float = 0,
    ode_solver: OdeSolverBase = OdeSolver.RK2(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 12,
    expand_dynamics: bool = True,
    **extra_params,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    model=FESCyclingDynamics(biorbd_model_path,
                             with_contact=True,
                             with_residual_torque=with_torque,
                             **extra_params)
    # model=CustomFesDynamics(biorbd_model_path,
    #                          with_contact=True,
    #                          with_residual_torque=with_torque,
    #                          **extra_params)

    idx_crank = model.name_dof.index('Gear_RotZ')
    muscles = model.muscle_names
    right_muscles = [m for m in muscles if m.endswith('_r')]
    left_muscles = [m for m in muscles if m.endswith('_l')]
    joint_idx = [model.name_dof.index('knee_r_RotZ')] * len(right_muscles) + [model.name_dof.index('knee_l_RotZ')] * len(left_muscles)
    extra_params.update({'joint_idx': joint_idx})
    extra_params.update({'idx_crank': idx_crank})

    # DynamicsOptions
    dynamics = DynamicsOptionsList()
    dynamics.add(
        DynamicsOptions(
            ode_solver=ode_solver,
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            with_contact=True,
            with_torque=with_torque,
            with_friction=with_friction,
            **extra_params,
        )
    )
    # Add constraint functions
    constraints = ConstraintList()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=1e-1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="cn", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=1e0)
    # if with_friction:
    #     objective_functions.add(
    #         FesObjective.custom_func_maximize_crank_power,
    #         custom_type=ObjectiveFcn.Mayer,
    #         node=Node.ALL_SHOOTING,
    #         index=idx_crank,
    #         target=-target_power,   # should be negative here
    #         quadratic=True,
    #         weight=1e3,
    #     )

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=idx_crank,
        node=Node.ALL,
        target=target_velocity,
        # target= dq_init[0, kni_pos_value_at],
        weight=1e3,
    )

    x_bounds = BoundsList()
    # x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    # x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    # x_bounds["qdot"].min[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # x_bounds["qdot"].max[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # setting the initial position for the legs and gear (starting from a position different from 0 or pi)
    # x_bounds["q"].min[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    # x_bounds["q"].max[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"].min[:, [0]] = q_init
    x_bounds["q"].max[:, [0]] = q_init
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"].max[0, :] = [0]  # gear moving backwards only (negative velocity)
    x_bounds["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    x_bounds["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

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
    # set to a steady position in the beginning with no forces (starting node @ phase 0)
    x_bounds["f"][:, [0]] = 0
    x_bounds["cn"][:, [0]] = 0

    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    # negative torque and positive torque would correspond to the flexor and extensor muscle activation
    if with_torque:
        u_bounds["tau"] = (
            [-max_joint_torque, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0])
        u_bounds["tau"].min[:, :] = 0
        u_bounds["tau"].max[:, :] = 0
        u_bounds["tau"].min[[0], [0]] = -max_joint_torque
    u_bounds.add("pw",
                 min_bound=[fes_min] * model.nb_muscles,
                 max_bound=[fes_max] * model.nb_muscles)
    x_init = InitialGuessList()
    # x_init.add('q', q_init, InterpolationType.EACH_FRAME)
    # x_init.add('qdot', dq_init, InterpolationType.EACH_FRAME)
    for m in range(model.nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc

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

def find_nearest(q_pos, value, idx_dof=0):
    array = np.asarray(q_pos[idx_dof,:] % (-2 * np.pi))
    array = np.asarray(q_pos[idx_dof,:] % (2 * np.pi)) if q_pos[idx_dof,0] > 0 else array
    idx = (np.abs(array - value)).argmin()
    return q_pos[:,idx, np.newaxis]


def get_initial_position(crank_angle=80):
    # dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/Fes_leg_pedalling'
    from sandbox.cycling_torque import main as main_torque
    if not os.path.isfile(dump_path):
        print(f"Initializing kinematics. Remember to update this file when the biomechanical model changes!")
        biorbd_model_path = 'models/legcyc.bioMod'
        q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
        main_torque(biorbd_model_path, dump_path, q_init)
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T

    q_init = find_nearest(qvec[:, 0:25],
                         value=-((crank_angle / 180) * np.pi),
                         idx_dof=0)

    return q_init



# PROBLEM PARAMETERS
f_min, f_max, f_init = 0, 2000, 0
cn_min, cn_max, cn_init = 0.0, 1.5, 0.0
tau_min, tau_max, tau_init = 0.00, 0.00, 0.0
# fes_min, fes_max, fes_init = 86e-6, 1600e-6, 86e-6  # todo: verificar scala?
alpha_min, alpha_max, alpha_init = -1.0, 2.0, 0.0
# pd_offset = fes_min
stim_freq = 35  # stimulation frequency in Hertz
n_pulse_trunc = 5
phase_period = round(1 / stim_freq, 3)
# pd0 = 1.11878174e-04
# pdt = 5.69997838e-04

pd0 = 0.0001684
pdt = 0.0002979
fes_min, fes_max, fes_init = pd0, 1600e-6, pd0  # todo: verificar scala?


def main():
    """
        Basic FES-driven example to superimpose markers
    """
    biorbd_model_path = "models/legcyc.bioMod"
    # biorbd_model_path = "models/leg4.bioMod"
    # --- Prepare the ocp --- #

    filename = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/cycling_muscles_params.pkl'
    muscle_params = read_params(biorbd_model_path, filename)

    for m, muscle_name in enumerate(muscle_params):
        vec = muscle_params[muscle_name][0]
        if len(vec) < 6:
            padded_vec = np.pad(vec, (6 - len(vec), 0), mode='constant', constant_values=0)
            muscle_params[muscle_name][0] = padded_vec.tolist()

    # filename = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/pd0ptt.pkl'
    # muscle_params_pd0_pdt = read_params(biorbd_model_path, filename)

    friction_coef_b=0.78
    friction_coef_mu=0.05
    target_velocity_rpm=41.92
    target_power=25.0
    target_v = (-(target_velocity_rpm * 2 * np.pi) / 60)  # rpm to rad/s
    torque_crank = target_power / target_v
    friction_coef_b = round(torque_crank / target_v, 4)

    with_residual_torque = False
    n_shooting = 15
    max_joint_torque = 50
    extra_params = {'t_phase': phase_period,
                    'n_pulse_trunc': n_pulse_trunc,
                    # 'pd_offset': pd_offset,
                    'muscle_params': muscle_params,
                    'pd0': pd0,
                    'pdt': pdt,
                    'friction_coef_mu': friction_coef_mu,
                    'friction_coef_b': friction_coef_b,
                    }
    q_init = get_initial_position(crank_angle=80)
    ocp = prepare_ocp(
        biorbd_model_path,
        final_time=phase_period * n_shooting,
        max_joint_torque=max_joint_torque,
        with_torque=with_residual_torque,
        n_shooting=n_shooting,
        q_init=q_init,
        target_power=target_power,
        target_velocity=target_v,
        **extra_params,
    )

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    # ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=False, _tol=1e-3)
    solver.set_maximum_iterations(1000)
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
