"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import os
import pickle
# import biorbd
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
    DynamicsOptions, ConstraintList, ObjectiveList, VariableScalingList, Node, ContactType, InterpolationType,
)
from sandbox.FES_leg_pedalling_calculate_power import plot_power, calculate_power
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import unpack_sol
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params
from sandbox.Param_iden_cycling_Brazil.FES_leg_param_ident_pdpdt_plot import main as get_pd0pdt

# import the custom model
from sandbox.custom_package.fes_dynamics import FESCyclingDynamics, CustomFesDynamics
from sandbox.custom_package.fes_objectives import FesObjective


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    q_init: np.array,
    dq_init: np.array,
    with_torque: bool = False,
    with_friction: bool = False,
    max_joint_torque: float = 0.0,
    target_power: int = 0,
    fes_min: float = 0.0,
    fes_max: float = 0.0,
    idx: int =0,
    target_velocity: float = 0,
    ode_solver: OdeSolverBase = OdeSolver.RK2(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 14,
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
                             with_friction=with_friction,
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
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=1e3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="cn", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=0.2)
    # if with_friction:
    #     objective_functions.add(
    #         FesObjective.custom_func_maximize_crank_power,
    #         custom_type=ObjectiveFcn.Mayer,
    #         node=Node.ALL,
    #         index=idx_crank,
    #         target=target_power,   # should be negative here
    #         quadratic=True,
    #         weight=1e2,
    #         **extra_params,
    #     )

    if 'rect_fem_r' in muscles:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscles.index('rect_fem_r'),
            mus_num_2=muscles.index('rect_fem_l'),
            pw_offset=fes_min,
            quadratic=True,
            weight=1e5,
        )
    if 'vasmed_r' in muscles:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscles.index('vasmed_r'),
            mus_num_2=muscles.index('vasmed_l'),
            pw_offset=fes_min,
            quadratic=True,
            weight=1e5,
        )
    if 'bflh140_r' in muscles:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscles.index('bflh140_r'),
            mus_num_2=muscles.index('bflh140_l'),
            pw_offset=fes_min,
            quadratic=True,
            weight=1e5,
        )
    # # todo: verify if Lagrange objective is better while trying to achieve lower co-contraction
    objective_functions.add(
        # ObjectiveFcn.Lagrange.TRACK_STATE,
        ObjectiveFcn.Mayer.TRACK_STATE,
        node=Node.ALL,
        key="qdot",
        index=idx_crank,
        target= target_velocity,
        weight=1e2,
    )

    x_bounds = BoundsList()
    x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    x_bounds["qdot"].min[:, [0]] = dq_init[:, idx, np.newaxis]  # so the contact points would be still
    x_bounds["qdot"].max[:, [0]] = dq_init[:, idx, np.newaxis]  # so the contact points would be still
    # setting the initial position for the legs and gear (starting from a position different from 0 or pi)
    x_bounds["q"].min[:, [0]] = q_init[:, idx, np.newaxis]
    x_bounds["q"].max[:, [0]] = q_init[:, idx, np.newaxis]
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
    x_init.add('q', q_init[:, :n_shooting+1], InterpolationType.EACH_FRAME)
    x_init.add('qdot', dq_init[:, :n_shooting+1], InterpolationType.EACH_FRAME)
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
        u_scaling=u_scaling,
        use_sx=False,
        n_threads=n_threads,
    )

def find_nearest(q_pos, value, idx_dof=0):
    array = np.asarray(q_pos[idx_dof,:] % (-2 * np.pi))
    array = np.asarray(q_pos[idx_dof,:] % (2 * np.pi)) if q_pos[idx_dof,0] > 0 else array
    idx = (np.abs(array - value)).argmin()
    return idx


def get_initial_position(biorbd_model_path='models/legcyc.bioMod', crank_angle=80):
    # dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/Fes_leg_pedalling'

    from sandbox.cycling_torque import main as main_torque
    if not os.path.isfile(dump_path):
        print(f"Initializing kinematics. Remember to update this file when the biomechanical model changes!")
        q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
        main_torque(biorbd_model_path, dump_path, q_init)
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T
    dqvec = np.array(unpack_sol(sol_dict["qdot"])).T

    model = BiorbdModel(biorbd_model_path)
    # index = [s.to_string() for s in model.name_dof()].index('knee_r_RotZ')
    index = model.name_dof.index('knee_r_RotZ')
    idx = find_nearest(qvec[:, 0:25],
                         value=-((crank_angle / 180) * np.pi),
                         idx_dof=0)

    return qvec, dqvec, [float(qvec[index].min()), float(qvec[index].max())], idx



# PROBLEM PARAMETERS
f_min, f_max, f_init = 0, 5000, 0
cn_min, cn_max, cn_init = 0.0, 1.5, 0.0
tau_min, tau_max, tau_init = 0.00, 0.00, 0.0
# fes_min, fes_max, fes_init = 86e-6, 1600e-6, 86e-6  # todo: verificar scala?
alpha_min, alpha_max, alpha_init = -1.0, 2.0, 0.0
# pd_offset = fes_min
stim_freq = 35  # stimulation frequency in Hertz
n_pulse_trunc = 5
phase_period = round(1 / stim_freq, 3)


def main():
    """
        Basic FES-driven example to superimpose markers
    """
    biorbd_model_path = "models/legcyc.bioMod"
    # biorbd_model_path = "models/leg4.bioMod"
    # --- Prepare the ocp --- #

    filename = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/cycling_muscles_params.pkl'
    muscle_params = read_params(biorbd_model_path, filename)

    a, pd0, pdt = get_pd0pdt(print_flag=False)
    print(f'using pd0 and pdt average values. pd0 = {pd0} and pdt = {pdt}')
    fes_min, fes_max, fes_init = pd0, 1600e-6, pd0  # todo: verificar scala?

    for m, muscle_name in enumerate(muscle_params):
        vec = muscle_params[muscle_name][0]
        if len(vec) < 6:
            padded_vec = np.pad(vec, (6 - len(vec), 0), mode='constant', constant_values=0)
            muscle_params[muscle_name][0] = padded_vec.tolist()

    # filename = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/pd0ptt.pkl'
    # muscle_params_pd0_pdt = read_params(biorbd_model_path, filename)

    friction_coef_b=0.78
    friction_coef_mu=0.02
    target_velocity_rpm=41.92
    target_power=20.0
    target_v = (-(target_velocity_rpm * 2 * np.pi) / 60)  # rpm to rad/s
    torque_crank = target_power / target_v
    friction_coef_b = round(torque_crank / target_v, 4)

    with_residual_torque = True
    with_friction = True
    n_shooting = 75
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

    # Plotting the A function, because if its results extrapolate beyond realistic values,
    # it is highly likely to cause problems.
    q_init, dq_init, knee_range, idx = get_initial_position(biorbd_model_path, crank_angle=50)

    print_flag = True
    if print_flag:
        from Param_iden_cycling_Brazil.statistical_analysis.plot_a_function import main as plot_a_function
        plot_a_function(biorbd_model_path, muscle_params, knee_range)

    ocp_2B_calculated = True
    power_2B_calculated = True

    folder_path = 'Param_iden_cycling_Brazil/opt_sol_results/fes_leg_pedalling/'
    filename = f'friction_{round(phase_period * n_shooting, 2)}s_{n_shooting}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}'

    if ocp_2B_calculated:
        ocp = prepare_ocp(
            biorbd_model_path,
            final_time=phase_period * n_shooting,
            max_joint_torque=max_joint_torque,
            with_torque=with_residual_torque,
            with_friction=with_friction,
            n_shooting=n_shooting,
            q_init=q_init,
            dq_init=dq_init,
            target_power=target_power,
            target_velocity=target_v,
            fes_min=fes_min,
            fes_max=fes_max,
            idx=idx,
            **extra_params,
        )

        # Custom plots
        # ocp.add_plot_penalty(CostType.ALL)

        # --- Print ocp structure --- #
        # ocp.print(to_console=True, to_graph=False)

        # --- Solve the ocp --- #
        solver = Solver.IPOPT(show_online_optim=False, _tol=1e-3)
        solver.set_maximum_iterations(5000)
        sol = ocp.solve(solver=solver)

        sol.graphs()
        # --- Show results --- #
        # sol.graphs(show_bounds=True)
        sol.print_cost()

        # --- Animation --- #
        # not implemented yet
        # sol.animate()

        # save data
        states = sol.stepwise_states()
        controls = sol.stepwise_controls()
        q_sol, qdot_sol, cn_sol, f_sol = states["q"], states["qdot"], states["cn"], states["f"]
        pw_sol = controls["pw"]
        tau_joints = controls["tau"] if "tau" in controls else None
        tau_res_sol = controls["resistive_torque"] if "resistive_torque" in controls else None
        data = {
            "q": q_sol, "qdot": qdot_sol, "cn": cn_sol, "f": f_sol,
            "pw": pw_sol, "tau_resistive": tau_res_sol, "tau_joints": tau_joints,
            "time": sol.stepwise_time(), "n_iterations": sol.iterations, "final_time": ocp.phase_time,
            "solver_time": sol.solver_time_to_optimize, "real_time": sol.real_time_to_optimize,
            "n_node": n_shooting, "freq": stim_freq, "n_trunc": n_pulse_trunc,
            "target_vel": target_velocity_rpm, 'target_pwr': target_power,
        }
        with open(f'{folder_path}' + filename + '.pkl', "wb") as file:
            del sol.ocp
            pickle.dump(data, file)

    # calculate power with a torque driven example (inverse dynamics with contact points is not implemented yet)
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/fes_leg_pedalling/'
    print(f'Calculating power {dump_path}{filename}')

    import biorbd as brbd
    if power_2B_calculated:
        calculate_power(dump_path,
                        biorbd_model_path,
                        # 'models/leg_model_cyclic_muscle_inversed_root_3_gravity_0.bioMod',
                        filename,
                        stim_freq=35,
                        friction_coef_mu=friction_coef_mu,
                        friction_coef_b=friction_coef_b,
                        print_res=True,
                        set_gravity_to_zero=True)
    plot_power(dump_path,
               brbd.Model(biorbd_model_path),
               filename,
               friction_coef_mu=friction_coef_mu,
               friction_coef_b=friction_coef_b,
               set_gravity_to_zero=False)     # plot power


if __name__ == "__main__":
    main()
