import os
import pickle
import numpy as np
import biorbd as brbd
from bioptim.dynamics.configure_problem import _check_contacts_in_biorbd_model, ConfigureProblem
from casadi import MX, SX, horzcat, casadi

from bioptim.limits.path_conditions import InitialGuessList
from bioptim import (
    BiorbdModel,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    BoundsList,
    Solver,
    Node,
    Axis,
    PhaseDynamics,
    ObjectiveList,
    OptimalControlProgram,
    OdeSolverBase,
    OdeSolver, PlotType, NonLinearProgram, DynamicsFunctions, DynamicsEvaluation,
)
from matplotlib import pyplot as plt

from custom_package.fes_configure_probem import FesDynamicsFcn
from custom_package.fes_dynamics import calculate_friction_torque
from custom_package.fes_plotting import fes_plot_power, organize_data, unpack_sol, get_data




def plot_power(dump_path, model, file, friction_coef_mu,friction_coef_b, set_gravity_to_zero):

    path = dump_path + f'{file}.pkl'

    n_q = model.nbDof()
    n_mus = model.nbMuscles()
    muscle_names = tuple(s.to_string() for s in model.muscleNames())
    data = get_data(path)
    print(f"{path}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")

    offset = 4
    q_sol, qdot_sol = data["q"][offset:], data["qdot"][offset:]
    f_sol, pw_sol, time_sol = data["f"][offset:], data["pw"][offset:], data["time"][offset:]

    time = data["time"]
    t_array = np.concatenate(
        [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
         enumerate(time)])

    q_data_aux = organize_data(q_sol, n_q)
    dq_data_aux = organize_data(qdot_sol, n_q)
    f_data_aux = organize_data(f_sol, n_mus)
    pw_data_aux = organize_data(pw_sol, n_mus)

    file_name = dump_path + f'eval_pwr_{file}.pkl'
    data = get_data(file_name)
    q_sol_torque, qdot_sol_torque = data["q"][offset:], data["qdot"][offset:]
    tau_control = data["tau_joints"][offset:] if data["tau_joints"] else None
    pwr2, ddq_gear, pwr, dq_gear, tau_gear, q_gear = [], [], [], [], [], []
    dq_gear_torque, q_gear_torque = [], []
    mult_factor = len(q_data_aux[0]) // len(tau_control)
    aux = np.array([[float(tau[0, 0])] * mult_factor for tau in tau_control]).flatten()
    tau_gear.append(aux)
    q_data_aux_torque = organize_data(q_sol_torque, n_q)
    dq_data_aux_torque = organize_data(qdot_sol_torque, n_q)

    if set_gravity_to_zero:
        model.setGravity(np.array([0, 0, 0]))

    for t in range(len(f_data_aux[0])):
        _f, _q, _dq = [], [], []
        _q_torque, _dq_torque = [], []
        for m in range(n_mus):
            _f.append(float(f_data_aux[m][t]))
        for q in range(n_q):
            _q.append(float(q_data_aux[q][t]))
            _dq.append(float(dq_data_aux[q][t]))
            _q_torque.append(float(q_data_aux_torque[q][t]))
            _dq_torque.append(float(dq_data_aux_torque[q][t]))

        f, q, dq = np.array(_f), np.array(_q), np.array(_dq)
        q_torque, dq_torque = np.array(_q_torque), np.array(_dq_torque)
        muscles_tau = model.muscularJointTorque(f, q, dq).to_array()

        friction_torque = calculate_friction_torque(dq[0], mu=friction_coef_mu, B=friction_coef_b)
        muscles_tau[0] = muscles_tau[0] + friction_torque

        ddq = model.ForwardDynamicsConstraintsDirect(q, dq, muscles_tau)
        ddq_gear.append(float(ddq.to_array()[0]))
        q_gear.append(float(q[0]))
        dq_gear.append(float(dq[0]))

        # just to compare the two examples
        q_gear_torque.append(float(q_torque[0]))
        dq_gear_torque.append(float(dq_torque[0]))

        pwr2.append(friction_coef_b * np.pow(dq[0], 2))

    # --------------------------------- First Figure ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    # ------------------------------------ plot 1 ------------------------------------
    pwr = tau_gear[0] * dq_gear[:-1]
    axes[0].plot(pwr, label="Power [W] (torque * qdot)", color="tab:green")
    axes[0].plot(pwr2, label=f"B * qdot² (B = {friction_coef_b})", color="tab:blue")
    axes[0].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)

    # ------------------------------------ plot 2 ------------------------------------
    axes[1].plot(dq_gear, label=f"dq [rad/s]", color="tab:green")       # plot velocity
    axes[1].plot(dq_gear_torque, label=f"dq_torque [rad/s]")  # plot velocity
    axes[1].plot(np.diff(q_gear) / (t_array[6] - t_array[5]), "b--", label=f"dq [rad/s]")    # manual calc. of velocity
    axes[1].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    axes[1].set_ylabel('dq [rad/s]', color='green')

    ax2 = axes[1].twinx()
    ax2.plot(ddq_gear, label=f"dqq [rad/s²]", color="tab:red")                              # plot acceleration
    ax2.plot(np.diff(dq_gear) / (t_array[6] - t_array[5]), "b--", label=f"dqq [rad/s²]")    # manual calc. of acc.
    ax2.legend(loc='upper right', fontsize='small', frameon=True, ncol=1)
    ax2.set_ylabel('dqq [rad/s²]', color='red')

    # ------------------------------------ plot 3 ------------------------------------
    axes[2].plot(tau_gear[0], label="torque [Nm]", color="tab:blue")
    axes[2].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    ax3 = axes[2].twinx()
    ax3.plot(ddq_gear, label=f"dqq [rad/s²]", color="tab:red")
    ax3.legend(loc='upper right', fontsize='small', frameon=True, ncol=1)
    plt.tight_layout()
    # plt.show()

    # --------------------------------- Second Figure ---------------------------------
    fig, axes = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(16, 6))
    # ------------------------------------ plot 1 ------------------------------------
    q_size = len(q_data_aux[0])
    pulse_width = np.array(pw_sol)
    node_count = pulse_width.shape[0]
    repeat_factor = q_size // node_count
    pw_squeezed = np.squeeze(pulse_width, axis=2)
    pw_stretched = np.repeat(pw_squeezed, repeat_factor, axis=0)
    pw_stretched = pw_stretched[:q_size, :]

    radial_offsets = np.linspace(10.5, 20.5, 8)
    # fig, axes = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(16, 6))
    q_radians = q_data_aux[0] % (2 * np.pi) * -1
    for idx in range(n_mus):
        # Usa diretamente os dados estendidos por músculo
        pw_data_stretched = pw_stretched[:, idx]
        # Define os pontos em que há ativação
        activated_points = np.where(pw_data_stretched[:-5] > 300e-6)[0]
        if len(activated_points) > 0:
            axes[0].plot(q_radians[activated_points],
                         np.full_like(activated_points, radial_offsets[idx]),
                         'p',
                         label=f"{muscle_names[idx]}")
    axes[0].set_xticks(np.linspace(0, 2 * np.pi, 72, endpoint=False))
    axes[0].set_title("a) stimulation Pattern", fontsize=14)
    # axes[0].set_xticklabels(['0°', '90°', '180°', '270°'])
    axes[0].set_yticks([])  # Remove radial ticks
    axes[0].set_yticklabels([])  # Remove radial tick labels
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0), fontsize='small', frameon=True, ncol=1)
    axes[0].legend(loc='center', fontsize='medium', frameon=False, ncol=1)
    # plt.show()

    q_radians = q_data_aux[0] * -1
    qdot_radians = dq_data_aux[0]
    axes[1].plot(q_radians, qdot_radians, label="Velocity [rad/s]", color="tab:blue")
    axes[1].set_title("b) crank Angular Velocity", fontsize=14)
    axes[1].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[1].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes[1].tick_params(labelsize=10)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes[1].plot(q_radians[0], qdot_radians[0], 'ro')  # Red dot at start
    axes[1].text(q_radians[0], qdot_radians[0] + 0.1, 'Start', color='red', fontsize=10, ha='left')
    #
    axes[1].plot(q_radians[-1], qdot_radians[-1], 'ro')  # Red dot at start
    axes[1].text(q_radians[-1], qdot_radians[-1] + 0.1, 'End', color='red', fontsize=10, ha='right')
    #
    # print mean values
    mean_vel = np.mean(qdot_radians)
    std_rpm = np.std(qdot_radians)
    print(f'Mean velocity: {mean_vel:.2f} ± {std_rpm:.2f} rad/s')
    mean_rpm = np.mean(qdot_radians) * (60 / (2 * np.pi))
    std_rpm = np.std(qdot_radians) * (60 / (2 * np.pi))
    print(f'Mean velocity: {mean_rpm:.2f} ± {std_rpm:.2f} RPM')
    # plt.legend(fontsize='small', frameon=False, ncol=4)

    pwr_radians = np.array(pwr)
    q_radians = q_radians[:-1]
    axes[2].plot(q_radians, pwr_radians, label="Power [W]", color="tab:green")
    axes[2].fill_between(q_radians, 0, pwr_radians, alpha=0.1, color="tab:green")
    # pwr_radians = np.array(pwr2) * -1
    # axes[2].plot(q_radians, pwr_radians[:-1], label="Power [W]", color="tab:blue")
    axes[2].set_title("c) power Output", fontsize=14)
    axes[2].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[2].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes[2].grid(True, linestyle='--', linewidth=0.5)
    axes[2].legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes[2].plot(q_radians[0], pwr_radians[0], 'ro')  # Red dot at start
    axes[2].text(q_radians[0], pwr_radians[0] + 0.1, 'Start', color='red', fontsize=10, ha='left')

    axes[2].plot(q_radians[-1], pwr_radians[-1], 'ro')  # Red dot at start
    axes[2].text(q_radians[-1], pwr_radians[-1] + 0.1, 'End', color='red', fontsize=10, ha='right')

    # print mean values
    mean_pwr = np.mean(pwr_radians)
    std_pwr = np.std(pwr_radians)
    print(f'Mean power: {mean_pwr:.2f} ± {std_pwr:.2f} W')

    plt.savefig('opt_sol_results/images/stim_pattern.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()


def dump_torque_driven(ocp, sol, filename, n_phase):
    states = sol.stepwise_states()
    controls = sol.stepwise_controls()
    solver_time = sol.solver_time_to_optimize
    real_time = sol.real_time_to_optimize
    n_iterations = sol.iterations
    time = sol.stepwise_time()
    final_time = ocp.phase_time

    q_sol = states["q"]
    qdot_sol = states["qdot"]
    tau_joints = controls["tau"] if "tau" in controls else None
    tau_res_sol = controls["resistive_torque"] if "resistive_torque" in controls else None

    data = {
        "q": q_sol,
        "qdot": qdot_sol,
        "tau_resistive": tau_res_sol,
        "tau_joints": tau_joints,
        "time": time,
        "solver_time": solver_time,
        "real_time": real_time,
        "n_iterations": n_iterations,
        "final_time": final_time,
    }

    sol_file_name = filename                # fp stands for force production
    with open(sol_file_name, "wb") as file:
        del sol.ocp
        pickle.dump(data, file)

    return

def custom_forces_from_torque_driven(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries: MX.sym,
    nlp,
    with_passive_torque: bool = False,
    with_ligament: bool = False,
    **extra_parameters: list[MX] | list[SX],
) -> MX | SX:

    q = nlp.get_var_from_states_or_controls("q", states, controls)
    qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
    tau = nlp.get_var_from_states_or_controls("tau", states, controls)

    friction_vector = casadi.MX.zeros(tau.shape)
    friction_coef_mu = extra_parameters['friction_coef_mu']
    friction_coef_b = extra_parameters['friction_coef_b']
    friction_torque = calculate_friction_torque(qdot[0], mu=friction_coef_mu, B=friction_coef_b)
    friction_vector[0] = friction_torque
    tau = tau + friction_vector

    external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
    return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)


def custom_torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_friction: bool = False,
        fes_driven_task_flag: bool = False,
        prediction_flag: bool = False,
        only_parameter_a: bool = False,
        identification_flag: bool = False,
        **extra_parameters: list[MX] | list[SX]
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    # remember to check the inertia at the gear in the biomechanical model (bioMod)
    friction_coef_mu = extra_parameters['friction_coef_mu']
    friction_coef_b = extra_parameters['friction_coef_b']
    friction_torque = calculate_friction_torque(qdot[0], mu=friction_coef_mu, B=friction_coef_b)
    friction_vector = casadi.MX.zeros(tau.shape)
    friction_vector[0] = friction_torque
    tau = tau + friction_vector
    # tau = tau[0, :] = tau[0, :] + friction

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces=None)
    dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    return DynamicsEvaluation(dxdt=dxdt, defects=None)



def configure_custom_torque_driven(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_friction: bool = False,
        **extra_parameters):

    _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)

    # DECLARE STATES
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    # DECLARE CONTROLS
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    # ConfigureProblem.configure_new_variable("resistive_torque",
    #                                         ["gear_torque"],
    #                                         ocp,
    #                                         nlp,
    #                                         as_states=False,
    #                                         as_controls=True)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        custom_torque_driven,
        with_contact=with_contact,
        # with_passive_torque=with_passive_torque,
        # with_ligament=with_ligament,
        with_friction=with_friction,
        **extra_parameters,
    )

    # Configure the contact forces
    if with_contact:
        ConfigureProblem.configure_contact_function(ocp,
                                                    nlp,
                                                    custom_forces_from_torque_driven,
                                                    **extra_parameters)


def prepare_ocp(
        model_path,
        cycle_shooting,
        cycle_duration,
        target_pos,
        target_vel,
        max_joint_torque,
        q_init,
        dq_init,
        tau_init,
        friction_coef_mu,
        friction_coef_b,
        set_gravity_to_zero,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        with_friction=False,
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
):
    model = BiorbdModel(model_path)
    if set_gravity_to_zero:
        model.set_gravity(MX([0, 0, 0]))

    extra_parameters = {'friction_coef_mu': friction_coef_mu,
                        'friction_coef_b': friction_coef_b}
    dynamics = Dynamics(configure_custom_torque_driven,
                        expand_dynamics=expand_dynamics,
                        phase_dynamics=phase_dynamics,
                        with_contact=True,
                        **extra_parameters,
                        )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        node=Node.ALL,
        key="q",
        target= target_pos,         # target position at gear crank
        weight=1e3,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        node=Node.ALL,
        key="qdot",
        target= target_vel,         # target velocity at gear crank
        weight=1e4,
    )

    x_bound = BoundsList()
    x_bound["q"] = model.bounds_from_ranges("q")
    x_bound["q"].min[:, [0]] = q_init
    x_bound["q"].max[:, [0]] = q_init
    x_bound["qdot"] = model.bounds_from_ranges("qdot")
    x_bound["qdot"].min[:, [0]] = dq_init  # Initial velocity = 0 so the contact points would be still
    x_bound["qdot"].max[:, [0]] = dq_init  # Initial velocity = 0 so the contact points would be still (starting node with 0 veloc.)

    u_bound = BoundsList()
    # ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    u_bound["tau"] = ([-max_joint_torque, 0, 0, 0, 0, 0, 0], [max_joint_torque, 0, 0, 0, 0, 0, 0])

    x_init = InitialGuessList()
    x_init["q"] = q_init
    u_init = InitialGuessList()

    ocp = OptimalControlProgram(
        model,
        dynamics,
        cycle_shooting,
        cycle_duration,
        objective_functions=objective_functions,
        x_bounds=x_bound,
        u_bounds=u_bound,
        x_init=x_init,
        u_init=u_init,
        ode_solver=ode_solver,
        n_threads=16,
    )

    return ocp


def calculate_power(dump_path,
                    model_path,
                    file,
                    stim_freq: int = 35,
                    friction_coef_mu: float = 0.1,
                    friction_coef_b: float = 0.7784,
                    print_res: bool = True,
                    set_gravity_to_zero: bool = True):

    path = dump_path + f'{file}.pkl'
    sol_dict = pickle.load(open(path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T
    dqvec = np.array(unpack_sol(sol_dict["qdot"])).T

    n_shooting = len(dqvec[0]) - 1
    phase_period = round(1 / stim_freq, 3)
    duration = phase_period * n_shooting
    q_init = [[float(value[0])] for i, value in enumerate(sol_dict["q"][0])]
    dq_init = [[float(value[0])] for i, value in enumerate(sol_dict["qdot"][0])]
    tau_init = 0

    ocp = prepare_ocp(model_path,
                      cycle_duration=duration,
                      cycle_shooting=n_shooting,
                      q_init=q_init,
                      dq_init=dq_init,
                      tau_init=tau_init,
                      target_pos=qvec[:7,],
                      target_vel=dqvec[:7,],
                      friction_coef_mu=friction_coef_mu,
                      friction_coef_b=friction_coef_b,
                      max_joint_torque=100,
                      set_gravity_to_zero=set_gravity_to_zero,
                    )

    # Solve the program
    sol = ocp.solve(solver=Solver.IPOPT(show_online_optim=False,
                                        _max_iter=500,
                                        _tol=1e-4,
                                        _print_level=5))

    if print_res:
        sol.graphs()
        # sol.print_cost()
        # sol.animate()
    print("Initialization failure" if sol.status else "Initialization success")
    filename = f'opt_sol_results/fes_leg_pedalling/eval_pwr_{file}.pkl'
    dump_torque_driven(ocp, sol, filename, 1)



if __name__ == "__main__":

    dump_path = f'opt_sol_results/fes_leg_pedalling/'
    model_path = "../models/leg_model_cyclic_muscle_inversed_root_3.bioMod"
    file = 'friction_3.05s_105n_41.92rpm_20.0W_fric_0.2_1.0378'
    # filename = f'friction_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}_opt_crank'
    file = 'friction_2.09s_72n_41.92rpm_25.0W_fric_0.2_1.2973' # _backing_up_4th_order

    friction_coef_mu = 0.2
    friction_coef_b = float(file[-6:])
    model = brbd.Model(model_path)

    # calculate_power(dump_path,
    #                 model_path,
    #                 file,
    #                 stim_freq = 35,
    #                 friction_coef_mu=friction_coef_mu,
    #                 friction_coef_b=friction_coef_b,
    #                 print_res = False,
    #                 set_gravity_to_zero=True)
    plot_power(dump_path, model, file, friction_coef_mu, friction_coef_b, set_gravity_to_zero=False) # plot power
