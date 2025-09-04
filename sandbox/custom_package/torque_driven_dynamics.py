"""
This script implements a custom dynamics to work with bioptim.
"""
from biorbd_casadi import biorbd
from casadi import MX, SX, exp, vertcat, casadi, horzcat, pi, fmod
import numpy as np


from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    BiorbdModel,
    ParameterList,
)

from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)

# Model parameters
tauc = 0.011  # 11ms
pd0 = 0.000086  # average value
pdt = 0.000138  # average value
km = 0.180
tau1 = 0.044099
tau2 = 0.018522
a_scale = 5000
stim_amp = 1
r0 = 5
scaling_factor = 1e5  # <- this impacts on the time to convergence. 1e5 faster but cn dynamics with slower slope


def calculate_friction_torque(angular_velocity, mu, B):
    """
    Calculate the frictional torque including Coulomb and viscous friction effects.
    Compatible with CasADi MX/SX types.

    Parameters:
    - angular_velocity: CasADi MX/SX angular velocity (rad/s)
    - mu: Coulomb friction coefficient (Nm)
    - B: Viscous friction coefficient (Nms/rad)

    Returns:
    - friction_torque: CasADi MX/SX torque (Nm)
    """
    coulomb_friction = -mu * casadi.sign(angular_velocity)  # Coulomb friction
    viscous_friction = -B * angular_velocity                # Viscous friction: linear with speed

    return coulomb_friction + viscous_friction


class CustomDynamicFunctions(BiorbdModel):

    def __init__(self, bio_model: str | biorbd.Model, friction_coefficients: np.ndarray = None):
        super().__init__(bio_model, friction_coefficients)  # Call the parent class constructor

    def set_muscle_ascale(self, value: MX, n_mus: int):
        self.model.muscle(n_mus).characteristics().setMuscleAscale(value)

    def set_muscle_km_param(self, value: MX, n_mus: int):
        self.model.muscle(n_mus).characteristics().setMuscleDingKmParam(value)

    def set_muscle_tau1_param(self, value: MX, n_mus: int):
        self.model.muscle(n_mus).characteristics().setMuscleDingTau1Param(value)

    def set_muscle_tau2_param(self, value: MX, n_mus: int):
        self.model.muscle(n_mus).characteristics().setMuscleDingTau2Param(value)

    def get_ding_params(self, n_mus: int):
        a_scale = self.model.muscle(n_mus).characteristics().muscleAscale().to_mx()
        tau1 = self.model.muscle(n_mus).characteristics().muscleDingTau1Param().to_mx()
        tau2 = self.model.muscle(n_mus).characteristics().muscleDingTau2Param().to_mx()
        km = self.model.muscle(n_mus).characteristics().muscleDingKmParam().to_mx()
        return a_scale, tau1, tau2, km

    def muscle_joint_torque_from_muscle_forces(self, forces: MX, q: MX, qdot: MX) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(forces, q_biorbd, qdot_biorbd).to_mx()


    @staticmethod
    def contact_forces_from_fes_driven(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_torque: bool = False,
            with_resistive_torque: bool = False,
            with_ligament: bool = False,
            **extra_parameters: list[MX] | list[SX],
    ) -> MX | SX:
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)

        muscles_tau = CustomDynamicFunctions.muscle_joint_torque_from_muscle_forces(nlp.model, f, q, qdot)
        tau = muscles_tau + DynamicsFunctions.get(nlp.controls["tau"], controls) if with_torque else muscles_tau
        # tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        # tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
        tau[0, :] = tau[0, :] + DynamicsFunctions.get(nlp.controls["resistive_torque"], controls) if with_resistive_torque else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)


    @staticmethod
    def contact_forces_from_fes_cycling(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_torque: bool = False,
            with_ligament: bool = False,
            **extra_parameters: list[MX] | list[SX],
    ) -> MX | SX:
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls else None

        muscles_tau = CustomDynamicFunctions.muscle_joint_torque_from_muscle_forces(nlp.model, f, q, qdot)
        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        # tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_torque else tau
        # tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        friction_coef_b = extra_parameters.get('friction_coef_b')
        friction_coef_mu = extra_parameters.get('friction_coef_mu')
        idx_crank = extra_parameters.get('idx_crank')
        friction_torque = calculate_friction_torque(qdot[idx_crank], mu=friction_coef_mu, B=friction_coef_b) # if with_friction else 0
        friction_vector = casadi.MX.zeros(tau.shape)
        friction_vector[idx_crank] = friction_torque
        tau = tau + friction_vector
        # tau[0, :] = tau[0, :] + friction_torque
        tau = tau + DynamicsFunctions.get(nlp.controls["tau"], controls) if with_torque else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)


    # no approximation
    @staticmethod
    def fes_dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_contact: bool = False,
            with_torque: bool = False,
            with_friction: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_parameters: list[MX] | list[SX]
    ) -> DynamicsEvaluation:

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        cn = DynamicsFunctions.get(nlp.states["cn"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        # CONTROL VARIABLES
        joints_tau = 0
        joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_torque else joints_tau
        pwd = DynamicsFunctions.get(nlp.controls["pw"], controls)
        # PARAMETERS
        t_phase = extra_parameters.get('t_phase')
        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        pd_offset = extra_parameters.get('pd_offset')
        joint_index = extra_parameters.get('joint_index')

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        alpha_temp = MX.zeros(n_pulse_trunc)
        name_mus = nlp.model.muscle_names

        for m, muscle in enumerate(name_mus):
            # see Modeling the length dependence of isometric force in human quadriceps muscles (perumal 2002)
            # see Mathematical models for fatigue minimization during functional electrical stimulation (ding 2003)
            a_scale, tau1, tau2, km = CustomDynamicFunctions.get_ding_params(nlp.model, m)
            if identification_flag:
                a_scale = nlp.parameters["a_scale_param"].cx
                if not only_parameter_a:
                    tau1 = nlp.parameters["tau_1_param"].cx
                    tau2 = nlp.parameters["tau_2_param"].cx
                    km = nlp.parameters["km_param"].cx

            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            alpha_temp[1:n_pulse_trunc] = alpha[:n_pulse_trunc - 1]  # shift to the right (1 position)
            if identification_flag or prediction_flag:  # 1 stimulation channel only
                a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd_offset)))
                a_sc = a_scale * (1 - exp(-(pwd - pd_offset) / pdt))
            else:
                a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd_offset)))
                a_sc = a_scale * (1 - exp(-(pwd[m] - pd_offset) / pdt))
            if prediction_flag or fes_driven_task_flag: # a_scale depending on the knee angle
                coef = extra_parameters.get('muscle_params')[muscle][0]
                x = q[joint_index[m]]
                a_scale = coef[0] * x ** 5 + coef[1] * x ** 4 + coef[2] * x ** 3 + coef[3] * x ** 2 + coef[4] * x + \
                          coef[5]
                a_scale = casadi.if_else(a_scale > 0, a_scale, 0)
                a_sc = a_scale * (1 - exp(-(pwd[m] - pd_offset) / pdt))
            alpha_temp[0] = (a - 0.5) / 0.5  # normalize alpha between 0 and 1
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            # sum_var_list.append(sum_aux)
            alpha_dot_list.append(alpha_dot)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))
        # cn_dot = (1 / tauc) * casadi.vertcat(*sum_var_list) - (cn / tauc)
        # f_dot = a_sc * (cn / (km + cn)) - (f / (tau1 + tau2 * (cn / (km + cn))))

        muscles_tau = CustomDynamicFunctions.muscle_joint_torque_from_muscle_forces(nlp.model, f, q, qdot)
        # muscles_tau = muscles_tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else muscles_tau
        # muscles_tau = muscles_tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else muscles_tau

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau + joints_tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


    @staticmethod
    def fes_param_indent_dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_contact: bool = False,
            with_torque: bool = False,
            with_friction: bool = False,
            with_resistive_torque: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_parameters: list[MX] | list[SX]
    ) -> DynamicsEvaluation:

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        cn = DynamicsFunctions.get(nlp.states["cn"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        # CONTROL VARIABLES
        pwd = DynamicsFunctions.get(nlp.controls["pw"], controls)
        # PARAMETERS
        t_phase = extra_parameters.get('t_phase')
        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        pd_offset = extra_parameters.get('pd_offset')
        joint_index = extra_parameters.get('joint_index')

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        alpha_temp = MX.zeros(n_pulse_trunc)
        name_mus = nlp.model.muscle_names

        for m, muscle in enumerate(name_mus):
            # see Modeling the length dependence of isometric force in human quadriceps muscles (perumal 2002)
            # see Mathematical models for fatigue minimization during functional electrical stimulation (ding 2003)
            a_scale, tau1, tau2, km = CustomDynamicFunctions.get_ding_params(nlp.model, m)
            a_scale = nlp.parameters["a_scale_param"].cx
            if not only_parameter_a:
                tau1 = nlp.parameters["tau_1_param"].cx
                tau2 = nlp.parameters["tau_2_param"].cx
                km = nlp.parameters["km_param"].cx
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            alpha_temp[1:n_pulse_trunc] = alpha[:n_pulse_trunc - 1]  # shift to the right (1 position)
            # 1 channel only
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd_offset)))
            a_sc = a_scale * (1 - exp(-(pwd - pd_offset) / pdt))
            alpha_temp[0] = (a - 0.5) / 0.5  # normalize alpha between 0 and 1
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            # sum_var_list.append(sum_aux)
            alpha_dot_list.append(alpha_dot)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))
        # cn_dot = (1 / tauc) * casadi.vertcat(*sum_var_list) - (cn / tauc)
        # f_dot = a_sc * (cn / (km + cn)) - (f / (tau1 + tau2 * (cn / (km + cn))))

        muscles_tau = CustomDynamicFunctions.muscle_joint_torque_from_muscle_forces(nlp.model, f, q, qdot)
        # muscles_tau = muscles_tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else muscles_tau
        # muscles_tau = muscles_tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else muscles_tau
        # joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_orque else joints_tau
        muscles_tau[0, :] = muscles_tau[0, :] + DynamicsFunctions.get(nlp.controls["resistive_torque"],
                                                      controls) if with_resistive_torque else muscles_tau

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)



    @staticmethod
    def fes_cycling_dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_contact: bool = False,
            with_torque: bool = False,
            with_friction: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_parameters: list[MX] | list[SX]
    ) -> DynamicsEvaluation:

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        cn = DynamicsFunctions.get(nlp.states["cn"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        # CONTROL VARIABLES
        # joints_tau = 0
        # joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_torque else joints_tau
        pwd = DynamicsFunctions.get(nlp.controls["pw"], controls)
        # PARAMETERS
        t_phase = extra_parameters.get('t_phase')
        n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        pd_offset = extra_parameters.get('pd_offset')
        joint_index = extra_parameters.get('joint_index')

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list, sum_var_list = [], [], [], []
        name_mus = nlp.model.muscle_names

        for m, muscle in enumerate(name_mus):
            a_scale, tau1, tau2, km = CustomDynamicFunctions.get_ding_params(nlp.model, m)
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd_offset)))
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])

            # for the following, we considered a fifth order polynom
            coef = extra_parameters.get('muscle_params')[muscle][0]
            # coef = [-71.87396332, 530.82703873, -1445.0495578, 1802.55293297, -1004.23625738, 222.40557217]
            # coef = [39614.532, -164339.282, 224888.991, -100244.656]
            x = q[joint_index[m]]
            # a_scale = (coef[0] * x ** 5 + coef[1] * x ** 4 + coef[2] * x ** 3 + coef[3] * x ** 2 + coef[4] * x + \
            #           coef[5])**2
            a_scale = (sum(c * x**(len(coef) - 1 - i) for i, c in enumerate(coef)))**2
            # a_scale = -10
            a_sc = a_scale * (1 - exp(-(pwd[m] - pd_offset) / pdt))
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)

            # alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            # coeff = extra_parameters.get('muscle_params')[muscle][0]
            # alpha_temp[1:n_pulse_trunc] = alpha[:n_pulse_trunc - 1]  # shift to the right (1 position)
            # a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd_offset)))
            # alpha_temp[0] = (a - 0.5) / 0.5  # normalize alpha between 0 and 1
            # # if fes_driven_task_flag:
            # x = q[joint_index[m]]    # q[int(m/(n_mus/2))+3]    # gear, pedal_r, knee_r, hip_r, pedal_l, knee_l, hip_l
            # a_scale = 0
            # for i in range(len(coeff)):
            #     a_scale += coeff[-1-i] * x**i
            # a_sc = a_scale * (1 - exp(-(pwd[m] - pd_offset) / pdt))
            # alpha_dot = (alpha_temp - alpha) / t_phase
            # sum_aux = 0
            # for i in range(n_pulse_trunc):
            #     ti = (num_t_phase - i) * t_phase
            #     sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            sum_var_list.append(sum_aux)

            alpha_dot = (alpha_temp - alpha) / t_phase
            alpha_dot_list.append(alpha_dot)
            # f_dot_list.append(a_sc * (cn[m] / (SX(km) + cn[m])) - (f[m] / (SX(tau1) + SX(tau2) * (cn[m] / (SX(km) + cn[m])))))
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            # cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))
        cn_dot = (1 / tauc) * casadi.vertcat(*sum_var_list) - (cn / tauc)
        # f_dot = a_sc * (cn / (km + cn)) - (f / (tau1 + tau2 * (cn / (km + cn))))

        muscles_tau = CustomDynamicFunctions.muscle_joint_torque_from_muscle_forces(nlp.model, f, q, qdot)
        # muscles_tau = muscles_tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else muscles_tau
        # muscles_tau = muscles_tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else muscles_tau

        # Add resistive torque to the gear only (joint at position 0)
        friction_coef_b = extra_parameters.get('friction_coef_b')
        friction_coef_mu = extra_parameters.get('friction_coef_mu')
        idx_crank = extra_parameters.get('idx_crank')
        friction_torque = calculate_friction_torque(qdot[idx_crank], mu=friction_coef_mu,
                                                    B=friction_coef_b)  # if with_friction else 0

        friction_vector = casadi.MX.zeros(muscles_tau.shape)
        friction_vector[idx_crank] = friction_torque
        muscles_tau = muscles_tau + friction_vector
        # muscles_tau[0, :] = muscles_tau[0, :] + friction_torque
        muscles_tau = muscles_tau + DynamicsFunctions.get(nlp.controls["tau"], controls) if with_torque else muscles_tau

        # dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               cn_dot,
                                               # casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)



