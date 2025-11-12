"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""

from typing import Callable

import numpy as np
from biorbd_casadi import GeneralizedCoordinates, GeneralizedVelocity
from casadi._casadi import exp

from bioptim.misc.fcn_enum import FcnEnum
from bioptim.misc.parameters_types import Bool, Str
import biorbd_casadi as biorbd

from bioptim import TorqueDynamics, FatigueList, States, Controls, DynamicsFunctions, DefectType, ContactType, \
    DynamicsEvaluation, NonLinearProgram, OdeSolver, BiorbdModel, ExternalForceSetTimeSeries, ExternalForceSetVariables, \
    ParameterList, ConfigureVariables
from casadi import sin, MX, Function, vertcat, casadi
from typing import Callable


# Model parameters
# pd0 = 0.000086  # average value Ding 2007
# pdt = 0.000138  # average value Ding 2007
tauc = 0.011  # 11ms or 20ms?
r0 = 5
scaling_factor = 1e6  # <- this impacts on the time to convergence. 1e5 faster but cn dynamics with slower slope


def set_muscle_ascale(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleAscale(value)


def set_muscle_km_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingKmParam(value)


def set_muscle_tau1_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingTau1Param(value)


def set_muscle_tau2_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingTau2Param(value)


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


class FESConfigureVariables(ConfigureVariables):

    @staticmethod
    def configure_fes_cn(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
    ):
        """
        """

        muscle_names = nlp.model.muscle_names
        ConfigureVariables.configure_new_variable(
            "cn",
            muscle_names,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            # combine_state_control_plot=True,
        )

    @staticmethod
    def configure_fes_f(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
    ):
        """
        """

        muscle_names = nlp.model.muscle_names
        ConfigureVariables.configure_new_variable(
            "f",
            muscle_names,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            # combine_state_control_plot=True,
        )

    @staticmethod
    def configure_fes_alpha(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
            **extra_params,
    ):
        """
        """

        name_elements = []
        n_pulse_trunc = 6   #extra_params['n_pulse_trunc'] Todo: pass this parameter automatically
        for i in range(n_pulse_trunc):
            name_elements.append(f'a{i + 1}')
        for i in range(nlp.model.nb_muscles):
            ConfigureVariables.configure_new_variable(f"alpha_{i}",  # to keep the previous stimulation pulses
                                                      name_elements,
                                                      ocp,
                                                      nlp,
                                                      as_states=as_states,
                                                      as_controls=as_controls,
                                                      as_algebraic_states=as_algebraic_states,
                                                      )

    @staticmethod
    def configure_fes_pw(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
    ):
        """
        Configure the muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the muscles should be a state
        as_controls: bool
            If the muscles should be a control
        as_algebraic_states: bool
            If the muscles should be an algebraic state
        """

        muscle_names = nlp.model.muscle_names
        ConfigureVariables.configure_new_variable(
            "pw",
            muscle_names,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            # combine_state_control_plot=True,
        )

    @staticmethod
    def configure_fes_pw_1_channel(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
    ):
        """
        """
        # 1 channel only
        ConfigureVariables.configure_new_variable(
            "pw",
            ["fes_input"],
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            # combine_state_control_plot=True,
        )

    @staticmethod
    def configure_resistive_torque(
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool,
    ):
        """
        """
        ConfigureVariables.configure_new_variable(
            "resistive_torque",
            ["gear_torque"],
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            # combine_state_control_plot=True,
        )


class FesStates(FcnEnum):
    F = (FESConfigureVariables.configure_fes_f,)
    CN = (FESConfigureVariables.configure_fes_cn,)
    ALPHA = (FESConfigureVariables.configure_fes_alpha,)


class FesControls(FcnEnum):
    PW = (FESConfigureVariables.configure_fes_pw,)
    PW1ch = (FESConfigureVariables.configure_fes_pw_1_channel,)
    RESTORQUE = (FESConfigureVariables.configure_resistive_torque,)


class CustomFesDynamics(BiorbdModel, TorqueDynamics):
    """
    This class is used to create a model actuated through muscle activation.

    x = [q, qdot, muscles (if with_excitation)]
    u = [muscles, tau (if with_residual_torque)]
    """

    def __init__(
            self,
            bio_model: Str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            with_residual_torque: Bool = False,
            with_friction: Bool = False,
            parameters: ParameterList = None,
            parameter_identification: Bool = False,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
            contact_types: list[ContactType] | tuple[ContactType] = (),
            fatigue: FatigueList = None,
            **extra_params,
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueDynamics.__init__(self, fatigue)

        self.state_configuration = [States.Q, States.QDOT,
                                    FesStates.CN, FesStates.F,
                                    FesStates.ALPHA,
                                    ]

        if parameter_identification:
            self.control_configuration = [FesControls.PW1ch, FesControls.RESTORQUE]
        elif with_residual_torque:
            self.control_configuration = [Controls.TAU, FesControls.PW]
        else:
            self.control_configuration = [FesControls.PW]

        self.algebraic_configuration = []
        self.functions = []
        self.with_residual_torque = with_residual_torque
        self.extra_params = extra_params
        self.with_friction = with_friction

    # def serialize(self) -> tuple[Callable, dict]:
    #     return CustomFesDynamics, dict(
    #         bio_model=self.path,
    #         friction_coefficients=self.friction_coefficients,
    #         external_force_set=self.external_force_set,
    #         contact_types=self.contact_types,
    #     )


    def get_basic_variables(
            self,
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
    ):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        cn = DynamicsFunctions.get(nlp.states["cn"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        # CONTROL VARIABLES
        pwd = DynamicsFunctions.get(nlp.controls["pw"], controls)
        # extra_params
        idx_crank = extra_params.get('idx_crank')
        # Residual torque
        if "tau" in nlp.controls:
            residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        else:
            residual_tau = MX.zeros(q.shape)
        # Resistive torque
        if "resistive_torque" in nlp.controls:
            resistive_torque = DynamicsFunctions.get(nlp.controls["resistive_torque"], controls)
        else:
            resistive_torque = MX.zeros(q.shape)
        # Muscle torque
        muscles_tau = residual_tau + self.muscle_joint_torque_from_muscle_forces(f, q, qdot)
        # resistive torque is added to the crank axis only (index 0)
        muscles_tau[idx_crank] = muscles_tau[idx_crank] + resistive_torque[idx_crank]

        if self.with_friction:
            friction_coef_b = extra_params.get('friction_coef_b')
            friction_coef_mu = extra_params.get('friction_coef_mu')
            friction_tau = calculate_friction_torque(qdot[idx_crank],
                                                        mu=friction_coef_mu,
                                                        B=friction_coef_b)
            friction_torque = MX.zeros(q.shape)
            friction_torque[idx_crank] = friction_tau
            muscles_tau = muscles_tau + friction_torque
            # muscles_tau = self.muscle_joint_torque_from_muscle_forces(f, q, qdot)
        # muscles_tau = muscles_tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else muscles_tau
        # muscles_tau = muscles_tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else muscles_tau
        # joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_orque else joints_tau

        return q, qdot, cn, f, pwd, muscles_tau


    def muscle_joint_torque_from_muscle_forces(self, forces: MX, q: MX, qdot: MX) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(forces, q_biorbd, qdot_biorbd).to_mx()


    def get_ding_params(self, n_mus: int):
        a_scale = self.model.muscle(n_mus).characteristics().muscleAscale().to_mx()
        tau1 = self.model.muscle(n_mus).characteristics().muscleDingTau1Param().to_mx()
        tau2 = self.model.muscle(n_mus).characteristics().muscleDingTau2Param().to_mx()
        km = self.model.muscle(n_mus).characteristics().muscleDingKmParam().to_mx()
        return a_scale, tau1, tau2, km


    def get_rigid_contact_forces(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            **extra_params,
    ):
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, muscles_tau, external_forces, nlp.parameters.cx)

    def dynamics(
            self,
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
            **extra_params,
    ):

        # # Get states indices
        # q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        t_phase = extra_params['t_phase']
        n_pulse_trunc = extra_params['n_pulse_trunc']
        pd0 = extra_params['pd0']
        pdt = extra_params['pdt']
        joint_index = extra_params['idx_crank']

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        name_mus = self.muscle_names

        for m, muscle in enumerate(name_mus):
            a_scale, tau1, tau2, km = self.get_ding_params(m)
            if identification_flag:
                a_scale = nlp.parameters["a_scale_param"].cx
                if not only_parameter_a:
                    tau1 = nlp.parameters["tau_1_param"].cx
                    tau2 = nlp.parameters["tau_2_param"].cx
                    km = nlp.parameters["km_param"].cx

            if identification_flag or prediction_flag:  # 1 stimulation channel only
                a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd0)))
                a_sc = a_scale * (1 - exp(-(pwd - pd0) / pdt))
            else:
                a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd0)))
                # a_scale = casadi.if_else(a_scale > 0, a_scale, 0)
                a_sc = a_scale * (1 - exp(-(pwd[m] - pd0) / pdt))
            if prediction_flag or fes_driven_task_flag: # a_scale depending on the knee angle
                coef = extra_params.get('muscle_params')[muscle][0]
                x = q[joint_index[m]]
                a_scale = (sum(c * x ** (len(coef) - 1 - i) for i, c in enumerate(coef))) ** 2
                a_scale = casadi.if_else(a_scale > 0, a_scale, 0)
                a_sc = a_scale * (1 - exp(-(pwd[m] - pd0) / pdt))

            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])

            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            alpha_dot_list.append((alpha_temp - alpha) / t_phase)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


class ParamIdentDynamics(CustomFesDynamics):
    """
    This class is used to create a model actuated through muscle activation.
    """

    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_resistive_torque: bool = False,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
        fatigue: FatigueList = None,
        **extra_params,
    ):
        super().__init__(
            bio_model=bio_model,
            friction_coefficients=friction_coefficients,
            with_residual_torque=with_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
            contact_types=contact_types,
            parameter_identification=True,
            fatigue=fatigue,
            **extra_params,
        )
        self.with_resistive_torque = with_resistive_torque


    def dynamics(
            self,
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
            with_residual_torque: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_params,
    ):
        # Get variables
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        nlp.model.set_gravity(MX([0, 0, 0]))

        t_phase = extra_params['t_phase']
        n_pulse_trunc = extra_params['n_pulse_trunc']

        # get pd0 pdt
        a_coeff = extra_params['pd0pdt'][0]
        pd0 = extra_params['pd0pdt'][1]
        pdt = extra_params['pd0pdt'][2]
        # fes_min = extra_params['fes_min']
        muscle_params = extra_params['muscle_params']

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        name_mus = self.muscle_names

        for m, muscle in enumerate(name_mus):
            tau1 = muscle_params[1]
            tau2 = muscle_params[2]
            km = muscle_params[3]
            a_scale = nlp.parameters["a_scale_param"].cx
            if not only_parameter_a:
                # tau1 = nlp.parameters["tau_1_param"].cx
                tau2 = nlp.parameters["tau_2_param"].cx
                km = nlp.parameters["km_param"].cx
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            a_sc = a_scale * a_coeff * (1 - exp(-(pwd - pd0) / pdt))
            # 1 channel only
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd0)))
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            alpha_dot_list.append(alpha_dot)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


class ParamTau1IdentDynamics(CustomFesDynamics):
    """
    This class is used to create a model actuated through muscle activation.
    """

    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_resistive_torque: bool = False,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
        fatigue: FatigueList = None,
        **extra_params,
    ):
        super().__init__(
            bio_model=bio_model,
            friction_coefficients=friction_coefficients,
            with_residual_torque=with_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
            contact_types=contact_types,
            parameter_identification=True,
            fatigue=fatigue,
            **extra_params,
        )
        self.with_resistive_torque = with_resistive_torque


    def dynamics(
            self,
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
            with_residual_torque: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_params,
    ):
        # Get variables
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        nlp.model.set_gravity(MX([0, 0, 0]))

        t_phase = extra_params['t_phase']
        n_pulse_trunc = extra_params['n_pulse_trunc']
        # get pd0 pdt
        a_coeff = extra_params['pd0pdt'][0]
        pd0 = extra_params['pd0pdt'][1]
        pdt = extra_params['pd0pdt'][2]
        # fes_min = extra_params['fes_min']
        muscle_params = extra_params['muscle_params']

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        name_mus = self.muscle_names

        for m, muscle in enumerate(name_mus):
            tau1 = nlp.parameters["tau_1_param"].cx
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            # a_sc = a_scale * (1 - exp(-(pwd - pd0) / pdt))
            # 1 channel only
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd0)))
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            alpha_dot_list.append(alpha_dot)
            # f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            f_dot_list.append(-f[m]/tau1)
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


class ParamPredictionDynamics(CustomFesDynamics):
    """
    This class is used to create a model actuated through muscle activation.
    """

    def __init__(
            self,
            bio_model: Str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            with_contact: bool = False,
            with_residual_torque: bool = False,
            with_resistive_torque: bool = False,
            parameters: ParameterList = None,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
            contact_types: list[ContactType] | tuple[ContactType] = (),
            fatigue: FatigueList = None,
            **extra_params,
    ):
        super().__init__(
            bio_model=bio_model,
            friction_coefficients=friction_coefficients,
            with_residual_torque=with_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
            contact_types=contact_types,
            parameter_identification=True,
            fatigue=fatigue,
            **extra_params,
        )
        self.with_resistive_torque = with_resistive_torque

    def dynamics(
            self,
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
            with_residual_torque: bool = False,
            fes_driven_task_flag: bool = False,
            prediction_flag: bool = False,
            only_parameter_a: bool = False,
            identification_flag: bool = False,
            **extra_params,
    ):
        # Get variables
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        nlp.model.set_gravity(MX([0, 0, 0]))

        t_phase = extra_params['t_phase']
        n_pulse_trunc = extra_params['n_pulse_trunc']
        a_coeff = extra_params['pd0pdt'][0]
        pd0 = extra_params['pd0pdt'][1]
        pdt = extra_params['pd0pdt'][2]
        # pd0 = extra_params['pd0']
        # pdt = extra_params['pdt']
        # fes_min = extra_params['fes_min']
        muscle_params = extra_params['muscle_params']

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        name_mus = nlp.model.muscle_names

        for m, muscle in enumerate(name_mus):

            coef = muscle_params[0]
            tau1 = muscle_params[1]
            tau2 = muscle_params[2]
            km = muscle_params[3]

            # knee_angle = q[joint_index[m]]
            # x = 1.6009166
            knee_angle = extra_params['knee_angle']
            a_scale = (sum(c * knee_angle ** (len(coef) - 1 - i) for i, c in enumerate(coef))) ** 2
            # a_scale = sum(c * knee_angle ** (len(coef) - 1 - i) for i, c in enumerate(coef))
            # a_scale = casadi.if_else(a_scale > 0, a_scale, 0)
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            # 1 channel only
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd - pd0)))
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])
            a_sc = a_scale * a_coeff * (1 - exp(-(pwd - pd0) / pdt))
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            alpha_dot_list.append(alpha_dot)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


class FESCyclingDynamics(CustomFesDynamics):
    """
    This class is used to create a model actuated through muscle activation.
    """

    def __init__(
            self,
            bio_model: Str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            with_contact: bool = False,
            with_friction: bool = False,
            with_residual_torque: bool = False,
            with_resistive_torque: bool = False,
            parameters: ParameterList = None,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
            contact_types: list[ContactType] | tuple[ContactType] = (),
            fatigue: FatigueList = None,
            **extra_params,
    ):
        super().__init__(
            bio_model=bio_model,
            friction_coefficients=friction_coefficients,
            with_residual_torque=with_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
            contact_types=contact_types,
            parameter_identification=False,
            fatigue=fatigue,
            **extra_params,
        )
        self.with_resistive_torque = with_resistive_torque
        self.with_friction = with_friction

    def dynamics(
            self,
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
            **extra_params,
    ):

        # # Get states indices
        # q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, cn, f, pwd, muscles_tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        t_phase = extra_params['t_phase']
        n_pulse_trunc = extra_params['n_pulse_trunc']
        pd0_pdt_data = extra_params['pd0_pdt_data']
        # pd0 = extra_params['pd0']
        # pdt = extra_params['pdt']
        joint_index = extra_params['joint_idx']
        muscle_params = extra_params['muscle_params']

        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        # alpha_temp = MX.zeros(n_pulse_trunc)
        name_mus = nlp.model.muscle_names

        for m, muscle in enumerate(name_mus):
            # a_scale, tau1, tau2, km = CustomFesDynamics.get_ding_params(nlp.model, m)
            coef = muscle_params[muscle][0]
            tau1 = muscle_params[muscle][1]
            tau2 = muscle_params[muscle][2]  # todo if i add tau_2 it does not converge
            km = muscle_params[muscle][3]

            a_coeff = pd0_pdt_data[muscle][0]
            pd0 = pd0_pdt_data[muscle][1]
            pdt = pd0_pdt_data[muscle][2]

            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd0)))
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])

            x = q[joint_index[m]]
            # x = MX(1.76)
            # todo attention to these coefficients. May extrapolate realistic values!
            a_scale = (sum(c * x ** (len(coef) - 1 - i) for i, c in enumerate(coef))) ** 2
            # a_scale = sum(c * x ** (len(coef) - 1 - i) for i, c in enumerate(coef))
            # a_scale = casadi.if_else(a_scale > 0, a_scale, 0)
            # a_scale = casadi.if_else(a_scale > 20000, 20000, a_scale)
            a_sc = a_scale * a_coeff * (1 - exp(-(pwd[m] - pd0) / pdt))
            # a_sc = casadi.if_else(a_sc > 0, a_sc, 0)

            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)

            alpha_dot_list.append((alpha_temp - alpha) / t_phase)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        #  joints_tau = casadi.MX.zeros(muscles_tau.shape)
        #  todo add friction
        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)


class CustomTorqueDynamics(BiorbdModel, TorqueDynamics):
    """
    """

    def __init__(
            self,
            bio_model: Str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            with_residual_torque: Bool = False,
            with_friction: Bool = False,
            parameters: ParameterList = None,
            parameter_identification: Bool = False,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
            contact_types: list[ContactType] | tuple[ContactType] = (),
            fatigue: FatigueList = None,
            **extra_params,
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueDynamics.__init__(self, fatigue)

        self.state_configuration = [States.Q, States.QDOT]
        self.control_configuration = [Controls.TAU]

        self.algebraic_configuration = []
        self.functions = []
        self.with_residual_torque = with_residual_torque
        self.extra_params = extra_params

    def get_rigid_contact_forces(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
    ):
        q, qdot, tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
        )

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    def get_basic_variables(
            self,
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
    ):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        # CONTROL VARIABLES
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        # extra_params
        idx_crank = self.extra_params.get('idx_crank')
        friction_coef_b = self.extra_params.get('friction_coef_b')
        friction_coef_mu = self.extra_params.get('friction_coef_mu')
        friction_tau = calculate_friction_torque(qdot[idx_crank],
                                                    mu=friction_coef_mu,
                                                    B=friction_coef_b)
        friction_torque = MX.zeros(q.shape)
        friction_torque[idx_crank] = friction_tau
        tau = tau + friction_torque

        return q, qdot, tau



    def dynamics(
            self,
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
            **extra_params,
    ):

        # # Get states indices
        # q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, tau = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, tau, [], [])
        return DynamicsEvaluation(dxdt=vertcat(qdot, ddq), defects=None)
