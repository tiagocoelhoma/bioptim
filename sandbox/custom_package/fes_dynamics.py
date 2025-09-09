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
tauc = 0.011  # 11ms
# pd0 = 0.000086  # average value
# pdt = 0.000138  # average value
# km = 0.180
# tau1 = 0.044099
# tau2 = 0.018522
# a_scale = 5000
# stim_amp = 1
r0 = 5
scaling_factor = 1e6  # <- this impacts on the time to convergence. 1e5 faster but cn dynamics with slower slope


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
    ):
        """
        """
        n_pulse_trunc = 5  # todo fix this
        name_elements = []
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


class FesStates(FcnEnum):
    F = (FESConfigureVariables.configure_fes_f,)
    CN = (FESConfigureVariables.configure_fes_cn,)
    ALPHA = (FESConfigureVariables.configure_fes_alpha,)


class FesControls(FcnEnum):
    PW = (FESConfigureVariables.configure_fes_pw,)
    # RESTORQUE = (FESConfigureVariables.configure_fes_resistive_torque,)


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
            parameters: ParameterList = None,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
            contact_types: list[ContactType] | tuple[ContactType] = (),
            fatigue: FatigueList = None,
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueDynamics.__init__(self, fatigue)

        self.state_configuration = [States.Q, States.QDOT,
                                    FesStates.CN, FesStates.F, FesStates.ALPHA]

        self.control_configuration = [FesControls.PW]
        if with_residual_torque:
            self.control_configuration = [Controls.TAU, FesControls.PW]

        self.algebraic_configuration = []
        self.functions = []
        self.with_residual_torque = with_residual_torque

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
        # PARAMETERS
        t_phase = extra_params.get('t_phase')
        # n_pulse_trunc = extra_parameters.get('n_pulse_trunc')
        # pd0 = extra_parameters.get('pd0')
        # pdt = extra_parameters.get('pdt')
        # fes_min = extra_parameters.get('fes_min')
        # muscle_params = extra_parameters.get('muscle_params')


        # fatigue_states, mus_activations = DynamicsFunctions.get_fatigue_states(states, nlp, mus_activations)
        #
        # # Compute the torques due to muscles
        # muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)
        #
        # # Add additional torques
        # if self.with_residual_torque:
        #     muscles_tau += DynamicsFunctions.get_fatigable_tau(nlp, states, controls)
        # muscles_tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)
        #
        # # Get external forces
        # external_forces = nlp.get_external_forces(
        #     "external_forces", states, controls, algebraic_states, numerical_timeseries
        # )
        # return q, qdot, muscles_tau, external_forces, mus_activations
        return q, qdot, cn, f, pwd


    def muscle_joint_torque_from_muscle_forces(self, forces: MX, q: MX, qdot: MX) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(forces, q_biorbd, qdot_biorbd).to_mx()


    def dynamics(
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

        # Get states indices
        q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, cn, f, pwd = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            **extra_params,
        )

        t_phase = 1/35
        num_t_phase = casadi.floor(time / t_phase)  # geting the time of the last pulse
        ri = casadi.if_else(num_t_phase, 1 + (r0 - 1) * exp(-t_phase / tauc), 1)

        alpha_dot_list, f_dot_list, cn_dot_list = [], [], []
        name_mus = nlp.model.muscle_names

        # todo erase
        pd0 = 0.12
        muscle_params = [2500, 0.01, 0.02, 0.5]
        only_parameter_a = True
        n_pulse_trunc = 5
        pdt = 0.011


        for m, muscle in enumerate(name_mus):
            # see Modeling the length dependence of isometric force in human quadriceps muscles (perumal 2002)
            # see Mathematical models for fatigue minimization during functional electrical stimulation (ding 2003)
            # a_scale, tau1, tau2, km = CustomDynamicFunctions.get_ding_params(nlp.model, m)
            tau1 = muscle_params[1]
            tau2 = muscle_params[2]
            km = muscle_params[3]
            # a_scale = nlp.parameters["a_scale_param"].cx
            a_scale = muscle_params[0] #todo erase
            if not only_parameter_a:
                tau1 = nlp.parameters["tau_1_param"].cx
                tau2 = nlp.parameters["tau_2_param"].cx
                km = nlp.parameters["km_param"].cx
            alpha = DynamicsFunctions.get(nlp.states[f"alpha_{m}"], states)
            # 1 channel only
            a = 1.0 / (1.0 + exp(-scaling_factor * (pwd[m] - pd0)))     #todo verify pwd[m] or pwd 1channel or not
            alpha_temp = casadi.vertcat((a - 0.5) / 0.5, alpha[:n_pulse_trunc - 1])
            a_sc = a_scale * (1 - exp(-(pwd[m] - pd0) / pdt))
            # a_sc = a_scale / (1 + exp(-(pwd - pd0) / pdt))
            # offset = a_scale / (1 + np.exp(-(fes_min - pd0) / pdt))
            # a_sc = a_scale / (1 + exp(-(pwd - pd0) / pdt)) - offset
            # a_sc = a_scale / (1 + exp(-(pwd - pd0) / pdt))
            alpha_dot = (alpha_temp - alpha) / t_phase
            sum_aux = 0
            for i in range(n_pulse_trunc):
                ti = (num_t_phase - i) * t_phase
                sum_aux += ri * alpha_temp[i] * casadi.exp(-(time - ti) / tauc)
            alpha_dot_list.append(alpha_dot)
            f_dot_list.append(a_sc * (cn[m] / (km + cn[m])) - (f[m] / (tau1 + tau2 * (cn[m] / (km + cn[m])))))
            cn_dot_list.append((1 / tauc) * sum_aux - (cn[m] / tauc))

        # muscles_tau = self.muscle_joint_torque_from_muscle_forces(f, q, qdot)
        # muscles_tau = muscles_tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else muscles_tau
        # muscles_tau = muscles_tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else muscles_tau
        # joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if with_orque else joints_tau
        # todo muscles_tau[0, :] = muscles_tau[0, :] + DynamicsFunctions.get(nlp.controls["resistive_torque"], controls) if with_resistive_torque else muscles_tau

        muscles_tau = MX(0)
        with_contact = True
        # ddq = nlp.model.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])
        ddq = self.forward_dynamics(with_contact=with_contact)(q, qdot, muscles_tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(qdot,
                                               ddq,
                                               casadi.vertcat(*cn_dot_list),
                                               casadi.vertcat(*f_dot_list),
                                               casadi.vertcat(*alpha_dot_list)),
                                  defects=None)
        # return DynamicsEvaluation(
        #     dxdt=vertcat(states[1], self.forward_dynamics(with_contact=False)(states[0], states[1], controls[0], [])),
        #     defects=None,
        # )


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
        q, qdot, tau, external_forces, _ = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)


